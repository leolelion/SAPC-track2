#!/usr/bin/env python3
"""ONNX Runtime streaming latency harness for danielbodart/nemotron-speech-600m-onnx.

Mirrors `run_nemo_streaming_latency.py` but the encoder + decoder are ONNX
sessions (CPU EP). Mel-spectrogram preprocessing is done with NeMo's
preprocessor for exact numeric match against the FP32 reference (this benchmark
isolates the int8 *encoder/decoder* pareto change; the preprocessor latency is
small and not the variable under test).

Per the danielbodart export, the chunk size is fixed:
  - 56 mel frames new per step (560 ms audio)
  - 9 cache mel frames from previous step (zeros for first step)
  - 65 mel frames per encoder forward
  - encoder produces ~7 valid encoder frames per chunk, RNNT-decoded greedy

Outputs match the SAPC2 partial_results.json format.

Usage:
  python run_onnx_streaming_latency.py \
      --encoder /path/int8-dynamic/encoder_model.onnx \
      --decoder /path/int8-dynamic/decoder_model.onnx \
      --tokens  /path/shared/tokens.txt \
      --manifest /workspace/SAPC2/manifest/Dev_streaming.csv \
      --audio-root /workspace/SAPC2 \
      --realtime --threads 4 \
      --out-csv hyps/R1_int8_streaming.csv \
      --out-partial-json hyps/R1_int8_streaming.partial_results.json
"""
import argparse
import contextlib
import csv
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
import torch

import nemo.collections.asr as nemo_asr  # noqa: F401
from nemo.collections.asr.models import ASRModel

SAMPLE_RATE = 16000
WINDOW_STRIDE_SEC = 0.01
CHUNK_NEW = 56          # new mel frames per encoder step
CACHE_FRAMES = 9        # pre-encoder cache mel frames
ENC_INPUT_FRAMES = CHUNK_NEW + CACHE_FRAMES  # 65
N_MELS = 128
ENC_LAYERS = 24
ENC_DIM = 1024
LAST_CHANNEL_CACHE = 70  # encoder frames of left context cache
LAST_TIME_FRAMES = 8
DEC_HIDDEN = 640
DEC_LAYERS = 2
BLANK_ID = 1024


def read_wav_mono16k(path: str) -> np.ndarray:
    with wave.open(path, "rb") as f:
        if f.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: expected {SAMPLE_RATE}Hz, got {f.getframerate()}")
        if f.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono")
        if f.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit")
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def load_tokens(path: str) -> List[str]:
    """Load nemotron sentencepiece vocab. Lines look like '▁the 5'."""
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").rsplit(" ", 1)
            if len(parts) != 2:
                continue
            piece, idx = parts
            try:
                vocab[int(idx)] = piece
            except ValueError:
                continue
    return [vocab[i] for i in range(max(vocab) + 1)]


def detokenize(tokens: List[int], vocab: List[str]) -> str:
    pieces = [vocab[t] for t in tokens if t < len(vocab)]
    text = "".join(pieces).replace("\u2581", " ").strip()
    return text


def make_session(path: str, num_threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    if num_threads > 0:
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def stream_one_utterance(
    samples: np.ndarray,
    preprocessor,
    encoder_session: ort.InferenceSession,
    decoder_session: ort.InferenceSession,
    vocab: List[str],
    realtime: bool,
):
    audio_dur = len(samples) / SAMPLE_RATE

    # Offline mel preprocessing.
    audio_t = torch.from_numpy(samples).unsqueeze(0)
    length_t = torch.tensor([len(samples)], dtype=torch.long)
    with torch.inference_mode():
        mel, mel_len = preprocessor(input_signal=audio_t, length=length_t)
    mel_np = mel[0].numpy().astype(np.float32)  # [128, T]
    T = int(mel_len.item())

    # Init encoder caches.
    cache_lc = np.zeros((1, ENC_LAYERS, LAST_CHANNEL_CACHE, ENC_DIM), dtype=np.float32)
    cache_lt = np.zeros((1, ENC_LAYERS, ENC_DIM, LAST_TIME_FRAMES), dtype=np.float32)
    cache_ll = np.zeros((1,), dtype=np.int64)

    # Init decoder LSTM states.
    dec_state_h = np.zeros((DEC_LAYERS, 1, DEC_HIDDEN), dtype=np.float32)
    dec_state_c = np.zeros((DEC_LAYERS, 1, DEC_HIDDEN), dtype=np.float32)
    last_token = np.array([[0]], dtype=np.int32)
    target_length = np.array([1], dtype=np.int32)

    text_tokens: List[int] = []
    events = []
    audio_send_start = time.time()
    audio_end_oracle = audio_send_start + audio_dur

    pos = 0
    step_num = 0
    drain_done = False
    while True:
        if pos >= T:
            if drain_done:
                break
            # Drain step: feed one extra zero chunk so the encoder's
            # right-context can resolve the final frames of real audio.
            new_chunk = np.zeros((N_MELS, CHUNK_NEW), dtype=np.float32)
            new_end = T  # for cache_pre slicing
            drain_done = True
        else:
            new_end = min(pos + CHUNK_NEW, T)
            new_chunk = mel_np[:, pos:new_end]
            if new_chunk.shape[1] < CHUNK_NEW:
                pad = np.zeros((N_MELS, CHUNK_NEW - new_chunk.shape[1]), dtype=np.float32)
                new_chunk = np.concatenate([new_chunk, pad], axis=1)

        # Build cache prefix (last 9 mel frames before the new chunk).
        if drain_done:
            # During drain, cache_pre is the last 9 real mel frames (from end of audio).
            start_cache = max(0, T - CACHE_FRAMES)
            cache_pre = mel_np[:, start_cache:T]
            if cache_pre.shape[1] < CACHE_FRAMES:
                pad = np.zeros((N_MELS, CACHE_FRAMES - cache_pre.shape[1]), dtype=np.float32)
                cache_pre = np.concatenate([pad, cache_pre], axis=1)
        elif pos == 0:
            cache_pre = np.zeros((N_MELS, CACHE_FRAMES), dtype=np.float32)
        else:
            start_cache = max(0, pos - CACHE_FRAMES)
            cache_pre = mel_np[:, start_cache:pos]
            if cache_pre.shape[1] < CACHE_FRAMES:
                pad = np.zeros((N_MELS, CACHE_FRAMES - cache_pre.shape[1]), dtype=np.float32)
                cache_pre = np.concatenate([pad, cache_pre], axis=1)

        chunk_input = np.concatenate([cache_pre, new_chunk], axis=1)[None, :, :]  # [1, 128, 65]
        chunk_length = np.array([ENC_INPUT_FRAMES], dtype=np.int64)

        # Real-time pacing: this chunk's audio is fully available at audio time pos+CHUNK_NEW frames
        # (i.e. (pos+CHUNK_NEW)*10ms wall-clock relative to audio_send_start).
        audio_consumed_sec = min(new_end, T) * WINDOW_STRIDE_SEC
        if pos + CHUNK_NEW > T:
            # Padded last chunk: still treat the audio as "ended" at T*10ms
            audio_consumed_sec = T * WINDOW_STRIDE_SEC
        if realtime:
            arrival_wall = audio_send_start + audio_consumed_sec
            now = time.time()
            if arrival_wall > now:
                time.sleep(arrival_wall - now)

        # Run encoder.
        enc_outs = encoder_session.run(
            None,
            {
                "audio_signal": chunk_input,
                "length": chunk_length,
                "cache_last_channel": cache_lc,
                "cache_last_time": cache_lt,
                "cache_last_channel_len": cache_ll,
            },
        )
        encoder_out = enc_outs[0]            # [1, 1024, n_enc]
        encoded_lengths = enc_outs[1]        # [1]
        cache_lc = enc_outs[2]
        cache_lt = enc_outs[3]
        cache_ll = enc_outs[4]
        n_enc = int(encoded_lengths[0])

        # RNNT greedy decode over the new encoder frames.
        for f_idx in range(n_enc):
            enc_frame = encoder_out[:, :, f_idx : f_idx + 1]  # [1, 1024, 1]
            for _sym in range(10):
                dec_outs = decoder_session.run(
                    None,
                    {
                        "encoder_outputs": enc_frame,
                        "targets": last_token,
                        "target_length": target_length,
                        "input_states_1": dec_state_h,
                        "input_states_2": dec_state_c,
                    },
                )
                logits = dec_outs[0]  # [1, 1, 1, 1025]
                token = int(np.argmax(logits[0, 0, 0]))
                if token == BLANK_ID:
                    break
                text_tokens.append(token)
                last_token = np.array([[token]], dtype=np.int32)
                dec_state_h = dec_outs[2]
                dec_state_c = dec_outs[3]

        text = detokenize(text_tokens, vocab)
        events.append({"event": "partial", "time": time.time(), "text": text})
        if not drain_done:
            pos += CHUNK_NEW
        step_num += 1

    final_text = detokenize(text_tokens, vocab)
    final_visible_time = time.time()
    events.append({"event": "final_visible", "time": final_visible_time, "text": final_text})

    record = {
        "events": events,
        "timing": {
            "audio_send_start_time": audio_send_start,
            "audio_end_oracle_time": audio_end_oracle,
            "final_visible_time": final_visible_time,
            "first_partial_time": events[0]["time"] if events else None,
            "audio_dur_sec": audio_dur,
            "n_steps": step_num,
        },
    }
    return final_text, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, help="encoder ONNX path")
    ap.add_argument("--decoder", required=True, help="decoder ONNX path")
    ap.add_argument("--tokens", required=True, help="shared/tokens.txt")
    ap.add_argument("--preprocessor-source", default="nvidia/nemotron-speech-streaming-en-0.6b",
                    help="HF id or .nemo path to load preprocessor from")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-partial-json", required=True)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)

    print(f"[load preprocessor] {args.preprocessor_source}", flush=True)
    src = args.preprocessor_source
    if src.endswith(".nemo") and os.path.isfile(src):
        ref_model = ASRModel.restore_from(src, map_location="cpu")
    else:
        ref_model = ASRModel.from_pretrained(src, map_location="cpu")
    # Match CacheAwareStreamingAudioBuffer.extract_preprocessor: disable dither + pad_to
    # so the mel features are deterministic and bit-equivalent to NeMo's streaming path.
    from omegaconf import OmegaConf, open_dict
    cfg = ref_model._cfg
    OmegaConf.set_struct(cfg.preprocessor, False)
    with open_dict(cfg.preprocessor):
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
    preprocessor = ref_model.from_config_dict(cfg.preprocessor).to("cpu")
    preprocessor.eval()
    print(f"[preprocessor] dither=0.0 pad_to=0", flush=True)

    print(f"[load encoder] {args.encoder}", flush=True)
    encoder_session = make_session(args.encoder, args.threads)
    print(f"[load decoder] {args.decoder}", flush=True)
    decoder_session = make_session(args.decoder, args.threads)

    print(f"[load tokens] {args.tokens}", flush=True)
    vocab = load_tokens(args.tokens)
    print(f"[vocab] {len(vocab)} tokens", flush=True)

    rows = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"[manifest] {args.manifest}: {len(rows)} rows", flush=True)

    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_partial_json); out_json.parent.mkdir(parents=True, exist_ok=True)

    partial_results = {}
    predictions = []
    t_start = time.time()
    total_audio_sec = 0.0

    for i, row in enumerate(rows, 1):
        uid = row["id"]
        audio_path = os.path.join(args.audio_root, row["audio_filepath"])
        if not os.path.isfile(audio_path):
            predictions.append((uid, ""))
            continue
        try:
            samples = read_wav_mono16k(audio_path)
            final_text, record = stream_one_utterance(
                samples, preprocessor, encoder_session, decoder_session, vocab,
                realtime=args.realtime,
            )
        except Exception as e:
            print(f"[error] {uid}: {e}", flush=True)
            predictions.append((uid, ""))
            continue
        partial_results[uid] = record
        predictions.append((uid, final_text))
        total_audio_sec += record["timing"]["audio_dur_sec"]
        if i % 10 == 0 or i == len(rows):
            elapsed = time.time() - t_start
            rtf = elapsed / total_audio_sec if total_audio_sec > 0 else float("nan")
            print(f"[{i}/{len(rows)}] elapsed={elapsed:.1f}s audio={total_audio_sec:.1f}s "
                  f"RTF={rtf:.3f} last_steps={record['timing']['n_steps']}", flush=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "raw_hypos"])
        for uid, txt in predictions:
            w.writerow([uid, txt])
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(partial_results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    rtf = elapsed / total_audio_sec if total_audio_sec > 0 else float("nan")
    print(f"[done] {len(predictions)} utts  audio={total_audio_sec:.1f}s  wall={elapsed:.1f}s  RTF={rtf:.3f}")
    print(f"[done] csv={out_csv}")
    print(f"[done] partial={out_json}")


if __name__ == "__main__":
    sys.exit(main())
