#!/usr/bin/env python3
"""Cache-aware streaming latency harness for NeMo nemotron-streaming.

Drives an ASR model through `conformer_stream_step` chunk-by-chunk while
sleeping between steps so that step `k` fires at wall-clock time equal to the
audio duration through that step. This mirrors SAPC2's real-time arrival
model (100 ms input chunks at 100 ms intervals): the model cannot run a step
until enough audio has accumulated for one *model* chunk (1050 ms first then
1120 ms each subsequent for nemotron `[70,13]`).

Outputs:
  --out-csv          : (id, raw_hypos) prediction CSV (final streaming text)
  --out-partial-json : `<split>.partial_results.json` in the format expected
                       by `utils/compute_latency.py`

Usage:
  python run_nemo_streaming_latency.py \
      --model nvidia/nemotron-speech-streaming-en-0.6b \
      --manifest /workspace/SAPC2/manifest/Dev_streaming.csv \
      --audio-root /workspace/SAPC2 \
      --device cpu \
      --out-csv hyps/R1_streaming_Dev_streaming.csv \
      --out-partial-json hyps/R1_streaming_Dev_streaming.partial_results.json
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
from typing import Optional

import numpy as np
import torch
from omegaconf import open_dict

import nemo.collections.asr as nemo_asr  # noqa: F401  (registers model classes)
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

SAMPLE_RATE = 16000
WINDOW_STRIDE_SEC = 0.01  # 10ms mel hop


def read_wav_mono16k(path: str) -> np.ndarray:
    with wave.open(path, "rb") as f:
        if f.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: expected {SAMPLE_RATE}Hz, got {f.getframerate()}")
        if f.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono, got {f.getnchannels()} channels")
        if f.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit, got {f.getsampwidth() * 8}-bit")
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def calc_drop_extra_pre_encoded(asr_model, step_num: int, pad_and_drop_preencoded: bool) -> int:
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded


def extract_text(transcribed) -> str:
    if not transcribed:
        return ""
    item = transcribed[0]
    if isinstance(item, Hypothesis):
        return item.text or ""
    return str(item) if item is not None else ""


def configure_model(model, device: str):
    """Set greedy decoding strategy and move model to device."""
    decoding_cfg = model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy"
        decoding_cfg.preserve_alignments = False
        if hasattr(model, "joint"):  # RNNT
            decoding_cfg.greedy.max_symbols = 10
            decoding_cfg.fused_batch_size = -1
        model.change_decoding_strategy(decoding_cfg)
    model = model.to(device)
    model.eval()
    return model


def stream_one_utterance(model, audio_path: str, device: str, realtime: bool, pad_and_drop_preencoded: bool):
    """Stream a single utterance and return (final_text, partial_record)."""
    samples = read_wav_mono16k(audio_path)
    audio_dur = len(samples) / SAMPLE_RATE

    # Build a streaming buffer for just this utterance.
    buffer = CacheAwareStreamingAudioBuffer(
        model=model,
        online_normalization=False,  # nemotron streaming has no preprocessor normalization
        pad_and_drop_preencoded=pad_and_drop_preencoded,
    )
    # Append from already-loaded samples (avoid second wav read).
    buffer.append_audio(samples, stream_id=-1)

    streaming_cfg = model.encoder.streaming_cfg
    chunk_first_frames = (
        streaming_cfg.chunk_size[0] if isinstance(streaming_cfg.chunk_size, list) else streaming_cfg.chunk_size
    )
    chunk_subseq_frames = (
        streaming_cfg.chunk_size[1] if isinstance(streaming_cfg.chunk_size, list) else streaming_cfg.chunk_size
    )

    # Initial cache.
    cache_last_channel, cache_last_time, cache_last_channel_len = model.encoder.get_initial_cache_state(batch_size=1)

    previous_hypotheses = None
    pred_out_stream = None
    events = []
    final_text = ""

    audio_send_start_time = time.time()
    audio_end_oracle_time = audio_send_start_time + audio_dur

    audio_consumed_sec = 0.0  # Cumulative audio (sec) consumed by the model after each step

    autocast = (lambda: contextlib.nullcontext()) if device == "cpu" else torch.cuda.amp.autocast

    last_step_index = -1
    for step_num, (chunk_audio, chunk_lengths) in enumerate(buffer):
        # Compute when this chunk's audio is fully "available" in real time.
        if step_num == 0:
            audio_consumed_sec = chunk_first_frames * WINDOW_STRIDE_SEC
        else:
            audio_consumed_sec += chunk_subseq_frames * WINDOW_STRIDE_SEC

        if realtime:
            arrival_wall = audio_send_start_time + audio_consumed_sec
            now = time.time()
            if arrival_wall > now:
                time.sleep(arrival_wall - now)

        with torch.inference_mode():
            with autocast():
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                ) = model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=buffer.is_buffer_empty(),
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out_stream,
                    drop_extra_pre_encoded=calc_drop_extra_pre_encoded(model, step_num, pad_and_drop_preencoded),
                    return_transcription=True,
                )
        text = extract_text(transcribed_texts)
        events.append({"event": "partial", "time": time.time(), "text": text})
        final_text = text
        last_step_index = step_num

    final_visible_time = time.time()
    events.append({"event": "final_visible", "time": final_visible_time, "text": final_text})

    record = {
        "events": events,
        "timing": {
            "audio_send_start_time": audio_send_start_time,
            "audio_end_oracle_time": audio_end_oracle_time,
            "final_visible_time": final_visible_time,
            "first_partial_time": events[0]["time"] if events else None,
            "audio_dur_sec": audio_dur,
            "n_steps": last_step_index + 1,
        },
    }
    return final_text, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local .nemo path")
    ap.add_argument("--manifest", required=True, help="Streaming manifest CSV (id, audio_filepath, ...)")
    ap.add_argument("--audio-root", required=True, help="Prefix for audio_filepath column")
    ap.add_argument("--out-csv", required=True, help="Output prediction CSV (id, raw_hypos)")
    ap.add_argument("--out-partial-json", required=True, help="Output partial_results JSON")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--realtime", action="store_true", help="Sleep between steps to mirror real-time arrival")
    ap.add_argument("--pad-and-drop-preencoded", action="store_true")
    ap.add_argument("--att-context-size", default=None,
                    help="Override encoder att_context_size, e.g. '[70,1]' (JSON list)")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N rows (smoke test)")
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (CPU only)")
    args = ap.parse_args()

    if args.device == "cpu" and args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Load model.
    print(f"[load] {args.model}", flush=True)
    if args.model.endswith(".nemo") and os.path.isfile(args.model):
        model = ASRModel.restore_from(args.model, map_location=args.device)
    else:
        model = ASRModel.from_pretrained(args.model, map_location=args.device)
    if args.att_context_size is not None:
        if hasattr(model.encoder, "set_default_att_context_size"):
            acs = json.loads(args.att_context_size)
            model.encoder.set_default_att_context_size(att_context_size=acs)
            print(f"[att_context_size] override -> {acs}", flush=True)
        else:
            raise SystemExit("model encoder does not support set_default_att_context_size")
    model = configure_model(model, args.device)
    print(f"[encoder.att_context_size] {model.encoder.att_context_size}", flush=True)
    print(f"[encoder.streaming_cfg] {model.encoder.streaming_cfg}", flush=True)

    # Read manifest.
    rows = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"[manifest] {args.manifest}: {len(rows)} rows", flush=True)

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_partial_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    partial_results = {}
    predictions = []

    t_start = time.time()
    total_audio_sec = 0.0
    for i, row in enumerate(rows, 1):
        uid = row["id"]
        audio_path = os.path.join(args.audio_root, row["audio_filepath"])
        if not os.path.isfile(audio_path):
            print(f"[skip] missing audio: {audio_path}", flush=True)
            predictions.append((uid, ""))
            continue
        try:
            final_text, record = stream_one_utterance(
                model, audio_path,
                device=args.device,
                realtime=args.realtime,
                pad_and_drop_preencoded=args.pad_and_drop_preencoded,
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
            print(
                f"[{i}/{len(rows)}] elapsed={elapsed:.1f}s audio={total_audio_sec:.1f}s "
                f"RTF={rtf:.3f} last_steps={record['timing']['n_steps']}",
                flush=True,
            )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "raw_hypos"])
        for uid, text in predictions:
            writer.writerow([uid, text])
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(partial_results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    rtf = elapsed / total_audio_sec if total_audio_sec > 0 else float("nan")
    print(f"[done] {len(predictions)} utts  audio={total_audio_sec:.1f}s  wall={elapsed:.1f}s  RTF={rtf:.3f}")
    print(f"[done] csv={out_csv}")
    print(f"[done] partial={out_json}")


if __name__ == "__main__":
    sys.exit(main())
