#!/usr/bin/env python3
"""Phase 4c — TEMPORARY DIAGNOSTIC. NOT shipped.

Re-runs the Nemotron streaming inference on a list of utterance IDs with
two modes:

  --mode normal      → standard greedy RNN-T decode (same as shipped model.py)
  --mode force_nonblank → suppress BLANK in greedy decode; at each encoder
                         frame, emit exactly one non-blank argmax then move
                         on (1 token per frame, no MAX_SYMBOLS_PER_FRAME loop)

Purpose: measure what the model would say on the empty subset if forced
to commit to a token at every frame. Used to validate that empty outputs
were the model declining to guess — by showing that forcing produces
mostly garbage and CER stays at the noise floor.

This script intentionally duplicates the streaming-loop logic from
`track2_starting_kit/nemotron_streaming/model.py` so it can be modified
without risk of leaking into the shipped submission.
"""
import argparse
import csv
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
N_MELS = 128
HOP_SAMPLES = 160
WINDOW_SAMPLES = 400
CHUNK_NEW = 56
CACHE_FRAMES = 9
ENC_LAYERS = 24
ENC_DIM = 1024
LAST_CHANNEL_CACHE = 70
LAST_TIME_FRAMES = 8
DEC_LAYERS = 2
DEC_HIDDEN = 640
VOCAB_SIZE = 1025
BLANK_ID = 1024
MAX_SYMBOLS_PER_FRAME = 10
CHUNK_SIZE_INPUT = 1600  # 100 ms at 16 kHz


# ── Tokenizer + preprocessor utilities (copied from model.py) ──────────────

def _load_tokens(path: str):
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            piece, _, idx = line.rpartition(" ")
            try:
                vocab[int(idx)] = piece
            except ValueError:
                continue
    return [vocab[i] for i in range(max(vocab) + 1)]


def _detokenize(tokens, vocab):
    pieces = [vocab[t] for t in tokens if 0 <= t < len(vocab)]
    return "".join(pieces).replace("▁", " ").strip()


def _load_preprocessor():
    ref = ASRModel.from_pretrained(
        "nvidia/nemotron-speech-streaming-en-0.6b", map_location="cpu"
    )
    pp = ref.preprocessor
    pp.featurizer.dither = 0.0
    pp.featurizer.pad_to = 0
    pp.eval()
    return pp


def _make_session(path: str, num_threads: int):
    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def _read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as f:
        assert f.getframerate() == SAMPLE_RATE
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# ── DiagnosticModel: shipped streaming logic + optional blank suppression ─

class DiagnosticModel:
    def __init__(self, encoder_path, decoder_path, tokens_path, num_threads,
                 force_nonblank: bool):
        self.force_nonblank = force_nonblank
        self._enc = _make_session(encoder_path, num_threads)
        self._dec = _make_session(decoder_path, num_threads)
        self._enc_out_names = [o.name for o in self._enc.get_outputs()]
        self._dec_out_names = [o.name for o in self._dec.get_outputs()]
        self._vocab = _load_tokens(tokens_path)
        self._preprocessor = _load_preprocessor()
        self._reset_state()

    def _reset_state(self):
        self._raw_chunks = []
        self._total_samples = 0
        self._buffer_idx = 0
        self._step_num = 0
        self._emitted_tokens = []
        self._cached_mel = None
        self._cached_feat_len = 0
        self._cached_n_samples = 0
        self._drain_done = False
        self._cache_lc = np.zeros((1, ENC_LAYERS, LAST_CHANNEL_CACHE, ENC_DIM), dtype=np.float32)
        self._cache_lt = np.zeros((1, ENC_LAYERS, ENC_DIM, LAST_TIME_FRAMES), dtype=np.float32)
        self._cache_ll = np.zeros((1,), dtype=np.int64)
        self._dec_h = np.zeros((DEC_LAYERS, 1, DEC_HIDDEN), dtype=np.float32)
        self._dec_c = np.zeros((DEC_LAYERS, 1, DEC_HIDDEN), dtype=np.float32)
        self._last_token = np.array([[0]], dtype=np.int32)
        self._target_length = np.array([1], dtype=np.int32)

    @staticmethod
    def _min_samples_for_frames(n_frames: int) -> int:
        if n_frames <= 0:
            return 0
        return (n_frames - 1) * HOP_SAMPLES + WINDOW_SAMPLES

    def _ensure_features(self):
        if not self._raw_chunks or self._cached_n_samples == self._total_samples:
            return
        all_audio = np.concatenate(self._raw_chunks)
        audio_t = torch.from_numpy(all_audio).unsqueeze(0)
        length_t = torch.tensor([len(all_audio)], dtype=torch.long)
        with torch.inference_mode():
            mel, mel_len = self._preprocessor(input_signal=audio_t, length=length_t)
        self._cached_mel = mel[0].numpy().astype(np.float32)
        self._cached_feat_len = int(mel_len.item())
        self._cached_n_samples = self._total_samples

    def _run_steps(self, is_final: bool):
        if not self._raw_chunks:
            return
        if not is_final and self._total_samples < self._min_samples_for_frames(
            self._buffer_idx + CHUNK_NEW
        ):
            return
        self._ensure_features()
        T = self._cached_feat_len
        if T <= 0:
            return
        while True:
            chunk_end = self._buffer_idx + CHUNK_NEW
            if chunk_end <= T:
                mel_chunk = self._cached_mel[:, self._buffer_idx:chunk_end]
                valid_in = CHUNK_NEW
            elif is_final and self._buffer_idx < T:
                remaining = T - self._buffer_idx
                if remaining <= 0:
                    break
                mel_real = self._cached_mel[:, self._buffer_idx:T]
                pad = np.zeros((N_MELS, CHUNK_NEW - remaining), dtype=np.float32)
                mel_chunk = np.concatenate([mel_real, pad], axis=1)
                valid_in = remaining
            else:
                break
            if self._step_num == 0:
                cache_pre = np.zeros((N_MELS, CACHE_FRAMES), dtype=np.float32)
            else:
                start_c = max(0, self._buffer_idx - CACHE_FRAMES)
                cache_pre = self._cached_mel[:, start_c:self._buffer_idx]
                if cache_pre.shape[1] < CACHE_FRAMES:
                    pad = np.zeros((N_MELS, CACHE_FRAMES - cache_pre.shape[1]), dtype=np.float32)
                    cache_pre = np.concatenate([pad, cache_pre], axis=1)
            chunk_input = np.concatenate([cache_pre, mel_chunk], axis=1)[None, :, :].astype(np.float32)
            chunk_length = np.array([CACHE_FRAMES + valid_in], dtype=np.int64)
            self._encode_and_decode(chunk_input, chunk_length)
            self._buffer_idx += CHUNK_NEW
            self._step_num += 1
            if is_final and chunk_end >= T:
                break

    def _run_drain(self):
        if self._drain_done or self._step_num == 0:
            return
        T = self._cached_feat_len
        start_c = max(0, T - CACHE_FRAMES)
        cache_real = self._cached_mel[:, start_c:T]
        if cache_real.shape[1] < CACHE_FRAMES:
            pad = np.zeros((N_MELS, CACHE_FRAMES - cache_real.shape[1]), dtype=np.float32)
            cache_real = np.concatenate([pad, cache_real], axis=1)
        new_chunk = np.zeros((N_MELS, CHUNK_NEW), dtype=np.float32)
        chunk_input = np.concatenate([cache_real, new_chunk], axis=1)[None, :, :].astype(np.float32)
        chunk_length = np.array([CACHE_FRAMES + CHUNK_NEW], dtype=np.int64)
        self._encode_and_decode(chunk_input, chunk_length)
        self._drain_done = True

    def _encode_and_decode(self, chunk_input, chunk_length):
        enc_outs = self._enc.run(
            None,
            {
                "audio_signal": chunk_input,
                "length": chunk_length,
                "cache_last_channel": self._cache_lc,
                "cache_last_time": self._cache_lt,
                "cache_last_channel_len": self._cache_ll,
            },
        )
        named = dict(zip(self._enc_out_names, enc_outs))
        encoder_out = named["outputs"]
        n_enc = int(named["encoded_lengths"][0])
        for n, v in named.items():
            ln = n.lower()
            if "cache_last_channel" in ln and ("_len" in ln or "_length" in ln):
                self._cache_ll = v
            elif "cache_last_channel" in ln and "next" in ln:
                self._cache_lc = v
            elif "cache_last_time" in ln and "next" in ln:
                self._cache_lt = v

        for f_idx in range(n_enc):
            enc_frame = encoder_out[:, :, f_idx:f_idx + 1]
            if self.force_nonblank:
                # Diagnostic mode: emit exactly one non-blank token per encoder
                # frame. No MAX_SYMBOLS_PER_FRAME inner loop.
                dec_outs = self._dec.run(
                    None,
                    {
                        "encoder_outputs": enc_frame,
                        "targets": self._last_token,
                        "target_length": self._target_length,
                        "input_states_1": self._dec_h,
                        "input_states_2": self._dec_c,
                    },
                )
                dnamed = dict(zip(self._dec_out_names, dec_outs))
                logits = dnamed["outputs"][0, 0, 0]
                token = int(np.argmax(logits[:BLANK_ID]))  # restrict to non-blank
                self._emitted_tokens.append(token)
                self._last_token = np.array([[token]], dtype=np.int32)
                self._dec_h = dnamed["output_states_1"]
                self._dec_c = dnamed["output_states_2"]
            else:
                # Normal greedy RNN-T (shipped behavior)
                for _sym in range(MAX_SYMBOLS_PER_FRAME):
                    dec_outs = self._dec.run(
                        None,
                        {
                            "encoder_outputs": enc_frame,
                            "targets": self._last_token,
                            "target_length": self._target_length,
                            "input_states_1": self._dec_h,
                            "input_states_2": self._dec_c,
                        },
                    )
                    dnamed = dict(zip(self._dec_out_names, dec_outs))
                    logits = dnamed["outputs"][0, 0, 0]
                    token = int(np.argmax(logits))
                    if token == BLANK_ID:
                        break
                    self._emitted_tokens.append(token)
                    self._last_token = np.array([[token]], dtype=np.int32)
                    self._dec_h = dnamed["output_states_1"]
                    self._dec_c = dnamed["output_states_2"]

    def transcribe(self, audio: np.ndarray) -> str:
        self._reset_state()
        for s in range(0, len(audio), CHUNK_SIZE_INPUT):
            self._raw_chunks.append(audio[s:s + CHUNK_SIZE_INPUT])
            self._total_samples += min(CHUNK_SIZE_INPUT, len(audio) - s)
            self._run_steps(is_final=False)
        self._run_steps(is_final=True)
        self._run_drain()
        text = _detokenize(self._emitted_tokens, self._vocab)
        return " ".join(text.split())


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--utt-ids", required=True,
                    help="CSV with column 'id' listing utterance IDs to decode")
    ap.add_argument("--manifest", required=True,
                    help="Full manifest with audio_filepath column")
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--mode", choices=["normal", "force_nonblank"], default="force_nonblank")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--heartbeat", default=None)
    ap.add_argument("--heartbeat-interval", type=int, default=50)
    args = ap.parse_args()

    print(f"[phase4c] mode={args.mode} threads={args.threads}", flush=True)
    print(f"[phase4c] encoder: {args.encoder}", flush=True)
    print(f"[phase4c] loading model ...", flush=True)
    t0 = time.time()
    model = DiagnosticModel(args.encoder, args.decoder, args.tokens, args.threads,
                            force_nonblank=(args.mode == "force_nonblank"))
    print(f"[phase4c] model load: {time.time()-t0:.1f}s", flush=True)

    # Load target utt IDs (the empties)
    target_ids = []
    with open(args.utt_ids) as f:
        for r in csv.DictReader(f):
            target_ids.append(r["id"])
    target_set = set(target_ids)
    print(f"[phase4c] {len(target_ids)} target utts", flush=True)

    # Map id → audio_filepath
    audio_by_id = {}
    with open(args.manifest) as f:
        for r in csv.DictReader(f):
            if r["id"] in target_set:
                audio_by_id[r["id"]] = r["audio_filepath"]

    rows = [(uid, audio_by_id[uid]) for uid in target_ids if uid in audio_by_id]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[phase4c] resolved {len(rows)} audio paths", flush=True)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "raw_hypos"])

    sweep_start = time.time()
    for i, (uid, rel) in enumerate(rows):
        path = os.path.join(args.audio_root, rel)
        samples = _read_wav(path)
        hyp = model.transcribe(samples)
        with open(args.out_csv, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([uid, hyp])
        if args.heartbeat and ((i + 1) % args.heartbeat_interval == 0 or i + 1 == len(rows)):
            elapsed = time.time() - sweep_start
            with open(args.heartbeat, "w") as hb:
                hb.write(
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} utt={i+1}/{len(rows)} "
                    f"elapsed={elapsed:.1f}s mode={args.mode}\n"
                )
            print(f"  [{i+1}/{len(rows)}] elapsed={elapsed:.1f}s", flush=True)
    print(f"[phase4c] done. wrote {args.out_csv} ({len(rows)} rows). "
          f"wall={time.time()-sweep_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
