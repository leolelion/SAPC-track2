#!/usr/bin/env python3
# Re-exec with venv Python if the current interpreter doesn't match.
# NeMo requires Python 3.11; the venv is built with python3.11.
# Running this script directly with `python3` on a machine where
# `python3` resolves to 3.13 would cause C-extension import failures.
import os, sys
from pathlib import Path as _Path
_venv_py = _Path(__file__).parent / "venv" / "bin" / "python3"
if _venv_py.exists() and os.path.realpath(sys.executable) != os.path.realpath(str(_venv_py)):
    os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)

"""
Test script for the NeMo FastConformer streaming Model wrapper.

Usage:
  python test_streaming.py            # loads real model (downloads ~32M params)
  python test_streaming.py --mock     # stub model, no download, interface check
  python test_streaming.py --audio /path/to/file.wav  # real WAV file
  python test_streaming.py --decoder rnnt  # override decoder type

What this tests:
  - Model.__init__()          loads without error
  - set_partial_callback()    callback fires at the right interval
  - reset()                   clears caches between utterances
  - accept_chunk()            accepts 100ms float32 chunks, returns str
  - input_finished()          returns final transcription

Latency metrics printed per utterance:
  - reset() latency
  - accept_chunk() latency (avg ms/chunk)
  - input_finished() latency
  - Simulated TTFT
  - Wall-time RTF

Test utterances (synthetic white noise):
  - 2 s  (20 chunks)
  - 5 s  (50 chunks)
  - 10 s (100 chunks)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mock",
    action="store_true",
    help="Use a stub model (no download). Tests interface only.",
)
parser.add_argument(
    "--audio",
    type=str,
    default=None,
    help="Path to a WAV file (16 kHz mono) to transcribe instead of synthetic audio.",
)
parser.add_argument(
    "--decoder",
    type=str,
    default=None,
    help="Override config.model.decoder_type (ctc | rnnt).",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

# ---------------------------------------------------------------------------
# Stub model (--mock mode)
# ---------------------------------------------------------------------------
class _StubModel:
    """Drop-in replacement for Model that never touches NeMo."""

    def __init__(self):
        self._cb = None
        self._chunks: List[np.ndarray] = []
        self._chunks_since = 0
        self._last = ""
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(_DIR / "config.yaml")
        self._interval = cfg.streaming.partial_interval_chunks
        print(f"[stub] NeMo FastConformer loaded  (partial_interval={self._interval})")

    def set_partial_callback(self, fn):
        self._cb = fn

    def reset(self):
        self._chunks = []
        self._chunks_since = 0
        self._last = ""

    def accept_chunk(self, chunk: np.ndarray) -> str:
        self._chunks.append(chunk)
        self._chunks_since += 1
        if self._interval > 0 and self._chunks_since >= self._interval:
            dur = len(np.concatenate(self._chunks)) / 16000
            self._last = f"[stub partial @ {dur:.1f}s]"
            self._chunks_since = 0
            if self._cb:
                self._cb(self._last)
        return self._last

    def input_finished(self) -> str:
        dur = len(np.concatenate(self._chunks)) / 16000 if self._chunks else 0.0
        text = f"[stub final, {dur:.1f}s audio, {len(self._chunks)} chunks]"
        if self._cb:
            self._cb(text)
        return text


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
if args.mock:
    print("=" * 60)
    print("MOCK MODE — no model weights loaded")
    print("=" * 60)
    model = _StubModel()
else:
    if args.decoder is not None:
        import model as _model_module
        _model_module._config.model.decoder_type = args.decoder
        print(f"Decoder overridden to: {args.decoder}")

    print("=" * 60)
    print("Loading NeMo FastConformer …")
    print("(first run downloads weights from HuggingFace)")
    print("=" * 60)
    from model import Model  # type: ignore

    t0 = time.perf_counter()
    model = Model()
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f} s\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600  # 100ms
RNG = np.random.default_rng(42)


def make_audio(duration_s: float) -> np.ndarray:
    n = int(duration_s * SAMPLE_RATE)
    return (RNG.standard_normal(n) * 0.01).astype(np.float32)


def load_wav(path: str) -> np.ndarray:
    import wave
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE, f"WAV must be 16kHz, got {wf.getframerate()}"
        assert wf.getnchannels() == 1, "WAV must be mono"
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def chunked(audio: np.ndarray, chunk_size: int = CHUNK_SAMPLES):
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.concatenate([chunk, np.zeros(chunk_size - len(chunk), dtype=np.float32)])
        yield chunk


def run_utterance(model, audio: np.ndarray, label: str) -> None:
    duration_s = len(audio) / SAMPLE_RATE
    chunks = list(chunked(audio))
    print(f"\n{'─' * 60}")
    print(f"  {label}: {duration_s:.1f}s  ({len(chunks)} × 100ms chunks)")
    print(f"{'─' * 60}")

    partials_received: List[tuple] = []
    t_start = time.perf_counter()
    _chunk_counter = [0]

    def on_partial(text: str):
        elapsed = time.perf_counter() - t_start
        partials_received.append((_chunk_counter[0], elapsed, text))
        if text.strip():
            print(f"  [callback] chunk {_chunk_counter[0]:>3d}  +{elapsed*1000:6.0f} ms  → {text!r}")

    model.set_partial_callback(on_partial)

    t_reset = time.perf_counter()
    model.reset()
    print(f"  reset()          {(time.perf_counter()-t_reset)*1000:.1f} ms")

    t_chunks_start = time.perf_counter()
    prev_text = ""
    chunk_latencies = []

    for chunk in chunks:
        _chunk_counter[0] += 1
        t_c = time.perf_counter()
        result = model.accept_chunk(chunk)
        chunk_ms = (time.perf_counter() - t_c) * 1000
        chunk_latencies.append(chunk_ms)
        if result and result != prev_text:
            print(f"  accept_chunk({_chunk_counter[0]:>3d})  {chunk_ms:6.1f} ms  → {result!r}")
            prev_text = result

    chunks_total_ms = (time.perf_counter() - t_chunks_start) * 1000
    avg_chunk_ms = sum(chunk_latencies) / len(chunk_latencies) if chunk_latencies else 0
    print(f"  {len(chunks)} chunks done   total {chunks_total_ms:.0f} ms  (avg {avg_chunk_ms:.1f} ms/chunk)")

    t_fin = time.perf_counter()
    final = model.input_finished()
    fin_ms = (time.perf_counter() - t_fin) * 1000

    print(f"  input_finished() {fin_ms:.0f} ms")
    print(f"\n  FINAL TRANSCRIPTION:")
    print(f"  {final!r}")

    non_empty = [p for p in partials_received if p[2].strip()]
    print(f"\n  Partial callbacks (non-empty): {len(non_empty)}")
    if non_empty:
        first = non_empty[0]
        print(f"  Simulated TTFT:  {first[1]*1000:.0f} ms  (chunk {first[0]})")
    total_s = time.perf_counter() - t_start
    rtf = total_s / duration_s
    print(f"  Wall time:       {total_s:.2f} s  (RTF {rtf:.3f}×)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print(f"\nSAMPLE_RATE = {SAMPLE_RATE} Hz  |  CHUNK = {CHUNK_SAMPLES} samples (100ms)")
if not args.mock:
    import model as _m
    print(
        f"model = {_m._config.model.name}  |  "
        f"decoder = {_m._config.model.decoder_type}  |  "
        f"partial_interval = {_m._config.streaming.partial_interval_chunks}"
    )

if args.audio:
    print(f"\nReal audio: {args.audio}")
    audio = load_wav(args.audio)
    run_utterance(model, audio, f"real audio ({len(audio)/SAMPLE_RATE:.1f}s)")
else:
    for duration in [2.0, 5.0, 10.0]:
        audio = make_audio(duration)
        run_utterance(model, audio, f"synthetic {duration:.0f}s")

print(f"\n{'=' * 60}")
print("All tests passed.")
print("=" * 60)
print("\nNext step: test with real dysarthric speech from the Dev set:")
print("  python test_streaming.py --audio /workspace/SAPC2/processed/Dev/<utt>.wav")
print("\nTo try RNNT decoder (more accurate, slower):")
print("  python test_streaming.py --decoder rnnt")
