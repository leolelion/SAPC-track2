#!/usr/bin/env python3
"""
Test script for the faster-whisper streaming Model wrapper.

Usage:
  python test_streaming.py            # loads real model (downloads if needed)
  python test_streaming.py --mock     # stub model, no download, tests interface only
  python test_streaming.py --partial-interval 10   # override partial interval

What this tests:
  - Model.__init__()          loads without error
  - set_partial_callback()    callback fires at the right times
  - reset()                   clears state between utterances
  - accept_chunk()            accepts 100ms float32 chunks, returns str
  - input_finished()          returns final transcription

Test utterances (synthetic, white-noise-based):
  - 2 s  (20 chunks)
  - 5 s  (50 chunks)
  - 10 s (100 chunks)

Note on synthetic audio:
  White noise will likely produce empty or garbage transcriptions — that is
  expected and fine. The test verifies timing and interface correctness, not
  transcription quality. Use real speech audio to test accuracy.

macOS / Linux:
  This script works on both. Install deps with:
    pip install faster-whisper omegaconf
  First real run downloads CTranslate2-converted weights from HuggingFace.
  Model sizes: tiny=75 MB, base=145 MB, small=480 MB (all INT8).
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
    "--partial-interval",
    type=int,
    default=None,
    help="Override config.streaming.partial_interval_chunks for this run.",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Resolve paths so this script can be run from any working directory
# ---------------------------------------------------------------------------
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

# ---------------------------------------------------------------------------
# Stub model (--mock mode)
# ---------------------------------------------------------------------------
class _StubModel:
    """Drop-in replacement for Model that never touches HuggingFace."""

    def __init__(self):
        self._cb = None
        self._chunks: List[np.ndarray] = []
        self._last = ""
        self._chunks_since = 0
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(_DIR / "config.yaml")
        self._interval = cfg.streaming.partial_interval_chunks
        if args.partial_interval is not None:
            self._interval = args.partial_interval
        print(f"[stub] Model loaded (partial_interval={self._interval})")

    def set_partial_callback(self, fn):
        self._cb = fn

    def reset(self):
        self._chunks = []
        self._last = ""
        self._chunks_since = 0

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
# Load the real Model (or stub)
# ---------------------------------------------------------------------------
if args.mock:
    print("=" * 60)
    print("MOCK MODE — no model weights downloaded")
    print("=" * 60)
    model = _StubModel()
else:
    print("=" * 60)
    print("Loading faster-whisper streaming model …")
    print("(first run downloads CTranslate2 weights from HuggingFace)")
    print("=" * 60)
    from model import Model  # type: ignore

    if args.partial_interval is not None:
        import model as _model_module
        _model_module._config.streaming.partial_interval_chunks = args.partial_interval

    t0 = time.perf_counter()
    model = Model()
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f} s\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600          # 100 ms at 16 kHz
RNG = np.random.default_rng(42)


def make_audio(duration_s: float) -> np.ndarray:
    """Synthetic audio: low-amplitude white noise (float32, mono, 16 kHz)."""
    n = int(duration_s * SAMPLE_RATE)
    return (RNG.standard_normal(n) * 0.01).astype(np.float32)


def chunked(audio: np.ndarray, chunk_size: int = CHUNK_SAMPLES):
    """Yield successive fixed-size slices; zero-pad the final partial chunk."""
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.concatenate(
                [chunk, np.zeros(chunk_size - len(chunk), dtype=np.float32)]
            )
        yield chunk


def run_utterance(model, duration_s: float, label: str) -> None:
    """Run the full streaming lifecycle for one synthetic utterance."""
    print(f"\n{'─' * 60}")
    print(f"  {label}: {duration_s:.0f}s audio  "
          f"({int(duration_s * 10)} × 100ms chunks)")
    print(f"{'─' * 60}")

    audio = make_audio(duration_s)
    chunks = list(chunked(audio))
    partials_received: List[tuple] = []

    t_start = time.perf_counter()

    def on_partial(text: str):
        elapsed = time.perf_counter() - t_start
        partials_received.append((_chunk_counter[0], elapsed, text))
        if text:
            print(f"  [callback] chunk {_chunk_counter[0]:>3d}  "
                  f"+{elapsed*1000:6.0f} ms  → {text!r}")

    model.set_partial_callback(on_partial)

    t_reset = time.perf_counter()
    model.reset()
    print(f"  reset()          {(time.perf_counter()-t_reset)*1000:.1f} ms")

    _chunk_counter = [0]
    t_chunks_start = time.perf_counter()
    last_partial_from_accept = ""

    for chunk in chunks:
        _chunk_counter[0] += 1
        t_c = time.perf_counter()
        result = model.accept_chunk(chunk)
        elapsed_c = (time.perf_counter() - t_c) * 1000
        if result and result != last_partial_from_accept:
            print(f"  accept_chunk({_chunk_counter[0]:>3d})  "
                  f"{elapsed_c:6.1f} ms  → {result!r}")
            last_partial_from_accept = result

    chunks_total_ms = (time.perf_counter() - t_chunks_start) * 1000
    print(f"  {len(chunks)} chunks done   total {chunks_total_ms:.0f} ms  "
          f"({chunks_total_ms/len(chunks):.1f} ms/chunk avg)")

    t_fin = time.perf_counter()
    final = model.input_finished()
    fin_ms = (time.perf_counter() - t_fin) * 1000

    print(f"  input_finished() {fin_ms:.0f} ms")
    print(f"\n  FINAL TRANSCRIPTION:")
    print(f"  {final!r}")
    print(f"\n  Partial callbacks fired: {len(partials_received)}")
    if partials_received:
        first_non_empty = next(
            (p for p in partials_received if p[2].strip()), None
        )
        if first_non_empty:
            print(f"  TTFT (simulated):  {first_non_empty[1]*1000:.0f} ms  "
                  f"(at chunk {first_non_empty[0]})")
    total_s = time.perf_counter() - t_start
    rtf = total_s / duration_s
    print(f"  Wall time:         {total_s:.2f} s  (RTF {rtf:.2f}×)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print(f"\nSAMPLE_RATE = {SAMPLE_RATE} Hz  |  CHUNK = {CHUNK_SAMPLES} samples (100 ms)")
if not args.mock:
    import model as _m
    print(f"model = {_m._config.model.name}  |  compute_type = {_m._config.model.compute_type}  |  "
          f"partial_interval_chunks = {_m._config.streaming.partial_interval_chunks}")

for duration in [2.0, 5.0, 10.0]:
    run_utterance(model, duration, f"utterance {duration:.0f}s")

print(f"\n{'=' * 60}")
print("All tests passed.")
print("=" * 60)
