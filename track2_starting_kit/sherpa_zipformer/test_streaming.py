#!/usr/bin/env python3
"""
Test script for the sherpa-onnx Zipformer streaming Model wrapper.

Usage:
  python test_streaming.py            # loads real model (requires setup.sh)
  python test_streaming.py --mock     # stub model, no download, interface check
  python test_streaming.py --variant kroko   # override model.variant in config
  python test_streaming.py --audio /path/to/file.wav  # real audio file

What this tests:
  - Model.__init__()          loads without error
  - set_partial_callback()    callback fires (true streaming — fires per token)
  - reset()                   clears decoder state between utterances
  - accept_chunk()            accepts 100ms float32 chunks, returns str
  - input_finished()          returns final transcription

Latency metrics printed per utterance:
  - reset() latency
  - accept_chunk() latency (avg ms/chunk across the utterance)
  - input_finished() latency
  - Simulated TTFT (time from first chunk to first non-empty callback)
  - Wall-time RTF

Test utterances (synthetic white noise, no real speech):
  - 2 s  (20 chunks)
  - 5 s  (50 chunks)
  - 10 s (100 chunks)

macOS / Linux:
  Install deps with:  pip install sherpa-onnx omegaconf
  Then run setup.sh to download ONNX weights before using --real mode.
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
    "--variant",
    type=str,
    default=None,
    help="Override config.model.variant (standard | kroko | small).",
)
parser.add_argument(
    "--audio",
    type=str,
    default=None,
    help="Path to a WAV file (16 kHz mono float32) to transcribe instead of synthetic audio.",
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
    """Drop-in replacement for Model that never touches any weights."""

    def __init__(self):
        self._cb = None
        self._chunks: List[np.ndarray] = []
        self._tokens_so_far = 0
        print("[stub] Zipformer Model loaded (true-streaming stub)")

    def set_partial_callback(self, fn):
        self._cb = fn

    def reset(self):
        self._chunks = []
        self._tokens_so_far = 0

    def accept_chunk(self, chunk: np.ndarray) -> str:
        self._chunks.append(chunk)
        n = len(self._chunks)
        # Simulate one new token every 10 chunks (~1 s of audio)
        new_tokens = n // 10
        if new_tokens > self._tokens_so_far:
            self._tokens_so_far = new_tokens
            dur = len(np.concatenate(self._chunks)) / 16000
            text = f"[stub partial @ {dur:.1f}s tokens={new_tokens}]"
            if self._cb:
                self._cb(text)
            return text
        return "" if not self._chunks else f"[stub {len(self._chunks)} chunks]"

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
    print("MOCK MODE — no model weights loaded")
    print("=" * 60)
    model = _StubModel()
else:
    # Apply variant override before importing Model
    if args.variant is not None:
        import model as _model_module
        from omegaconf import OmegaConf
        _model_module._config.model.variant = args.variant
        print(f"Variant overridden to: {args.variant}")

    print("=" * 60)
    print("Loading sherpa-onnx Zipformer …")
    print("(requires setup.sh to have been run)")
    print("=" * 60)
    from model import Model  # type: ignore

    t0 = time.perf_counter()
    model = Model()
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f} s\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600  # 100 ms at 16 kHz
RNG = np.random.default_rng(42)


def make_audio(duration_s: float) -> np.ndarray:
    """Synthetic audio: low-amplitude white noise (float32, mono, 16 kHz).

    Note: the model will likely output empty or near-empty text for noise.
    Use --audio with a real WAV file to verify transcription accuracy.
    """
    n = int(duration_s * SAMPLE_RATE)
    return (RNG.standard_normal(n) * 0.01).astype(np.float32)


def load_wav(path: str) -> np.ndarray:
    """Load a 16 kHz mono WAV file as float32 samples."""
    import wave
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE, (
            f"WAV must be 16 kHz, got {wf.getframerate()} Hz"
        )
        assert wf.getnchannels() == 1, "WAV must be mono"
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def chunked(audio: np.ndarray, chunk_size: int = CHUNK_SAMPLES):
    """Yield fixed-size slices; zero-pad the final partial chunk."""
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.concatenate(
                [chunk, np.zeros(chunk_size - len(chunk), dtype=np.float32)]
            )
        yield chunk


def run_utterance(model, audio: np.ndarray, label: str) -> None:
    """Run the full streaming lifecycle for one utterance."""
    duration_s = len(audio) / SAMPLE_RATE
    chunks = list(chunked(audio))
    print(f"\n{'─' * 60}")
    print(f"  {label}: {duration_s:.1f}s audio  ({len(chunks)} × 100ms chunks)")
    print(f"{'─' * 60}")

    partials_received: List[tuple] = []
    t_start = time.perf_counter()
    _chunk_counter = [0]

    def on_partial(text: str):
        elapsed = time.perf_counter() - t_start
        partials_received.append((_chunk_counter[0], elapsed, text))
        if text.strip():
            print(
                f"  [callback] chunk {_chunk_counter[0]:>3d}  "
                f"+{elapsed*1000:6.0f} ms  → {text!r}"
            )

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
            print(
                f"  accept_chunk({_chunk_counter[0]:>3d})  "
                f"{chunk_ms:6.1f} ms  → {result!r}"
            )
            prev_text = result

    chunks_total_ms = (time.perf_counter() - t_chunks_start) * 1000
    avg_chunk_ms = sum(chunk_latencies) / len(chunk_latencies) if chunk_latencies else 0
    print(
        f"  {len(chunks)} chunks done   total {chunks_total_ms:.0f} ms  "
        f"(avg {avg_chunk_ms:.1f} ms/chunk)"
    )

    t_fin = time.perf_counter()
    final = model.input_finished()
    fin_ms = (time.perf_counter() - t_fin) * 1000

    print(f"  input_finished() {fin_ms:.0f} ms")
    print(f"\n  FINAL TRANSCRIPTION:")
    print(f"  {final!r}")

    non_empty_partials = [p for p in partials_received if p[2].strip()]
    print(f"\n  Partial callbacks (non-empty): {len(non_empty_partials)}")
    if non_empty_partials:
        first = non_empty_partials[0]
        print(
            f"  Simulated TTFT:  {first[1]*1000:.0f} ms  (chunk {first[0]})"
        )
    total_s = time.perf_counter() - t_start
    rtf = total_s / duration_s
    print(f"  Wall time:       {total_s:.2f} s  (RTF {rtf:.3f}×)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print(f"\nSAMPLE_RATE = {SAMPLE_RATE} Hz  |  CHUNK = {CHUNK_SAMPLES} samples (100 ms)")
if not args.mock:
    import model as _m
    print(f"variant = {_m._config.model.variant}  |  "
          f"decoding = {_m._config.decoding.method}  "
          f"max_active_paths = {_m._config.decoding.max_active_paths}")

if args.audio:
    # Single real-audio test
    print(f"\nReal audio: {args.audio}")
    audio = load_wav(args.audio)
    run_utterance(model, audio, f"real audio ({len(audio)/SAMPLE_RATE:.1f}s)")
else:
    # Synthetic utterances at three durations
    for duration in [2.0, 5.0, 10.0]:
        audio = make_audio(duration)
        run_utterance(model, audio, f"synthetic {duration:.0f}s")

print(f"\n{'=' * 60}")
print("All tests passed.")
print("=" * 60)
print("\nNext step: test with real dysarthric speech from the Dev set:")
print("  python test_streaming.py --audio /workspace/SAPC2/processed/Dev/<utt>.wav")
