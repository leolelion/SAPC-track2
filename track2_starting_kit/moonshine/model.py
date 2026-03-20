#!/usr/bin/env python3
"""
Moonshine Streaming Model — SAPC2 Track 2
==========================================

Wraps moonshine-voice (v2, C library + ONNX Runtime) with the 5-method
streaming interface required by the ingestion program.

TRUE streaming via sliding-window self-attention (KV-cache). accept_chunk()
feeds new audio only — the C library does NOT re-process the full buffer.
This gives bounded latency regardless of utterance length.

  Model variants (English-only):
    tiny_streaming    Smallest, bundled in package (no download needed)
    base_streaming    Balanced quality, ~download required
    small_streaming   Moderate size, ~download required
    medium_streaming  Highest quality, ~download required
    tiny              Non-streaming batch mode (bundled)
    base              Non-streaming batch mode (download required)

Required interface (called by ingestion program):
  __init__()                       — Load model weights (once)
  set_partial_callback(fn) -> None — Register partial result callback
  reset()             -> None      — Reset state per audio file
  accept_chunk(buf)   -> str       — Feed 100 ms audio chunk
  input_finished()    -> str       — Signal end of audio, return text

To change settings, edit config.yaml (not this file).

Directory layout after running setup.sh:
  moonshine/
  ├── model.py       ← this file
  ├── config.yaml    ← all tunable settings
  └── setup.sh       ← installs the backend package into ./venv
"""

# =====================================================================
# Section 1: Venv Path Injection
# =====================================================================
# setup.sh installs dependencies into a clean venv to avoid conflicts
# with the base conda environment's CUDA/numpy binaries.
import os
import sys

import glob as _glob
_venv_candidates = _glob.glob(
    os.path.join(os.path.dirname(__file__), "venv", "lib", "python3.*", "site-packages")
)
if _venv_candidates:
    sys.path.insert(0, _venv_candidates[0])

# =====================================================================
# Section 2: Always-available imports
# =====================================================================
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from omegaconf import OmegaConf

# =====================================================================
# Section 3: Config + Backend Imports
# =====================================================================
_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_config = OmegaConf.load(_DIR / "config.yaml")

from moonshine_voice import (                                # type: ignore
    Transcriber as _Transcriber,
    ModelArch as _ModelArch,
    get_model_for_language as _get_model_for_language,
)

_V2_ARCH_MAP = {
    "tiny":             _ModelArch.TINY,
    "base":             _ModelArch.BASE,
    "tiny_streaming":   _ModelArch.TINY_STREAMING,
    "base_streaming":   _ModelArch.BASE_STREAMING,
    "small_streaming":  _ModelArch.SMALL_STREAMING,
    "medium_streaming": _ModelArch.MEDIUM_STREAMING,
}

def _transcript_to_text(transcript) -> str:
    """Extract concatenated text from a moonshine_voice Transcript."""
    if transcript is None:
        return ""
    lines = [line.text.strip() for line in transcript.lines if line.text.strip()]
    return " ".join(lines)


# =====================================================================
# Section 4: Model — Public Interface for the Ingestion Program
# =====================================================================
class Model:
    """Streaming ASR model wrapping moonshine-voice (v2, C library).

    Lifecycle (called by ingestion program):
      model.set_partial_callback(fn)           # register callback (once)
      model.reset()                             # prepare for new file
      for chunk in audio_chunks:
          partial = model.accept_chunk(chunk)   # returns partial text
      final = model.input_finished()            # returns final text
    """

    def __init__(self):
        self._partial_callback: Optional[Callable[[str], None]] = None
        self._last_partial: str = ""
        self._chunks_since_partial: int = 0
        self._total_chunks: int = 0

        v2_arch_name = getattr(_config.model, "v2_arch", "tiny_streaming")
        arch = _V2_ARCH_MAP.get(v2_arch_name, _ModelArch.TINY_STREAMING)
        print(f"Loading moonshine-voice {v2_arch_name} …")
        model_path, arch = _get_model_for_language("en", arch)
        # update_interval=0.1 s makes the C library run inference at most
        # every 100 ms when add_audio() is called — fine-grained enough for
        # our 100 ms chunk cadence.
        self._transcriber = _Transcriber(model_path, arch, update_interval=0.1)
        self._transcriber.start()
        self._stream_started: bool = True
        print(f"moonshine-voice {v2_arch_name} loaded")

    # -----------------------------------------------------------------
    # Streaming Interface (called by the ingestion program)
    # -----------------------------------------------------------------

    def set_partial_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback for partial results: callback(text: str)."""
        self._partial_callback = callback

    def reset(self) -> None:
        """Reset state for a new audio file. Call once before each file."""
        self._last_partial = ""
        self._chunks_since_partial = 0
        self._total_chunks = 0

        if self._stream_started:
            try:
                self._transcriber.stop()
            except Exception:
                pass
        self._transcriber.start()
        self._stream_started = True

    def accept_chunk(self, audio_chunk: np.ndarray) -> str:
        """Feed one 100 ms audio chunk (float32, 16 kHz) and return partial text.

        Feeds the chunk to the C library's incremental KV-cache decoder;
        polls update_transcription() every partial_interval_chunks chunks.

        Returns the most recent partial transcription (empty string until the
        first partial fires).
        """
        interval = _config.streaming.partial_interval_chunks
        min_chunks = _config.streaming.min_chunks_for_partial

        self._chunks_since_partial += 1
        self._total_chunks += 1

        self._transcriber.add_audio(audio_chunk.tolist(), 16000)

        if (
            interval > 0
            and self._chunks_since_partial >= interval
            and self._total_chunks >= min_chunks
        ):
            transcript = self._transcriber.update_transcription()
            self._last_partial = _transcript_to_text(transcript)
            self._chunks_since_partial = 0
            if self._partial_callback is not None:
                self._partial_callback(self._last_partial)

        return self._last_partial

    def input_finished(self) -> str:
        """Signal end of audio. Runs final inference and returns the transcription."""
        if not self._stream_started:
            return ""
        # stop() flushes remaining audio and returns the final Transcript.
        final = self._transcriber.stop()
        self._stream_started = False
        text = _transcript_to_text(final)
        if not text:
            # Some implementations return None from stop(); poll once more.
            try:
                t = self._transcriber.update_transcription()
                text = _transcript_to_text(t)
            except Exception:
                pass

        if self._partial_callback is not None:
            self._partial_callback(text)
        return text
