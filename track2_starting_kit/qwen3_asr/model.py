#!/usr/bin/env python3
"""
Qwen3-ASR Streaming Model — SAPC2 Track 2
==========================================

Wraps Qwen/Qwen3-ASR-1.7B with the 5-method streaming interface
required by the ingestion program. Uses the official `qwen-asr`
Python package (transformers backend, CPU).

Streaming strategy
------------------
Qwen3-ASR processes complete audio utterances; it is not a
frame-by-frame streaming model. To satisfy the interface, audio
chunks are buffered and inference is run in two situations:

  1. In accept_chunk(): every `partial_interval_chunks` chunks
     (optional — disabled by default on CPU due to cost: each
     inference call takes ~5–15 s on a CPU for a 1.7B model).
  2. In input_finished(): once, on the full buffer (always).

With partial inference disabled (default, partial_interval_chunks=0):
  - accept_chunk() returns "" immediately — zero latency per call.
  - The transcription is returned only from input_finished().
  - TTFT and TTLT latency metrics will be poor, but CER/WER
    (accuracy) is unaffected. Accuracy is the primary metric.

With partial inference enabled (e.g., partial_interval_chunks=20):
  - Inference runs every 2 s of audio (20 × 100 ms chunks).
  - The partial callback fires after each inference, enabling TTFT.
  - CPU cost is multiplied by the number of partial calls per file.

Required interface (called by ingestion program):
  __init__()                       — Load model weights (once)
  set_partial_callback(fn) -> None — Register partial result callback
  reset()             -> None      — Reset state per audio file
  accept_chunk(buf)   -> str       — Feed 100 ms audio chunk
  input_finished()    -> str       — Signal end of audio, return text

To change settings, edit config.yaml (not this file).

Directory layout after running setup.sh:
  qwen3_asr/
  ├── model.py       ← this file
  ├── config.yaml    ← all tunable settings
  └── setup.sh       ← installs qwen-asr into /opt/qwen3_asr_venv
"""

# =====================================================================
# Section 1: Venv Path Injection
# =====================================================================
# setup.sh installs dependencies into a clean venv to avoid conflicts
# with the base conda environment's CUDA/numpy binaries.
# On macOS (local dev), /opt/qwen3_asr_venv won't exist, so this is
# a no-op and the locally activated venv's packages are used instead.
import os
import sys

_VENV_SITE = "/opt/qwen3_asr_venv/lib/python3.11/site-packages"
if os.path.isdir(_VENV_SITE):
    sys.path.insert(0, _VENV_SITE)

# =====================================================================
# Section 2: Imports
# =====================================================================
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from qwen_asr import Qwen3ASRModel

# =====================================================================
# Section 3: Config
# =====================================================================
_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_config = OmegaConf.load(_DIR / "config.yaml")


# =====================================================================
# Section 4: Model — Public Interface for the Ingestion Program
# =====================================================================
class Model:
    """Streaming ASR model wrapping Qwen3-ASR-1.7B (qwen-asr package).

    Lifecycle (called by ingestion program):
      model.set_partial_callback(fn)           # register callback (once)
      model.reset()                             # prepare for new file
      for chunk in audio_chunks:
          partial = model.accept_chunk(chunk)   # returns partial text
      final = model.input_finished()            # returns final text
    """

    def __init__(self):
        print(f"Loading {_config.model.name} (transformers backend, cpu) …")
        self._partial_callback: Optional[Callable[[str], None]] = None

        dtype = getattr(torch, _config.model.torch_dtype)
        self._asr = Qwen3ASRModel.from_pretrained(
            _config.model.name,
            dtype=dtype,
            device_map="cpu",
            max_new_tokens=_config.decoding.max_new_tokens,
        )

        # Per-file state — initialised properly in reset()
        self._audio_chunks: List[np.ndarray] = []
        self._last_partial: str = ""
        self._chunks_since_partial: int = 0

        print("Model loaded on cpu")

    # -----------------------------------------------------------------
    # Streaming Interface (called by the ingestion program)
    # -----------------------------------------------------------------

    def set_partial_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback for partial results: callback(text: str)."""
        self._partial_callback = callback

    def reset(self) -> None:
        """Reset state for a new audio file. Call once before each file."""
        self._audio_chunks = []
        self._last_partial = ""
        self._chunks_since_partial = 0

    def accept_chunk(self, audio_chunk: np.ndarray) -> str:
        """Feed one 100 ms audio chunk (float32, 16 kHz) and return partial text.

        If partial_interval_chunks == 0 (default), returns "" immediately.
        Otherwise runs inference every N chunks and fires the callback.
        """
        self._audio_chunks.append(audio_chunk)
        self._chunks_since_partial += 1

        interval = _config.streaming.partial_interval_chunks
        min_chunks = _config.streaming.min_chunks_for_partial

        if (
            interval > 0
            and self._chunks_since_partial >= interval
            and len(self._audio_chunks) >= min_chunks
        ):
            self._last_partial = self._transcribe()
            self._chunks_since_partial = 0
            if self._partial_callback is not None:
                self._partial_callback(self._last_partial)

        return self._last_partial

    def input_finished(self) -> str:
        """Signal end of audio. Runs final inference and returns the transcription."""
        if not self._audio_chunks:
            return ""
        final_text = self._transcribe()
        if self._partial_callback is not None:
            self._partial_callback(final_text)
        return final_text

    # -----------------------------------------------------------------
    # Private Helpers
    # -----------------------------------------------------------------

    def _transcribe(self) -> str:
        """Run Qwen3-ASR inference on the current audio buffer.

        Concatenates all buffered chunks and passes them to
        Qwen3ASRModel.transcribe() as a (ndarray, sample_rate) tuple.
        Returns the transcribed text string.
        """
        audio = np.concatenate(self._audio_chunks)  # (N,) float32, 16 kHz
        results = self._asr.transcribe(
            audio=(audio, _config.audio.sample_rate),
            language=OmegaConf.select(_config, "model.language", default=None),
        )
        return results[0].text.strip()
