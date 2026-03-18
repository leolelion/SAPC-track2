#!/usr/bin/env python3
"""
Moonshine Streaming Model — SAPC2 Track 2
==========================================

Wraps Useful Sensors' Moonshine (ONNX variant) with the 5-method
streaming interface required by the ingestion program.

Moonshine is designed for CPU/edge inference and is NOT a
frame-by-frame streaming model. Like Qwen3-ASR, it processes
complete audio utterances. To satisfy the SAPC2 interface, audio
chunks are buffered and inference runs in two situations:

  1. In accept_chunk(): every `partial_interval_chunks` chunks
     (enabled by default — Moonshine/tiny is fast enough on CPU
     that periodic partial inference is practical).
  2. In input_finished(): once, on the full buffer (always).

Model sizes:
  moonshine/tiny  ~30M params, fastest, lowest accuracy
  moonshine/base  ~100M params, slower, better accuracy

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
  └── setup.sh       ← installs useful-moonshine-onnx into /opt/moonshine_venv
"""

# =====================================================================
# Section 1: Venv Path Injection
# =====================================================================
# setup.sh installs dependencies into a clean venv to avoid conflicts
# with the base conda environment's CUDA/numpy binaries.
# On macOS (local dev), /opt/moonshine_venv won't exist, so this is
# a no-op and the locally activated venv's packages are used instead.
import os
import sys

import glob as _glob
_venv_candidates = _glob.glob(
    os.path.join(os.path.dirname(__file__), "venv", "lib", "python3.*", "site-packages")
)
if _venv_candidates:
    sys.path.insert(0, _venv_candidates[0])

# =====================================================================
# Section 2: Imports
# =====================================================================
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from omegaconf import OmegaConf
from moonshine_onnx import MoonshineOnnxModel
from moonshine_onnx.transcribe import load_tokenizer

# =====================================================================
# Section 3: Config
# =====================================================================
_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_config = OmegaConf.load(_DIR / "config.yaml")


# =====================================================================
# Section 4: Model — Public Interface for the Ingestion Program
# =====================================================================
class Model:
    """Streaming ASR model wrapping Moonshine-ONNX (useful-moonshine-onnx).

    Lifecycle (called by ingestion program):
      model.set_partial_callback(fn)           # register callback (once)
      model.reset()                             # prepare for new file
      for chunk in audio_chunks:
          partial = model.accept_chunk(chunk)   # returns partial text
      final = model.input_finished()            # returns final text
    """

    def __init__(self):
        model_name = _config.model.name
        print(f"Loading {model_name} (ONNX, cpu) …")
        self._partial_callback: Optional[Callable[[str], None]] = None

        # MoonshineOnnxModel downloads ONNX weights from HuggingFace on
        # first use and caches them. setup.sh pre-downloads them so that
        # inference time at evaluation is not spent on downloads.
        self._model = MoonshineOnnxModel(model_name=model_name)
        self._tokenizer = load_tokenizer()

        # Per-file state — reset in reset()
        self._audio_chunks: List[np.ndarray] = []
        self._last_partial: str = ""
        self._chunks_since_partial: int = 0

        print(f"{model_name} loaded (ONNX, cpu)")

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

        If partial_interval_chunks == 0, returns "" immediately.
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
        """Run Moonshine-ONNX inference on the current audio buffer.

        Concatenates all buffered chunks into a single float32 array
        and passes it to MoonshineOnnxModel.generate(). The returned
        token sequences are decoded to text via decode_tokens().
        """
        audio = np.concatenate(self._audio_chunks)[np.newaxis, :]  # (1, N) float32, 16 kHz
        tokens = self._model.generate(audio)
        transcription = self._tokenizer.decode_batch(tokens)
        return transcription[0].strip()
