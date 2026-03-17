#!/usr/bin/env python3
"""
Faster-Whisper Streaming Model — SAPC2 Track 2
===============================================

Wraps faster-whisper (CTranslate2 backend) with the 5-method
streaming interface required by the ingestion program.

faster-whisper is NOT a frame-by-frame streaming model. Like
Qwen3-ASR and Moonshine, it processes complete audio utterances.
Audio chunks are buffered and inference runs in two situations:

  1. In accept_chunk(): every `partial_interval_chunks` chunks.
     With small+INT8, inference takes ~0.5–3 s per utterance on CPU,
     so periodic partial inference is practical and enabled by default.
  2. In input_finished(): once, on the full buffer (always).

Recommended model variants for CPU:
  small   (244M) + int8  — primary candidate (~0.5–2× RTF on CPU)
  base    (74M)  + int8  — faster fallback if small is too slow
  distil-small.en        — English-only distilled, ~2× faster than small

Required interface (called by ingestion program):
  __init__()                       — Load model weights (once)
  set_partial_callback(fn) -> None — Register partial result callback
  reset()             -> None      — Reset state per audio file
  accept_chunk(buf)   -> str       — Feed 100 ms audio chunk
  input_finished()    -> str       — Signal end of audio, return text

To change settings, edit config.yaml (not this file).

Directory layout after running setup.sh:
  faster_whisper/
  ├── model.py       ← this file
  ├── config.yaml    ← all tunable settings
  └── setup.sh       ← installs faster-whisper into /opt/faster_whisper_venv
"""

# =====================================================================
# Section 1: Venv Path Injection
# =====================================================================
# setup.sh installs dependencies into a clean venv to avoid conflicts
# with the base conda environment's CUDA/numpy binaries.
# On macOS (local dev), /opt/faster_whisper_venv won't exist, so this
# is a no-op and the locally activated venv's packages are used instead.
import os
import sys

_VENV_SITE = "/opt/faster_whisper_venv/lib/python3.11/site-packages"
if os.path.isdir(_VENV_SITE):
    sys.path.insert(0, _VENV_SITE)

# =====================================================================
# Section 2: Imports
# =====================================================================
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from omegaconf import OmegaConf
from faster_whisper import WhisperModel

# =====================================================================
# Section 3: Config
# =====================================================================
_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_config = OmegaConf.load(_DIR / "config.yaml")


# =====================================================================
# Section 4: Model — Public Interface for the Ingestion Program
# =====================================================================
class Model:
    """Streaming ASR model wrapping faster-whisper (CTranslate2 backend).

    Lifecycle (called by ingestion program):
      model.set_partial_callback(fn)           # register callback (once)
      model.reset()                             # prepare for new file
      for chunk in audio_chunks:
          partial = model.accept_chunk(chunk)   # returns partial text
      final = model.input_finished()            # returns final text
    """

    def __init__(self):
        model_name = _config.model.name
        compute_type = _config.model.compute_type
        print(f"Loading faster-whisper/{model_name} (cpu, {compute_type}) …")
        self._partial_callback: Optional[Callable[[str], None]] = None

        # WhisperModel downloads CTranslate2-converted weights from HuggingFace
        # on first use and caches them under ~/.cache/huggingface/. setup.sh
        # pre-downloads them to avoid download latency at evaluation time.
        self._model = WhisperModel(
            model_name,
            device="cpu",
            compute_type=compute_type,
        )

        # Per-file state — reset in reset()
        self._audio_chunks: List[np.ndarray] = []
        self._last_partial: str = ""
        self._chunks_since_partial: int = 0

        print(f"faster-whisper/{model_name} loaded (cpu, {compute_type})")

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
        """Run faster-whisper inference on the current audio buffer.

        Concatenates all buffered chunks into a single float32 array.
        faster-whisper.transcribe() returns a generator of Segment
        objects; we join their text fields into a single string.

        Key options:
          language    — forcing "en" skips the ~30-frame language
                        detection head, saving ~0.1 s per call on CPU.
          beam_size   — beam search width; 1 is greedy (fastest).
          condition_on_previous_text — disabled to prevent hallucination
                        cascades on dysarthric speech.
        """
        audio = np.concatenate(self._audio_chunks)  # (N,) float32, 16 kHz

        language = OmegaConf.select(_config, "model.language", default=None)
        segments, _ = self._model.transcribe(
            audio,
            language=language,
            beam_size=_config.decoding.beam_size,
            condition_on_previous_text=_config.decoding.condition_on_previous_text,
            no_speech_threshold=_config.decoding.no_speech_threshold,
            compression_ratio_threshold=_config.decoding.compression_ratio_threshold,
        )
        return " ".join(seg.text for seg in segments).strip()
