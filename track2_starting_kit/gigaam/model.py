#!/usr/bin/env python3
"""
GigaAM Streaming Model — SAPC2 Track 2
=======================================

Wraps Sber's GigaAM (v2 RNNT) with the 5-method streaming interface
required by the ingestion program.

⚠️  LANGUAGE WARNING — READ BEFORE BENCHMARKING ⚠️
GigaAM v2 was trained on 50,000+ hours of Russian speech. English
recognition quality is UNKNOWN and likely poor out of the box. Before
investing time in benchmarking this model against the competition Dev
set, verify English quality with a quick smoke test using 2–3 short
English utterances. If CER > 60% on clean English speech, skip this
model entirely and focus on Zipformer fine-tuning (see plan Phase 4).

Architecture overview:
  GigaAM-RNNT uses a Conformer encoder with chunkwise self-attention
  (chunk size configurable), enabling streaming inference without
  buffering the full utterance. In principle this gives true partial
  results during transcription. In practice, the `gigaam` Python API
  exposes transcribe() for complete utterances; this wrapper uses
  periodic partial inference (like Moonshine/Whisper) rather than
  true frame-level streaming.

Model variants:
  v2_rnnt — Conformer + RNNT decoder (~100M params, streaming-capable)
  v2_ctc  — Conformer + CTC decoder  (~100M params, streaming-capable)

Required interface (called by ingestion program):
  __init__()                       — Load model weights (once)
  set_partial_callback(fn) -> None — Register partial result callback
  reset()             -> None      — Reset state per audio file
  accept_chunk(buf)   -> str       — Feed 100 ms audio chunk
  input_finished()    -> str       — Signal end of audio, return text

To change settings, edit config.yaml (not this file).

Directory layout after running setup.sh:
  gigaam/
  ├── model.py    ← this file
  ├── config.yaml ← all tunable settings
  └── setup.sh    ← installs gigaam into /opt/gigaam_venv
"""

# =====================================================================
# Section 1: Venv Path Injection
# =====================================================================
# setup.sh installs gigaam into a clean venv to avoid numpy/CUDA conflicts.
import os
import sys

# gigaam is installed in the base env (setup.sh uses --no-deps to bypass
# the torch<=2.5.1 constraint, relying on the base env's torch).

# =====================================================================
# Section 2: Imports
# =====================================================================
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from omegaconf import OmegaConf

try:
    import torch
    # torch 2.6+ changed torch.load default to weights_only=True, breaking gigaam.
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
    import gigaam
except ImportError as exc:
    raise ImportError(
        "gigaam is not installed. Run setup.sh first, or:\n"
        "  pip install gigaam\n"
        "  (installs from https://github.com/salute-developers/GigaAM)"
    ) from exc

# =====================================================================
# Section 3: Config
# =====================================================================
_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_config = OmegaConf.load(_DIR / "config.yaml")


# =====================================================================
# Section 4: Model — Public Interface for the Ingestion Program
# =====================================================================
class Model:
    """Streaming ASR model wrapping GigaAM (gigaam Python package).

    ⚠️  English support: GigaAM was trained on Russian data. Zero-shot
    English accuracy is expected to be poor. Benchmark on English
    before investing further effort.

    Pseudo-streaming: audio chunks are buffered. Inference runs either
    every `partial_interval_chunks` chunks (partial results) or once
    at input_finished() (final result). GigaAM's transcribe() API
    does not expose per-frame streaming, so this is the same pattern
    used by the Moonshine and faster-whisper wrappers.

    Lifecycle (called by ingestion program):
      model.set_partial_callback(fn)           # register callback (once)
      model.reset()                             # prepare for new file
      for chunk in audio_chunks:
          partial = model.accept_chunk(chunk)   # returns partial text
      final = model.input_finished()            # returns final text
    """

    SAMPLE_RATE = 16000  # GigaAM expects 16 kHz

    def __init__(self):
        model_name = _config.model.name
        print(f"Loading GigaAM {model_name} …")
        print(
            "⚠️  Note: GigaAM is trained on Russian speech. "
            "English quality is unknown — benchmark carefully."
        )

        self._partial_callback: Optional[Callable[[str], None]] = None

        # gigaam.load_model() downloads weights from HuggingFace on first
        # run and caches them. setup.sh pre-downloads to avoid latency at
        # evaluation time.
        self._model = gigaam.load_model(model_name)

        # Per-file state — reset in reset()
        self._audio_chunks: List[np.ndarray] = []
        self._last_partial: str = ""
        self._chunks_since_partial: int = 0

        print(f"GigaAM {model_name} loaded (cpu)")

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

        If partial_interval_chunks > 0, runs inference every N chunks.
        Otherwise returns the last cached partial immediately.
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
        """Run GigaAM inference on the current audio buffer.

        Concatenates all buffered chunks into a (N,) float32 array and
        passes it to gigaam's transcribe(). The exact API call depends
        on the gigaam version — see setup.sh for the installed version.

        GigaAM transcribe() API (as of gigaam v0.1):
          model.transcribe(audio, sr=16000) -> str
        where audio is a 1D float32 numpy array (values in [-1, 1]).
        """
        audio = np.concatenate(self._audio_chunks)  # (N,) float32, 16 kHz

        # gigaam.transcribe() expects a wav file path, not a numpy array.
        import tempfile, wave, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            pcm = np.round(audio * 32767).astype(np.int16)
            with wave.open(tmp_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(pcm.tobytes())
            duration_s = len(audio) / self.SAMPLE_RATE
            if duration_s > 30:
                result = self._model.transcribe_longform(tmp_path)
            else:
                result = self._model.transcribe(tmp_path)
        finally:
            os.unlink(tmp_path)

        if isinstance(result, str):
            return result.strip()
        if isinstance(result, list):
            return " ".join(str(r) for r in result).strip()
        return str(result).strip()
