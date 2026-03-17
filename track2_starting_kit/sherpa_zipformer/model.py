#!/usr/bin/env python3
"""
Sherpa-ONNX Streaming Zipformer — SAPC2 Track 2
================================================

Wraps sherpa-onnx's OnlineRecognizer (Zipformer-Transducer) with the
5-method streaming interface required by the ingestion program.

Unlike the icefall-based streaming_zipformer/, this implementation:
  - Uses pre-exported ONNX models (no PyTorch at inference time)
  - Has a single clean dependency: `pip install sherpa-onnx`
  - Runs natively on CPU with ONNX Runtime (faster than PyTorch CPU)
  - Supports all three Zipformer size variants via config.yaml

This is the recommended submission vehicle for the competition.

Supported variants (set model.variant in config.yaml):
  standard  — ~70M params, LibriSpeech-trained (official baseline)
  kroko     — Zipformer2 architecture, edge-optimised
  small     — ~20M params, Pareto latency anchor

How it streams:
  The Zipformer-Transducer is a true streaming model. Each call to
  accept_chunk() feeds audio directly to the encoder and fires the
  partial callback as soon as new tokens are decoded. No buffering.

Required interface (called by ingestion program):
  __init__()                       — Load model weights (once)
  set_partial_callback(fn) -> None — Register partial result callback
  reset()             -> None      — Reset state per audio file
  accept_chunk(buf)   -> str       — Feed 100 ms audio chunk
  input_finished()    -> str       — Signal end of audio, return text

To change settings, edit config.yaml (not this file).

Directory layout after running setup.sh:
  sherpa_zipformer/
  ├── model.py       ← this file
  ├── config.yaml    ← all tunable settings
  ├── setup.sh       ← installs sherpa-onnx and downloads ONNX weights
  └── weights/
      └── <variant>/
          ├── encoder.onnx
          ├── decoder.onnx
          ├── joiner.onnx
          └── tokens.txt
"""

# =====================================================================
# Section 1: Imports
# =====================================================================
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from omegaconf import OmegaConf

try:
    import sherpa_onnx
except ImportError as exc:
    raise ImportError(
        "sherpa-onnx is not installed. Run setup.sh first, or:\n"
        "  pip install sherpa-onnx"
    ) from exc

# =====================================================================
# Section 2: Config
# =====================================================================
_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_config = OmegaConf.load(_DIR / "config.yaml")


# =====================================================================
# Section 3: Model — Public Interface for the Ingestion Program
# =====================================================================
class Model:
    """Streaming ASR model wrapping sherpa-onnx Zipformer-Transducer.

    True streaming: audio is processed frame-by-frame as chunks arrive;
    partial callbacks fire as soon as new tokens are decoded.

    Lifecycle (called by ingestion program):
      model.set_partial_callback(fn)           # register callback (once)
      model.reset()                             # prepare for new file
      for chunk in audio_chunks:
          partial = model.accept_chunk(chunk)   # returns partial text
      final = model.input_finished()            # returns final text
    """

    def __init__(self):
        variant = _config.model.variant
        weights_dir = _DIR / "weights" / variant
        print(f"Loading sherpa-onnx Zipformer ({variant}) from {weights_dir} …")

        self._partial_callback: Optional[Callable[[str], None]] = None
        self._recognizer = self._build_recognizer(weights_dir)
        self._stream = None  # created in reset()

        print(f"sherpa-onnx Zipformer ({variant}) loaded (cpu)")

    # -----------------------------------------------------------------
    # Streaming Interface (called by the ingestion program)
    # -----------------------------------------------------------------

    def set_partial_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback for partial results: callback(text: str)."""
        self._partial_callback = callback

    def reset(self) -> None:
        """Reset state for a new audio file. Call once before each file."""
        self._stream = self._recognizer.create_stream()

    def accept_chunk(self, audio_chunk: np.ndarray) -> str:
        """Feed one 100 ms audio chunk (float32, 16 kHz) and return partial text.

        Feeds the chunk to the encoder and decodes any newly available
        frames. Fires the partial callback if the hypothesis changed.
        """
        self._stream.accept_waveform(
            sample_rate=_config.audio.sample_rate,
            waveform=audio_chunk,
        )
        self._drain()
        text = self._recognizer.get_result(self._stream).text.strip()
        if text and self._partial_callback is not None:
            self._partial_callback(text)
        return text

    def input_finished(self) -> str:
        """Signal end of audio. Flushes encoder tail, returns final text."""
        # Append 0.3 s of silence to flush the encoder's right-context frames.
        tail = np.zeros(
            int(_config.audio.sample_rate * 0.3), dtype=np.float32
        )
        self._stream.accept_waveform(
            sample_rate=_config.audio.sample_rate,
            waveform=tail,
        )
        self._stream.input_finished()
        self._drain()
        final_text = self._recognizer.get_result(self._stream).text.strip()
        if self._partial_callback is not None:
            self._partial_callback(final_text)
        return final_text

    # -----------------------------------------------------------------
    # Private Helpers
    # -----------------------------------------------------------------

    def _drain(self) -> None:
        """Decode all frames that the recognizer has ready."""
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode(self._stream)

    def _build_recognizer(self, weights_dir: Path) -> sherpa_onnx.OnlineRecognizer:
        """Construct and return an OnlineRecognizer from config + weights."""
        cfg = _config
        w = weights_dir

        model_cfg = sherpa_onnx.OnlineModelConfig(
            transducer=sherpa_onnx.OnlineTransducerModelConfig(
                encoder=str(w / cfg.model.encoder_file),
                decoder=str(w / cfg.model.decoder_file),
                joiner=str(w / cfg.model.joiner_file),
            ),
            tokens=str(w / cfg.model.tokens_file),
            num_threads=cfg.model.num_threads,
            provider="cpu",
        )

        feat_cfg = sherpa_onnx.FeatureExtractorConfig(
            sampling_rate=cfg.audio.sample_rate,
            feature_dim=cfg.audio.feature_dim,
        )

        recognizer_cfg = sherpa_onnx.OnlineRecognizerConfig(
            feat_config=feat_cfg,
            model_config=model_cfg,
            decoding_method=cfg.decoding.method,
            max_active_paths=cfg.decoding.max_active_paths,
            enable_endpoint_detection=cfg.endpoint.enable,
            rule1_min_trailing_silence=cfg.endpoint.rule1_min_trailing_silence,
            rule2_min_trailing_silence=cfg.endpoint.rule2_min_trailing_silence,
            rule3_min_utterance_length=cfg.endpoint.rule3_min_utterance_length,
        )

        return sherpa_onnx.OnlineRecognizer(recognizer_cfg)
