#!/usr/bin/env python3
"""
Emformer RNNT — SAPC2 Track 2
==============================

Wraps torchaudio's EMFORMER_RNNT_BASE_LIBRISPEECH pipeline with the 5-method
streaming interface required by the ingestion program.

TRUE streaming: Emformer is a memory-efficient transformer designed for streaming
inference. The encoder processes fixed-size segments with a bounded right-context
(look-ahead). Each accept_chunk() call accumulates audio until a full segment is
ready, then runs the encoder + RNNT decoder on that segment only.

Architecture:
  - Emformer encoder: 16-frame segments, 4-frame right context, 80-dim log-mel
  - RNNT decoder: transducer, incremental token emission per segment
  - Vocabulary: SentencePiece BPE (LibriSpeech trained)

torchaudio 2.6+ API notes:
  - bundle.get_model() was removed. The model lives inside decoder.model.
  - decoder.infer(features, lengths, beam_width, state, hypothesis) — no model arg.
  - hypothesis is List[Tuple[tokens, score, state, lm_score]], NOT a Hypothesis object.
    Access tokens as hypothesis[0][0].
  - Feature extractor needs EXACTLY (segment_length + right_context_length) * hop_length
    but accounting for STFT framing: window_samples = (seg + ctx - 1) * hop
    This gives exactly (seg + ctx) frames with center-padded STFT.
  - stride_samples = (segment_length - 1) * hop_length (samples to advance per step)

Required interface (called by ingestion program):
  __init__()                       — Load model weights (once)
  set_partial_callback(fn) -> None — Register partial result callback
  reset()             -> None      — Reset state per audio file
  accept_chunk(buf)   -> str       — Feed 100ms audio chunk
  input_finished()    -> str       — Signal end of audio, return text

To change settings, edit config.yaml (not this file).
"""

# =====================================================================
# Section 1: Imports
# =====================================================================
import os
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

try:
    import torchaudio
    from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH
except ImportError as exc:
    raise ImportError(
        "torchaudio is not installed. It should be pre-installed in the "
        "competition Docker image. Install with: pip install torchaudio"
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
    """Streaming ASR wrapping torchaudio Emformer RNNT.

    Lifecycle (called by ingestion program):
      model.set_partial_callback(fn)           # register callback (once)
      model.reset()                             # prepare for new file
      for chunk in audio_chunks:
          partial = model.accept_chunk(chunk)   # returns partial text
      final = model.input_finished()            # returns final text

    Emformer streaming details:
      segment_length=16 frames, right_context_length=4 frames, hop_length=160 smp/frame.
      Feature extractor requires (seg+ctx-1)*hop = (16+4-1)*160 = 3040 samples per step
      to yield exactly 20 MFCC frames (with center-padded STFT).
      Stride per step: (seg-1)*hop = 15*160 = 2400 samples.
      At 16kHz: 3040 samples = 190ms window, 2400 samples = 150ms stride (40ms overlap).
      We receive 1600-sample (100ms) chunks, buffering until ≥3040 then draining by 2400.
    """

    def __init__(self):
        self._partial_callback: Optional[Callable[[str], None]] = None

        # ── Torch settings ──────────────────────────────────────────
        n_threads = _config.inference.num_threads
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(n_threads)

        # ── Load pipeline ────────────────────────────────────────────
        bundle = EMFORMER_RNNT_BASE_LIBRISPEECH
        print("Loading Emformer RNNT (downloading weights if needed) …")

        self._bundle = bundle
        # In torchaudio 2.6+, get_model() was removed.
        # The RNNT model lives inside the decoder as decoder.model.
        self._decoder = bundle.get_decoder()
        self._token_processor = bundle.get_token_processor()

        # ── Segment geometry ─────────────────────────────────────────
        # segment_length: encoder frames per processing step (post-subsampling)
        # right_context_length: look-ahead frames required by the encoder
        # hop_length: audio samples per MFCC frame
        #
        # For center-padded STFT: T output frames need (T-1)*hop_length audio samples.
        # So window_samples = (seg + ctx - 1) * hop yields exactly (seg + ctx) MFCC frames.
        seg = bundle.segment_length            # 16
        ctx = bundle.right_context_length      # 4
        hop = bundle.hop_length                # 160

        self._window_samples = (seg + ctx - 1) * hop   # 3040 — feed to feature extractor
        self._stride_samples = (seg - 1) * hop          # 2400 — advance per step

        # ── State initialised in reset() ─────────────────────────────
        self._sr = _config.audio.sample_rate
        self._audio_buffer: List[np.ndarray] = []
        self._decoder_state = None    # Emformer + RNNT decoder hidden state
        self._hypothesis = None       # running hypothesis (List[Tuple[tokens, ...]])
        self._feature_extractor = None
        self._last_partial: str = ""
        self._chunks_since_partial: int = 0
        self._total_chunks: int = 0

        params = sum(p.numel() for p in self._decoder.model.parameters()) / 1e6
        print(
            f"Emformer RNNT loaded ({params:.1f}M params, "
            f"window={self._window_samples}smp/{self._window_samples/self._sr*1000:.0f}ms, "
            f"stride={self._stride_samples}smp/{self._stride_samples/self._sr*1000:.0f}ms)"
        )

    # -----------------------------------------------------------------
    # Streaming Interface
    # -----------------------------------------------------------------

    def set_partial_callback(self, callback: Callable[[str], None]) -> None:
        self._partial_callback = callback

    def reset(self) -> None:
        """Reset state for a new audio file. Call once before each file."""
        self._audio_buffer = []
        self._decoder_state = None
        self._hypothesis = None
        # Re-create the streaming feature extractor to reset its internal state
        self._feature_extractor = self._bundle.get_streaming_feature_extractor()
        self._last_partial = ""
        self._chunks_since_partial = 0
        self._total_chunks = 0

    def accept_chunk(self, audio_chunk: np.ndarray) -> str:
        """Feed one 100ms audio chunk (float32, 16kHz) and return partial text."""
        interval = _config.streaming.partial_interval_chunks
        min_chunks = _config.streaming.min_chunks_for_partial

        self._audio_buffer.append(audio_chunk)
        self._chunks_since_partial += 1
        self._total_chunks += 1

        self._drain_buffer()

        if (
            interval > 0
            and self._chunks_since_partial >= interval
            and self._total_chunks >= min_chunks
            and self._hypothesis is not None
        ):
            self._last_partial = self._hypothesis_to_text()
            self._chunks_since_partial = 0
            if self._partial_callback is not None:
                self._partial_callback(self._last_partial)

        return self._last_partial

    def input_finished(self) -> str:
        """Signal end of audio. Flushes encoder and returns final text."""
        # Pad buffer with silence to push remaining audio through the encoder
        flush_s = _config.streaming.flush_padding_s
        flush_samples = int(flush_s * self._sr)
        remaining = np.concatenate(self._audio_buffer) if self._audio_buffer else np.array([], dtype=np.float32)
        # Pad enough to fill at least one more window
        needed = max(self._window_samples - len(remaining), 0) + flush_samples
        silence = np.zeros(needed, dtype=np.float32)
        self._audio_buffer.append(silence)

        self._drain_buffer(flush=True)

        text = self._hypothesis_to_text()
        if self._partial_callback is not None:
            self._partial_callback(text)
        return text

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _drain_buffer(self, flush: bool = False) -> None:
        """Process complete windows from the audio buffer."""
        if not self._audio_buffer:
            return

        buffered = np.concatenate(self._audio_buffer)
        ctx = torch.inference_mode() if _config.inference.inference_mode else torch.no_grad()

        while len(buffered) >= self._window_samples:
            window = buffered[: self._window_samples]
            buffered = buffered[self._stride_samples :]  # advance by stride
            with ctx:
                self._run_step(window)

        # In flush mode, process whatever's left (zero-pad to window_samples)
        if flush and len(buffered) > 0:
            padded = np.zeros(self._window_samples, dtype=np.float32)
            padded[: len(buffered)] = buffered
            with ctx:
                self._run_step(padded)
            buffered = np.array([], dtype=np.float32)

        self._audio_buffer = [buffered] if len(buffered) > 0 else []

    def _run_step(self, window: np.ndarray) -> None:
        """Run one Emformer encoder + RNNT decoder step.

        Args:
            window: exactly self._window_samples float32 audio samples
        """
        # Feature extraction: 1D tensor required (not batched) in torchaudio 2.6+
        waveform = torch.from_numpy(window)  # (window_samples,) — 1D
        features, lengths = self._feature_extractor(waveform)
        # features: (T, 80) where T = segment_length + right_context_length = 20
        # lengths: scalar tensor with value 20

        # RNNT streaming decode.
        # In torchaudio 2.6+: decoder.infer(features, lengths, beam_width, state, hypothesis)
        # No model argument — model is embedded in decoder.
        # Returns (hypotheses, state) where:
        #   hypotheses: List[Tuple[tokens, score, hypo_state, lm_score]]
        #   state: List[List[Tensor]] — Emformer + RNNT hidden states
        self._hypothesis, self._decoder_state = self._decoder.infer(
            features,
            lengths,
            _config.model.beam_width,
            state=self._decoder_state,
            hypothesis=self._hypothesis,
        )

    def _hypothesis_to_text(self) -> str:
        """Convert current best hypothesis tokens to text string."""
        if self._hypothesis is None:
            return ""
        try:
            # hypothesis is List[Tuple[tokens, score, state, lm_score]]
            # hypothesis[0] is the best hypothesis tuple
            # hypothesis[0][0] is the token list
            tokens = self._hypothesis[0][0]
            return self._token_processor(tokens)
        except Exception:
            return ""
