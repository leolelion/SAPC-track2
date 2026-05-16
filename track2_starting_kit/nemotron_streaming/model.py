#!/usr/bin/env python3
"""
Nemotron Streaming ASR 0.6B — SAPC2 Track 2 Submission
======================================================

Cache-aware streaming FastConformer-RNNT (nvidia/nemotron-speech-streaming-en-0.6b)
with att_context_size=[70, 1] (160ms model chunks) on CPU, 4 threads.

Streaming approach:
  - accept_chunk() receives 100ms (1600 samples) from the SAPC2 harness
  - Audio accumulates until enough mel frames exist for a model step
  - conformer_stream_step() runs one encoder+decoder step with cache carry-forward
  - Partial callback fires only when hypothesis text changes

Threading contract:
  All five interface methods are called from the Decoder thread only.
  _partial_callback is invoked exclusively from accept_chunk(), never from
  input_finished() or any background thread. No locking required.
"""

# =====================================================================
# Thread configuration — MUST happen before torch creates its threadpool
# =====================================================================
import os
import sys
import glob as _glob

_THREADS = int(os.environ.get("SAPC2_THREADS", "4"))

import torch

torch.set_num_threads(_THREADS)
torch.set_num_interop_threads(1)

# =====================================================================
# Venv path injection (NeMo installed via setup.sh into a local venv)
# =====================================================================
_DIR = os.path.dirname(os.path.abspath(__file__))
_venv_candidates = _glob.glob(
    os.path.join(_DIR, "venv", "lib", "python3.*", "site-packages")
) or _glob.glob(
    os.path.join(_DIR, "venv", "*", "lib", "python3.*", "site-packages")
)
if _venv_candidates:
    sys.path.insert(0, _venv_candidates[0])

# =====================================================================
# Imports
# =====================================================================
import numpy as np
from omegaconf import open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

# =====================================================================
# Constants
# =====================================================================
SAMPLE_RATE = 16000
HOP_SAMPLES = 160        # 10ms hop at 16kHz
WINDOW_SAMPLES = 400      # 25ms window at 16kHz
ATT_CONTEXT_SIZE = [70, 1]
HF_MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"


def _extract_text(transcribed):
    """Extract text string from conformer_stream_step output."""
    if not transcribed:
        return ""
    item = transcribed[0]
    if isinstance(item, Hypothesis):
        text = item.text or ""
    else:
        text = str(item) if item is not None else ""
    # Strip whitespace, collapse double spaces
    return " ".join(text.split())


def _min_samples_for_frames(n_frames):
    """Conservative lower bound on audio samples needed to produce n_frames mel frames."""
    if n_frames <= 0:
        return 0
    return (n_frames - 1) * HOP_SAMPLES + WINDOW_SAMPLES


class Model:
    """Cache-aware streaming ASR with Nemotron 0.6B."""

    def __init__(self):
        print(f"Loading {HF_MODEL_NAME} (CPU, {_THREADS} threads) ...")

        # ── Load model ──────────────────────────────────────────────
        self._model = ASRModel.from_pretrained(HF_MODEL_NAME, map_location="cpu")
        self._model.to("cpu")
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        # ── Set streaming context size ──────────────────────────────
        self._model.encoder.set_default_att_context_size(ATT_CONTEXT_SIZE)
        actual = self._model.encoder.att_context_size
        assert actual == ATT_CONTEXT_SIZE, (
            f"att_context_size mismatch: expected {ATT_CONTEXT_SIZE}, got {actual}"
        )

        # ── Configure greedy decoding ───────────────────────────────
        decoding_cfg = self._model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.preserve_alignments = False
            if hasattr(self._model, "joint"):
                decoding_cfg.greedy.max_symbols = 10
                decoding_cfg.fused_batch_size = -1
        self._model.change_decoding_strategy(decoding_cfg)

        # ── Configure preprocessor for streaming (deterministic) ────
        self._model.preprocessor.featurizer.dither = 0.0
        self._model.preprocessor.featurizer.pad_to = 0

        # ── Read streaming config ───────────────────────────────────
        cfg = self._model.encoder.streaming_cfg
        self._chunk_size = cfg.chunk_size                    # [9, 16] mel frames
        self._shift_size = cfg.shift_size                    # [9, 16]
        self._pre_encode_cache = cfg.pre_encode_cache_size   # [0, 9]
        self._drop_extra = cfg.drop_extra_pre_encoded        # 2
        self._n_features = self._model.cfg.preprocessor.features  # 128 for nemotron

        # Minimum frames for a chunk to produce encoder output after downsampling
        # CacheAwareStreamingAudioBuffer uses this to skip too-short final chunks
        subsampling = self._model.encoder.pre_encode
        if hasattr(subsampling, 'get_sampling_frames'):
            self._sampling_frames = subsampling.get_sampling_frames()  # [1, 8]
        else:
            self._sampling_frames = None

        print(f"  streaming_cfg: chunk_size={list(self._chunk_size)}, "
              f"shift_size={list(self._shift_size)}, "
              f"pre_encode_cache={list(self._pre_encode_cache)}, "
              f"drop_extra={self._drop_extra}")
        print(f"  att_context_size={actual}, threads={_THREADS}")

        # ── Per-utterance state (initialized in reset) ──────────────
        self._partial_callback = lambda _text: None
        self._reset_state()

    # -----------------------------------------------------------------
    # Public interface (5 methods)
    # -----------------------------------------------------------------

    def set_partial_callback(self, callback) -> None:
        """Register callback for partial results: callback(text: str)."""
        self._partial_callback = callback

    def reset(self) -> None:
        """Reset streaming state for a new audio file."""
        self._reset_state()

    def accept_chunk(self, audio_chunk: np.ndarray) -> str:
        """Accept 100ms audio chunk (float32, 16kHz, mono). Return current hypothesis."""
        self._raw_chunks.append(audio_chunk)
        self._total_samples += len(audio_chunk)
        self._run_steps(is_final=False)
        return self._last_emitted

    def input_finished(self) -> str:
        """Signal end of audio. Flush remaining frames and return final transcription."""
        self._run_steps(is_final=True)
        return self._last_emitted

    # -----------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------

    def _reset_state(self):
        self._raw_chunks = []
        self._total_samples = 0
        self._buffer_idx = 0       # current position in mel frames
        self._step_num = 0
        self._last_emitted = ""

        # Encoder cache
        cache = self._model.encoder.get_initial_cache_state(batch_size=1)
        self._cache_last_channel = cache[0]
        self._cache_last_time = cache[1]
        self._cache_last_channel_len = cache[2]
        self._previous_hypotheses = None
        self._pred_out_stream = None

        # Feature cache (avoid redundant preprocessing)
        self._cached_features = None
        self._cached_feat_len = 0
        self._cached_n_samples = 0

    # -----------------------------------------------------------------
    # Streaming config accessors
    # -----------------------------------------------------------------

    def _cfg_val(self, cfg_list, step):
        """Return cfg_list[0] for step 0, cfg_list[1] otherwise (handles scalar too)."""
        if isinstance(cfg_list, (list, tuple)):
            return cfg_list[0] if step == 0 else cfg_list[1]
        return cfg_list

    # -----------------------------------------------------------------
    # Core streaming loop
    # -----------------------------------------------------------------

    def _run_steps(self, is_final=False):
        """Run as many conformer_stream_step calls as current audio allows."""
        if not self._raw_chunks:
            return

        # Early exit: not enough audio for the next model step
        cs = self._cfg_val(self._chunk_size, self._step_num)
        needed_frames = self._buffer_idx + cs
        needed_samples = _min_samples_for_frames(needed_frames)
        if not is_final and self._total_samples < needed_samples:
            return

        # Preprocess all accumulated audio → mel features
        if self._cached_n_samples != self._total_samples:
            all_audio = np.concatenate(self._raw_chunks)
            audio_t = torch.from_numpy(all_audio).unsqueeze(0)  # [1, T]
            audio_len = torch.tensor([len(all_audio)], dtype=torch.long)
            with torch.no_grad():
                features, feat_len = self._model.preprocessor(
                    input_signal=audio_t, length=audio_len
                )
            # features: [1, n_features, T_frames]
            # feat_len: valid frame count (may be < T_frames due to padding)
            self._cached_features = features
            self._cached_feat_len = feat_len.item()
            self._cached_n_samples = self._total_samples

        total_frames = self._cached_feat_len

        while True:
            cs = self._cfg_val(self._chunk_size, self._step_num)
            ss = self._cfg_val(self._shift_size, self._step_num)
            pcs = self._cfg_val(self._pre_encode_cache, self._step_num)

            # Check if enough frames for this step
            if self._buffer_idx + cs > total_frames:
                if is_final and self._buffer_idx < total_frames:
                    # Partial last chunk: use whatever remains, but only if
                    # enough frames to produce encoder output after downsampling
                    remaining = total_frames - self._buffer_idx
                    if self._sampling_frames is not None:
                        min_frames = self._cfg_val(self._sampling_frames, self._step_num)
                        if remaining < min_frames:
                            break  # Too short — matches CacheAwareStreamingAudioBuffer behavior
                    cs = remaining
                else:
                    break

            # Extract mel chunk [buffer_idx : buffer_idx + cs]
            chunk_end = self._buffer_idx + cs
            mel_chunk = self._cached_features[:, :, self._buffer_idx:chunk_end]

            # Build pre-encode cache
            if self._step_num == 0:
                if pcs > 0:
                    cache_pre = torch.zeros(
                        1, self._n_features, pcs,
                        device=mel_chunk.device, dtype=mel_chunk.dtype,
                    )
                else:
                    cache_pre = mel_chunk.new_empty(1, self._n_features, 0)
            else:
                cache_start = max(0, self._buffer_idx - pcs)
                cache_pre = self._cached_features[:, :, cache_start:self._buffer_idx]
                if cache_pre.shape[2] < pcs:
                    pad = torch.zeros(
                        1, self._n_features, pcs - cache_pre.shape[2],
                        device=mel_chunk.device, dtype=mel_chunk.dtype,
                    )
                    cache_pre = torch.cat([pad, cache_pre], dim=2)

            # Assemble: [pre_encode_cache | new_mel_chunk]
            full_chunk = torch.cat([cache_pre, mel_chunk], dim=2)
            chunk_lengths = torch.tensor([full_chunk.shape[2]], dtype=torch.long)

            # keep_all_outputs=True only on the very last step (signals decoder to finalize)
            next_idx = self._buffer_idx + ss
            keep_all = is_final and (next_idx >= total_frames)

            # drop_extra_pre_encoded: 0 on step 0 (pad_and_drop_preencoded=False)
            drop_extra = 0 if self._step_num == 0 else self._drop_extra

            with torch.inference_mode():
                (
                    self._pred_out_stream,
                    transcribed_texts,
                    self._cache_last_channel,
                    self._cache_last_time,
                    self._cache_last_channel_len,
                    self._previous_hypotheses,
                ) = self._model.conformer_stream_step(
                    processed_signal=full_chunk,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=self._cache_last_channel,
                    cache_last_time=self._cache_last_time,
                    cache_last_channel_len=self._cache_last_channel_len,
                    keep_all_outputs=keep_all,
                    previous_hypotheses=self._previous_hypotheses,
                    previous_pred_out=self._pred_out_stream,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )

            text = _extract_text(transcribed_texts)

            # Fire callback from accept_chunk only, when text changes
            if not is_final and text != self._last_emitted:
                self._partial_callback(text)
            self._last_emitted = text

            # Advance
            self._buffer_idx += ss
            self._step_num += 1

            # Exit after partial last chunk
            if is_final and chunk_end >= total_frames:
                break
