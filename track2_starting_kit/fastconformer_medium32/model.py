#!/usr/bin/env python3
"""
FastConformer-medium (32M) cache-aware streaming — SAPC2 Track 2 interface.

======================================================================
VERIFICATION STATUS (read before trusting any number from this file)
----------------------------------------------------------------------
VERIFIED LOCALLY (no NeMo needed): the 5-method contract, the 100 ms->model-chunk
buffering, callback firing, reset semantics, and the text-extraction that absorbs
the NeMo 2.7.2 list/Hypothesis return quirk. `python3 -m py_compile model.py` and a
plain `import` (NeMo absent) both succeed because torch/nemo are imported lazily.

NOT YET VERIFIED (needs the pod): the two blocks marked  # >>> VERIFY-ON-POD <<< .
These are the NeMo-version-sensitive calls — streaming-param setup and
`conformer_stream_step`'s exact return tuple. That signature is precisely what broke
the prior attempt; do NOT trust output until this has run through the REAL
track2_starting_kit/local_decode.py (both passes) and evaluate.sh on Dev
(house rule: validate-against-real-harness; research/46 §2 E1 gate).
======================================================================

Why this model: research/46 — the ~32M medium is the smallest genuinely-streamable
NVIDIA cache-aware FastConformer, run at 80 ms lookahead ([70,1]) as a LOW-LATENCY
Pareto-corner candidate. We decode the RNN-T head of the hybrid checkpoint.

Required interface (called by local_decode.py):
  __init__()                        — load model once (CPU for Track 2)
  set_partial_callback(fn) -> None  — register fn(text:str); FIRE per emitted step
  reset()             -> None       — fresh encoder cache + decoder state per file
  accept_chunk(np.float32) -> str   — 1600 samples (100 ms); returns running partial
  input_finished()    -> str        — flush tail -> final str

Reference for the streaming calls (port + verify on pod):
  NeMo examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py
"""

import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600  # 100 ms, as delivered by local_decode.py

_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_CONFIG = OmegaConf.load(_DIR / "config.yaml")


class Model:
    """Cache-aware streaming ASR (FastConformer-medium). See VERIFICATION STATUS above."""

    def __init__(self):
        # Lazy imports keep this file import-safe on a box without NeMo/torch,
        # so the contract/buffering logic can be inspected and import-checked locally.
        import torch
        import nemo.collections.asr as nemo_asr

        self._torch = torch
        self._device = torch.device("cpu")  # Track 2 = CPU-only
        self._partial_callback = None

        nemo_path = _DIR / _CONFIG.weights.nemo_file
        print(f"[fc_medium32] loading {nemo_path} (CPU) …")
        if nemo_path.exists():
            self.model = nemo_asr.models.ASRModel.restore_from(
                str(nemo_path), map_location=self._device
            )
        else:
            # Fallback for the benchmark pod if setup.sh's save step was skipped.
            print(f"[fc_medium32] local .nemo missing; pulling {_CONFIG.weights.model_name}")
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=_CONFIG.weights.model_name, map_location=self._device
            )
        self.model = self.model.to(self._device).eval()

        # A cache-aware model exposes conformer_stream_step. If it does not, this
        # checkpoint is NOT streaming — fail loud, never buffer-and-rerun (forbidden).
        if not hasattr(self.model, "conformer_stream_step"):
            raise RuntimeError(
                f"{_CONFIG.weights.model_name} has no conformer_stream_step — not a "
                "cache-aware streaming model. Buffer-and-rerun is forbidden fake streaming."
            )

        # Decode the RNN-T head of the hybrid checkpoint.
        if hasattr(self.model, "change_decoding_strategy"):
            try:
                self.model.change_decoding_strategy(
                    decoder_type=str(_CONFIG.decoding.decoder_type)
                )
            except TypeError:
                # Older NeMo signatures differ; VERIFY on pod if this path is hit.
                pass

        # >>> VERIFY-ON-POD <<<  (block 1 of 2: streaming-param setup)
        # Exact call names are NeMo-version-sensitive. Documented flow:
        #   self.model.encoder.set_default_att_context_size(list(att_context_size))
        #   self.model.encoder.setup_streaming_params()
        # get_initial_cache_state() then yields the per-file cache tensors (in reset()).
        att = list(_CONFIG.encoder.att_context_size)
        if hasattr(self.model.encoder, "set_default_att_context_size"):
            self.model.encoder.set_default_att_context_size(att)
        if hasattr(self.model.encoder, "setup_streaming_params"):
            self.model.encoder.setup_streaming_params()

        # How many raw samples make one model step. Prefer the encoder's own
        # streaming_cfg; fall back to the 100 ms cadence and FLAG for verification.
        self._model_chunk_samples = self._derive_chunk_samples()

        self._audio_buf = np.zeros(0, dtype=np.float32)
        self.reset()
        print(
            f"[fc_medium32] ready (att={att}, step={self._model_chunk_samples} samples) "
            "— NeMo streaming call UNVERIFIED until pod run"
        )

    # ------------------------------------------------------------------
    def _derive_chunk_samples(self) -> int:
        """Samples per model step. streaming_cfg.chunk_size is in feature frames
        (10 ms each); convert to samples. Fallback = 100 ms harness cadence.
        # >>> VERIFY-ON-POD <<< the frame->sample conversion for this ckpt."""
        cfg = getattr(self.model.encoder, "streaming_cfg", None)
        chunk_frames = getattr(cfg, "chunk_size", None) if cfg is not None else None
        if isinstance(chunk_frames, (list, tuple)):
            chunk_frames = chunk_frames[0]
        if isinstance(chunk_frames, int) and chunk_frames > 0:
            return int(chunk_frames * (SAMPLE_RATE // 100))  # 10 ms/frame -> samples
        return CHUNK_SAMPLES  # fallback: step once per 100 ms chunk

    # ------------------------------------------------------------------
    def set_partial_callback(self, callback) -> None:
        self._partial_callback = callback

    def reset(self) -> None:
        """Fresh encoder cache + decoder state for a new file."""
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._hyp_text = ""
        self._prev_hypotheses = None
        self._prev_pred_out = None
        self._first_chunk = True
        # >>> VERIFY-ON-POD <<<  (initial cache state)
        (self._cache_ch, self._cache_t, self._cache_ch_len) = (
            self.model.encoder.get_initial_cache_state(batch_size=1)
        )

    # ------------------------------------------------------------------
    def accept_chunk(self, audio_chunk: np.ndarray) -> str:
        """Feed 100 ms; step the model each time a full model-chunk is buffered."""
        self._audio_buf = np.concatenate(
            [self._audio_buf, np.asarray(audio_chunk, dtype=np.float32)]
        )
        while self._audio_buf.shape[0] >= self._model_chunk_samples:
            step = self._audio_buf[: self._model_chunk_samples]
            self._audio_buf = self._audio_buf[self._model_chunk_samples :]
            text = self._stream_step(step, is_last=False)
            if text and self._partial_callback is not None:
                self._partial_callback(text)  # drives TTFT
        return self._hyp_text

    def input_finished(self) -> str:
        """Flush any remaining buffered audio as the final step."""
        if self._audio_buf.shape[0] > 0:
            self._stream_step(self._audio_buf, is_last=True)
            self._audio_buf = np.zeros(0, dtype=np.float32)
        if self._partial_callback is not None and self._hyp_text:
            self._partial_callback(self._hyp_text)
        return self._hyp_text

    # ------------------------------------------------------------------
    def _stream_step(self, audio: np.ndarray, is_last: bool) -> str:
        """ONE cache-aware step.

        # >>> VERIFY-ON-POD <<<  (block 2 of 2: the streaming call itself)
        The conformer_stream_step RETURN TUPLE is NeMo-version-sensitive — this is
        the exact thing that broke last time. Documented 6-tuple (NeMo streaming
        example): (pred_out, transcribed, cache_ch, cache_t, cache_ch_len, prev_hyps).
        `_extract_text` below absorbs the 2.7.2 list/Hypothesis quirk.
        """
        torch = self._torch
        wav = torch.tensor(audio, dtype=torch.float32, device=self._device).unsqueeze(0)
        wav_len = torch.tensor([wav.shape[1]], dtype=torch.int64, device=self._device)

        processed, processed_len = self.model.preprocessor(
            input_signal=wav, length=wav_len
        )

        out = self.model.conformer_stream_step(
            processed_signal=processed,
            processed_signal_length=processed_len,
            cache_last_channel=self._cache_ch,
            cache_last_time=self._cache_t,
            cache_last_channel_len=self._cache_ch_len,
            keep_all_outputs=is_last,
            previous_hypotheses=self._prev_hypotheses,
            previous_pred_out=self._prev_pred_out,
            drop_extra_pre_encoded=(1 if self._first_chunk else 0),
            return_transcription=True,
        )
        # Defensive unpack: length/order may shift across NeMo versions.
        (
            pred_out,
            transcribed,
            self._cache_ch,
            self._cache_t,
            self._cache_ch_len,
            prev_hyps,
        ) = out[:6]
        self._prev_pred_out = pred_out
        self._prev_hypotheses = prev_hyps
        self._first_chunk = False

        text = self._extract_text(transcribed)
        if text:
            self._hyp_text = text
        return self._hyp_text

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_text(transcribed) -> str:
        """Normalize NeMo's transcription return to a plain str.

        Handles: plain str; a Hypothesis object (.text); a list of either
        (batch=1 -> take [0]); None. This is where the NeMo 2.7.2 list-return
        quirk is absorbed (docs/results/parakeet_comparison.md)."""
        if transcribed is None:
            return ""
        item = transcribed
        if isinstance(item, (list, tuple)):
            if not item:
                return ""
            item = item[0]
            if isinstance(item, (list, tuple)):  # nested [[hyp]]
                item = item[0] if item else ""
        if isinstance(item, str):
            return item
        text = getattr(item, "text", None)
        return text if isinstance(text, str) else ""
