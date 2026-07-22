#!/usr/bin/env python3
"""
Cache-aware streaming Model (NeMo) — Track 2 5-method interface.

======================================================================
STATUS: SKELETON — NOT YET RUN OR VERIFIED.
There is no NeMo on the dev box and no SAP data here, so the streaming core
below (the `conformer_stream_step` call + cache tensors) is written to NeMo's
*documented* cache-aware API but has NOT been executed. It MUST be validated
on-pod against the REAL `track2_starting_kit/local_decode.py` (both passes)
before ANY number from it is trusted (Stage B, docs/benchmark_plan.md; memory
`validate-against-real-harness`). Treat every line here as a hypothesis.
======================================================================

Why this model and not Parakeet-TDT/RNNT: the offline Parakeet checkpoints do
NOT support real cache-aware streaming (verified — docs/repo_analysis.md).
Running them on the 100 ms chunk path would be buffered re-run = fake streaming
= forbidden. The genuine streaming NVIDIA candidate is the cache-aware
FastConformer-RNNT (`nvidia/nemotron-speech-streaming-en-0.6b` /
`stt_en_fastconformer_hybrid_large_streaming_multi`) — the exact family the
prior Nemotron attempt used and lost on the streaming-export gap
(memory `nemotron-vs-zipformer-roadblock`). This skeleton exists to re-test
that path faithfully, not to fake it.

Design contract (must match track2_starting_kit/local_decode.py):
  __init__()               load model + preprocessor once (CPU for Track 2)
  set_partial_callback(fn) register fn(text:str); FIRE IT per emitted token
                           (TTFT depends on the callback, not the return value)
  reset()                  fresh encoder cache + decoder state per file
  accept_chunk(np.float32) 1600 samples (100 ms) -> partial str; state carried
  input_finished()         flush tail -> final str

Reference for the real streaming calls (port + verify on pod):
  NeMo examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py
  (conformer_stream_step, get_initial_cache_state, setup_streaming_params).
Carry forward the fixes the prior attempt already found: SOS/blank init, warmup
frame drop, drop_extra_pre_encoded on the first chunk.
"""

import numpy as np

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600  # 100 ms, as delivered by local_decode.py


class Model:
    """Cache-aware streaming ASR. SKELETON — verify on pod before trusting."""

    def __init__(self, checkpoint: str = "nvidia/nemotron-speech-streaming-en-0.6b"):
        # Lazy imports: keep this file import-safe on a box without NeMo/torch.
        import torch
        import nemo.collections.asr as nemo_asr

        self._torch = torch
        self._device = torch.device("cpu")  # Track 2 = CPU-only
        self._partial_callback = None

        print(f"[nemotron_streaming] loading {checkpoint} (CPU) …")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=checkpoint)
        self.model = self.model.to(self._device).eval()

        # A cache-aware model exposes conformer_stream_step. If it does not,
        # this checkpoint is NOT a streaming model — fail loud, never fake it.
        if not hasattr(self.model, "conformer_stream_step"):
            raise RuntimeError(
                f"{checkpoint} has no conformer_stream_step — it is not a "
                "cache-aware streaming model. Do NOT buffer-and-rerun it "
                "(that is forbidden fake streaming). Use a cache-aware "
                "FastConformer-RNNT checkpoint instead."
            )

        # Configure the streaming operating point (chunk / left-context). The
        # supported points are model-specific (e.g. 80/160/320/560/1120 ms);
        # pick the closest to the harness's 100 ms cadence on the pod.
        # self.model.encoder.setup_streaming_params(...)   # VERIFY exact args on pod

        # Raw-audio staging buffer: local_decode feeds 100 ms; the model may
        # need a different fixed chunk, so we accumulate until one model chunk
        # is available, then step once. This is NOT re-encoding history — the
        # encoder cache carries context across steps.
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._model_chunk_samples = CHUNK_SAMPLES  # TODO: set from streaming params on pod
        self._reset_stream_state()
        print("[nemotron_streaming] ready (SKELETON — unverified)")

    # ------------------------------------------------------------------
    def set_partial_callback(self, callback) -> None:
        self._partial_callback = callback

    def reset(self) -> None:
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._reset_stream_state()

    def _reset_stream_state(self):
        """Fresh encoder cache + decoder state for a new file. VERIFY on pod."""
        self._hyp_text = ""
        self._prev_hypotheses = None
        self._prev_pred_out = None
        self._first_chunk = True
        # cache tensors from the model's initial state:
        # (self._cache_ch, self._cache_t, self._cache_ch_len) = \
        #     self.model.encoder.get_initial_cache_state(batch_size=1)
        self._cache_ch = self._cache_t = self._cache_ch_len = None

    # ------------------------------------------------------------------
    def accept_chunk(self, audio_chunk: np.ndarray) -> str:
        """Feed 100 ms; step the model when a full model-chunk is buffered."""
        self._audio_buf = np.concatenate([self._audio_buf, audio_chunk.astype(np.float32)])
        while self._audio_buf.shape[0] >= self._model_chunk_samples:
            step_audio = self._audio_buf[: self._model_chunk_samples]
            self._audio_buf = self._audio_buf[self._model_chunk_samples :]
            text = self._stream_step(step_audio, is_last=False)
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
        """ONE cache-aware step. SKELETON — the real NeMo call goes here.

        Port from NeMo's speech_to_text_cache_aware_streaming_infer.py:
          1) preprocessor(audio) -> processed_signal, processed_signal_length
          2) conformer_stream_step(processed_signal=..., processed_signal_length=...,
                 cache_last_channel=self._cache_ch, cache_last_time=self._cache_t,
                 cache_last_channel_len=self._cache_ch_len,
                 keep_all_outputs=is_last, previous_hypotheses=self._prev_hypotheses,
                 previous_pred_out=self._prev_pred_out,
                 drop_extra_pre_encoded=(1 if self._first_chunk else 0),
                 return_transcription=True)
             -> (transcribed, cache_ch, cache_t, cache_ch_len, prev_hyps, pred_out, ...)
          3) update self._cache_*, self._prev_*, self._first_chunk=False
          4) set self._hyp_text to the running transcription

        The exact return signature is NeMo-version-sensitive; that is precisely
        what the on-pod validation against local_decode.py pins down. Until then
        this raises rather than emit a fabricated partial.
        """
        raise NotImplementedError(
            "nemotron_streaming._stream_step is an unverified skeleton. Wire the "
            "NeMo cache-aware call on the pod and validate via real local_decode.py "
            "before use. See docs/benchmark_plan.md Stage B."
        )
