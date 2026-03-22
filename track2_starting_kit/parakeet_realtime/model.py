#!/usr/bin/env python3
"""
Parakeet RNNT 120M — SAPC2 Track 2
====================================

Wraps NVIDIA Parakeet RNNT (or a FastConformer-Transducer variant) with the
5-method streaming interface required by the ingestion program.

STREAMING APPROACH:
  Uses NeMo's cache-aware encoder (cache_aware_stream_step) if the model's
  encoder supports setup_streaming_params(). This gives true streaming with
  bounded per-chunk latency — identical architecture to nemo_fastconformer/
  but with a pure RNNT decoder (no CTC head).

  PARTIAL RESULTS: RNNT decode is run on accumulated encoder outputs after
  every N chunks. This re-decodes the full sequence from scratch on each
  partial (no persistent RNNT decoder state across partials). This is
  acceptable because:
    - partial_interval_chunks=10 limits how often this runs (every 1s)
    - RNNT greedy decode is O(T × vocab) which is fast on CPU for short utterances
    - For true RNNT streaming with persistent state, use the NeMo streaming API

MODEL VARIANTS:
  nvidia/parakeet-rnnt-0.12b: 120M param FastConformer-Transducer.
  This model may NOT have setup_streaming_params() if it was trained without
  cache-aware attention. Run setup.sh Step 1 to confirm.

  If cache-aware streaming is not available, the model falls back to:
  - Accumulate ALL audio in buffer
  - Decode only at input_finished()
  - Partials are empty (or estimated from partial audio)
  This WILL be slower (full audio re-encode on each partial) — update
  partial_interval_chunks to a higher value if that's the case.

Required interface (called by ingestion program):
  __init__()                       — Load model weights (once)
  set_partial_callback(fn) -> None — Register partial result callback
  reset()             -> None      — Reset state per audio file
  accept_chunk(buf)   -> str       — Feed 100ms audio chunk
  input_finished()    -> str       — Signal end of audio, return text

To change settings, edit config.yaml (not this file).
"""

# =====================================================================
# Section 1: Venv Path Injection
# =====================================================================
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
import torch
from omegaconf import OmegaConf

try:
    import nemo.collections.asr as nemo_asr
except ImportError as exc:
    raise ImportError(
        "NeMo ASR is not installed. Run setup.sh first, or:\n"
        "  pip install 'nemo_toolkit[asr]>=2.5.0'"
    ) from exc

# =====================================================================
# Section 3: Config
# =====================================================================
_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_config = OmegaConf.load(_DIR / "config.yaml")


# =====================================================================
# Section 4: Helpers
# =====================================================================

def _rnnt_greedy_decode(model, encoder_output: torch.Tensor) -> str:
    """Run RNNT greedy decode on accumulated encoder output.

    Args:
        model: NeMo RNNT model (EncDecRNNTBPEModel or EncDecHybridRNNTCTCBPEModel)
        encoder_output: (1, D, T) encoder output tensor

    Returns decoded text string.
    """
    try:
        # encoder_output shape: (1, D, T) — NeMo convention
        enc_len = torch.tensor([encoder_output.size(2)], dtype=torch.long)
        # NeMo RNNT decoding: transcribe_from_encoder_outputs
        # Use the model's built-in RNNT greedy decoder
        # EncDecRNNTBPEModel.decoding is an RNNTDecoding instance
        hyps, _ = model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoder_output,
            encoded_lengths=enc_len,
        )
        if hyps and hyps[0]:
            return hyps[0][0].text if hasattr(hyps[0][0], "text") else str(hyps[0][0])
        return ""
    except Exception as e:
        # Fallback: try the model's transcribe API on the encoder output
        try:
            # Some NeMo versions expose decode_ids directly
            hyps = model.decoding.decode(
                encoder_output=encoder_output,
                encoded_lengths=torch.tensor([encoder_output.size(2)]),
            )
            if hyps:
                return hyps[0].text if hasattr(hyps[0], "text") else ""
        except Exception:
            pass
        return ""


def _ctc_greedy_decode(
    log_probs: torch.Tensor, vocabulary: List[str], blank_id: int
) -> str:
    """CTC greedy decode for hybrid models with a CTC head."""
    token_ids = torch.argmax(log_probs, dim=-1).tolist()
    prev = -1
    tokens = []
    for tid in token_ids:
        if tid != prev:
            if tid != blank_id:
                tokens.append(tid)
            prev = tid
    if not tokens:
        return ""
    try:
        pieces = [vocabulary[t] for t in tokens if t < len(vocabulary)]
        text = "".join(pieces).replace("▁", " ").strip()
        return text
    except IndexError:
        return ""


# =====================================================================
# Section 5: Model — Public Interface for the Ingestion Program
# =====================================================================
class Model:
    """Streaming ASR wrapping NeMo Parakeet RNNT cache-aware inference.

    Lifecycle (called by ingestion program):
      model.set_partial_callback(fn)           # register callback (once)
      model.reset()                             # prepare for new file
      for chunk in audio_chunks:
          partial = model.accept_chunk(chunk)   # returns partial text
      final = model.input_finished()            # returns final text
    """

    def __init__(self):
        self._partial_callback: Optional[Callable[[str], None]] = None

        # ── Torch settings ──────────────────────────────────────────
        n_threads = _config.inference.num_threads
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(n_threads)

        # ── Load model ───────────────────────────────────────────────
        model_name = _config.model.name
        model_class_name = _config.model.model_class
        print(f"Loading Parakeet RNNT: {model_name} ({model_class_name}) …")

        # Resolve model class
        model_class = getattr(nemo_asr.models, model_class_name, None)
        if model_class is None:
            # Try common fallbacks
            for cls_name in ["EncDecRNNTBPEModel", "EncDecHybridRNNTCTCBPEModel"]:
                cls = getattr(nemo_asr.models, cls_name, None)
                if cls is not None:
                    model_class = cls
                    print(f"  model_class {model_class_name!r} not found, using {cls_name}")
                    break

        if model_class is None:
            # Last resort: generic loader
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_name, map_location="cpu"
            )
        else:
            self._model = model_class.from_pretrained(
                model_name=model_name, map_location="cpu"
            )
        self._model.eval()
        self._model.to("cpu")

        # ── Streaming configuration ──────────────────────────────────
        self._streaming_supported = False
        if hasattr(self._model, "encoder") and hasattr(self._model.encoder, "setup_streaming_params"):
            try:
                self._model.encoder.setup_streaming_params()
                cfg = self._model.encoder.streaming_cfg
                self._hop_length = 160  # 10ms at 16kHz
                self._enc_chunk_frames = cfg.chunk_size[1]
                self._samples_per_enc_chunk = self._enc_chunk_frames * self._hop_length
                self._streaming_supported = True
                print(
                    f"  Cache-aware streaming enabled: "
                    f"{self._enc_chunk_frames} frames / {self._samples_per_enc_chunk} samples per step"
                )
            except Exception as e:
                print(f"  WARNING: setup_streaming_params() failed ({e})")
                print("  Falling back to batch mode (accumulate all audio, decode at end)")
        else:
            print("  WARNING: encoder.setup_streaming_params() not available.")
            print("  Falling back to batch mode (accumulate all audio, decode at end).")

        # ── Decoder type detection ────────────────────────────────────
        self._has_ctc = hasattr(self._model, "ctc_decoder") and self._model.ctc_decoder is not None
        self._partial_decoder = _config.model.partial_decoder  # "ctc" or "rnnt"
        self._decoder_type = _config.model.decoder_type        # final: "rnnt"

        if self._partial_decoder == "ctc" and not self._has_ctc:
            print("  WARNING: partial_decoder=ctc but model has no CTC head. Switching to rnnt.")
            self._partial_decoder = "rnnt"

        if self._has_ctc:
            self._vocabulary = list(self._model.ctc_decoder.vocabulary)
            self._blank_id = self._model.ctc_decoder.num_classes_with_blank - 1

        # ── Feature extraction ────────────────────────────────────────
        self._preprocessor = self._model.preprocessor
        self._preprocessor.eval()

        # ── State (reset per utterance) ───────────────────────────────
        self._sr = _config.audio.sample_rate
        self._audio_buffer: List[np.ndarray] = []
        self._encoder_outputs: List[torch.Tensor] = []
        self._cache_last_channel: Optional[torch.Tensor] = None
        self._cache_last_time: Optional[torch.Tensor] = None
        self._cache_last_channel_len: Optional[torch.Tensor] = None
        self._chunks_since_partial: int = 0
        self._total_chunks: int = 0
        self._last_partial: str = ""

        params = sum(p.numel() for p in self._model.parameters()) / 1e6
        print(
            f"Parakeet RNNT loaded ({params:.1f}M params, "
            f"streaming={'yes' if self._streaming_supported else 'no (batch fallback)'}, "
            f"partial={self._partial_decoder})"
        )

    # -----------------------------------------------------------------
    # Streaming Interface
    # -----------------------------------------------------------------

    def set_partial_callback(self, callback: Callable[[str], None]) -> None:
        self._partial_callback = callback

    def reset(self) -> None:
        """Reset state for a new audio file."""
        self._audio_buffer = []
        self._encoder_outputs = []
        if self._streaming_supported:
            caches = self._model.encoder.get_initial_cache_state(batch_size=1)
            self._cache_last_channel = caches[0]
            self._cache_last_time = caches[1]
            self._cache_last_channel_len = caches[2]
        else:
            self._cache_last_channel = None
            self._cache_last_time = None
            self._cache_last_channel_len = None
        self._chunks_since_partial = 0
        self._total_chunks = 0
        self._last_partial = ""

    def accept_chunk(self, audio_chunk: np.ndarray) -> str:
        """Feed one 100ms audio chunk (float32, 16kHz) and return partial text."""
        interval = _config.streaming.partial_interval_chunks
        min_chunks = _config.streaming.min_chunks_for_partial

        self._audio_buffer.append(audio_chunk)
        self._chunks_since_partial += 1
        self._total_chunks += 1

        if self._streaming_supported:
            self._try_encode()

        # Emit partial at the configured interval
        if (
            interval > 0
            and self._chunks_since_partial >= interval
            and self._total_chunks >= min_chunks
        ):
            if self._streaming_supported and self._encoder_outputs:
                self._last_partial = self._decode_partial()
            # In batch fallback mode, we don't have encoder outputs yet — skip partial
            self._chunks_since_partial = 0
            if self._last_partial and self._partial_callback is not None:
                self._partial_callback(self._last_partial)

        return self._last_partial

    def input_finished(self) -> str:
        """Signal end of audio. Returns final transcription."""
        if self._streaming_supported:
            # Flush encoder with silence padding
            flush_s = _config.streaming.flush_padding_s
            flush_samples = int(flush_s * self._sr)
            silence = np.zeros(flush_samples, dtype=np.float32)
            self._audio_buffer.append(silence)
            self._try_encode(flush=True)

            if not self._encoder_outputs:
                return ""
            text = self._decode_final()
        else:
            # Batch fallback: encode all buffered audio at once
            if not self._audio_buffer:
                return ""
            text = self._batch_transcribe()

        if self._partial_callback is not None:
            self._partial_callback(text)
        return text

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _try_encode(self, flush: bool = False) -> None:
        """Run cache-aware encoder steps on buffered audio."""
        if not self._audio_buffer:
            return
        buffered = np.concatenate(self._audio_buffer)
        step = self._samples_per_enc_chunk
        ctx = torch.inference_mode() if _config.inference.inference_mode else torch.no_grad()

        while True:
            n = len(buffered)
            if n == 0:
                break
            if not flush and n < step:
                break
            take = n if flush else step
            chunk = buffered[:take]
            buffered = buffered[take:]
            with ctx:
                self._run_encoder_step(chunk)
            if flush:
                break

        self._audio_buffer = [buffered] if len(buffered) > 0 else []

    def _run_encoder_step(self, audio: np.ndarray) -> None:
        """Extract features and run one cache-aware encoder step."""
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        length = torch.tensor([len(audio)], dtype=torch.long)

        features, feat_len = self._preprocessor(
            input_signal=audio_tensor,
            length=length,
        )

        (
            encoder_output,
            _enc_len,
            self._cache_last_channel,
            self._cache_last_time,
            self._cache_last_channel_len,
        ) = self._model.encoder.cache_aware_stream_step(
            features,
            feat_len,
            cache_last_channel=self._cache_last_channel,
            cache_last_time=self._cache_last_time,
            cache_last_channel_len=self._cache_last_channel_len,
        )

        if encoder_output.size(2) > 0:
            self._encoder_outputs.append(encoder_output)

    def _decode_partial(self) -> str:
        """Decode accumulated encoder outputs for partial result."""
        if not self._encoder_outputs:
            return ""
        enc = torch.cat(self._encoder_outputs, dim=2)  # (1, D, T)

        if self._partial_decoder == "ctc" and self._has_ctc:
            log_probs = self._model.ctc_decoder(encoder_output=enc)
            return _ctc_greedy_decode(log_probs[0], self._vocabulary, self._blank_id)
        else:
            # RNNT greedy decode (re-decode from scratch)
            with torch.inference_mode():
                return _rnnt_greedy_decode(self._model, enc)

    def _decode_final(self) -> str:
        """Decode accumulated encoder outputs for final result (RNNT)."""
        if not self._encoder_outputs:
            return ""
        enc = torch.cat(self._encoder_outputs, dim=2)  # (1, D, T)
        with torch.inference_mode():
            return _rnnt_greedy_decode(self._model, enc)

    def _batch_transcribe(self) -> str:
        """Batch fallback: encode and decode all buffered audio at once.

        Used when the encoder does not support cache-aware streaming.
        SLOW — re-encodes full audio. Only for models without streaming params.
        """
        all_audio = np.concatenate(self._audio_buffer)
        audio_tensor = torch.from_numpy(all_audio).unsqueeze(0)
        length = torch.tensor([len(all_audio)], dtype=torch.long)

        with torch.inference_mode():
            features, feat_len = self._preprocessor(
                input_signal=audio_tensor, length=length
            )
            encoded, enc_len = self._model.encoder(
                audio_signal=features, length=feat_len
            )
            return _rnnt_greedy_decode(self._model, encoded)
