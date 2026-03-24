#!/usr/bin/env python3
"""
NeMo FastConformer Hybrid Medium Streaming — SAPC2 Track 2
==========================================================

Cache-aware streaming FastConformer with 80ms look-ahead.
32M parameters. Supports CTC (fast) and RNNT (accurate) decoders.

TRUE streaming: each accept_chunk() processes only the new 100ms audio window,
reusing cached encoder activations from previous chunks. No audio is re-processed.

Model: nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms

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

def _ctc_greedy_decode(
    log_probs: torch.Tensor, vocabulary: List[str], blank_id: int
) -> str:
    """CTC greedy decode: argmax collapse + blank removal.

    Args:
        log_probs: (T, vocab_size+1) log probabilities
        vocabulary: list of 1024 BPE pieces (no blank entry)
        blank_id: index of the blank token (1024 for this model)
    """
    token_ids = torch.argmax(log_probs, dim=-1).tolist()  # (T,)
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
    """Streaming ASR wrapping NeMo FastConformer cache-aware inference.

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

        # ── Load pretrained model ────────────────────────────────────
        model_name = _config.model.name
        local_nemo = _DIR / "model.nemo"
        if local_nemo.exists():
            print(f"Loading NeMo FastConformer from local file: {local_nemo}")
            self._model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
                restore_path=str(local_nemo),
                map_location="cpu",
            )
        else:
            print(f"Loading NeMo FastConformer: {model_name} (from HuggingFace) …")
            self._model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
                model_name=model_name,
                map_location="cpu",
            )
        self._model.eval()
        self._model.to("cpu")

        # ── Streaming configuration ──────────────────────────────────
        # The 80ms model was trained with att_context_size=[70, 13]; that is
        # the only value the encoder accepts — don't override it.
        # Initialise streaming params so streaming_cfg is populated.
        self._model.encoder.setup_streaming_params()
        cfg = self._model.encoder.streaming_cfg
        # chunk_size[1] is the per-step input frame count for non-first chunks.
        # With hop_length=160 (10ms): 112 frames × 160 = 17920 samples ≈ 1.12s
        self._hop_length = 160  # 10ms at 16kHz
        self._enc_chunk_frames = cfg.chunk_size[1]  # 112 input frames per step
        self._samples_per_enc_chunk = self._enc_chunk_frames * self._hop_length

        # ── Decoder type ─────────────────────────────────────────────
        self._decoder_type = _config.model.decoder_type  # "ctc" or "rnnt"

        # ── Vocabulary (for CTC decoding) ────────────────────────────
        # ctc_decoder.vocabulary has 1024 BPE pieces; blank is the extra
        # class at index num_classes_with_blank - 1 = 1024.
        self._vocabulary = list(self._model.ctc_decoder.vocabulary)
        self._blank_id = self._model.ctc_decoder.num_classes_with_blank - 1

        # ── Feature extraction (NeMo's built-in preprocessor) ────────
        self._preprocessor = self._model.preprocessor
        self._preprocessor.eval()

        # ── Buffer and state ─────────────────────────────────────────
        self._sr = _config.audio.sample_rate
        self._audio_buffer: List[np.ndarray] = []
        self._encoder_outputs: List[torch.Tensor] = []
        # Caches initialised properly in reset(); set to None for now.
        self._cache_last_channel: Optional[torch.Tensor] = None
        self._cache_last_time: Optional[torch.Tensor] = None
        self._cache_last_channel_len: Optional[torch.Tensor] = None
        self._chunks_since_partial: int = 0
        self._total_chunks: int = 0
        self._last_partial: str = ""

        params = sum(p.numel() for p in self._model.parameters()) / 1e6
        print(f"NeMo FastConformer loaded ({params:.1f}M params, decoder={self._decoder_type})")

    # -----------------------------------------------------------------
    # Streaming Interface
    # -----------------------------------------------------------------

    def set_partial_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback for partial results: callback(text: str)."""
        self._partial_callback = callback

    def reset(self) -> None:
        """Reset state for a new audio file. Call once before each file."""
        self._audio_buffer = []
        self._encoder_outputs = []
        # Initialise encoder caches to the model's zero-state
        caches = self._model.encoder.get_initial_cache_state(batch_size=1)
        self._cache_last_channel = caches[0]
        self._cache_last_time = caches[1]
        self._cache_last_channel_len = caches[2]
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

        # Try to run one encoder step on the buffered audio
        self._try_encode()

        # Emit partial at the configured interval
        if (
            interval > 0
            and self._chunks_since_partial >= interval
            and self._total_chunks >= min_chunks
            and self._encoder_outputs
        ):
            self._last_partial = self._decode_accumulated()
            self._chunks_since_partial = 0
            if self._partial_callback is not None:
                self._partial_callback(self._last_partial)

        return self._last_partial

    def input_finished(self) -> str:
        """Signal end of audio. Flushes encoder and returns final text."""
        # Append silence padding to flush the encoder's right context
        flush_s = _config.streaming.flush_padding_s
        flush_samples = int(flush_s * self._sr)
        silence = np.zeros(flush_samples, dtype=np.float32)
        self._audio_buffer.append(silence)

        # Process any remaining buffered audio
        self._try_encode(flush=True)

        if not self._encoder_outputs:
            return ""

        # Final decode — CTC with full context
        text = self._decode_accumulated()
        if self._partial_callback is not None:
            self._partial_callback(text)
        return text

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _try_encode(self, flush: bool = False) -> None:
        """Run encoder steps on buffered audio, one chunk at a time."""
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
            # In flush mode take whatever is left; otherwise take exactly step
            take = n if flush else step
            chunk = buffered[:take]
            buffered = buffered[take:]
            with ctx:
                self._run_encoder_step(chunk)
            if flush:
                break  # one pass over the remainder is enough

        self._audio_buffer = [buffered] if len(buffered) > 0 else []

    def _run_encoder_step(self, audio: np.ndarray) -> None:
        """Extract features and run one cache-aware encoder step."""
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
        length = torch.tensor([len(audio)], dtype=torch.long)

        # Feature extraction: (1, 80, T_frames)
        features, feat_len = self._preprocessor(
            input_signal=audio_tensor,
            length=length,
        )

        # cache_aware_stream_step returns:
        #   (enc_out, enc_len, cache_channel, cache_time, cache_channel_len)
        # enc_out shape: (1, D, T_enc)  — note D before T, must transpose
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

        # encoder_output shape: (1, D, T_enc) — keep as-is for ctc_decoder
        if encoder_output.size(2) > 0:
            self._encoder_outputs.append(encoder_output)

    def _decode_accumulated(self) -> str:
        """CTC-decode all accumulated encoder outputs."""
        if not self._encoder_outputs:
            return ""

        # Concatenate along time dim: (1, D, T_total)
        enc = torch.cat(self._encoder_outputs, dim=2)

        # ctc_decoder expects (1, D, T) and returns (1, T, vocab) log-probs
        log_probs = self._model.ctc_decoder(encoder_output=enc)

        text = _ctc_greedy_decode(log_probs[0], self._vocabulary, self._blank_id)
        return text
