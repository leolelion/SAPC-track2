#!/usr/bin/env python3
"""
NeMo FastConformer Hybrid Large Streaming Multi — SAPC2 Track 2
================================================================

Wraps nvidia/stt_en_fastconformer_hybrid_large_streaming_multi (114M params)
with the 5-method streaming interface required by the ingestion program.

KEY DIFFERENCE from nemo_fastconformer/ (32M):
  - Model: 114M params (3.5× larger) — higher accuracy but higher CPU cost
  - Multi-latency: one checkpoint supports 80ms, 480ms, and 1040ms latency modes
    without retraining. Set config.model.latency_mode to switch.
  - Pareto tool: submit the same fine-tuned model at different latency_mode values
    to occupy multiple points on the accuracy-vs-latency Pareto frontier.

LATENCY MODES (configured in config.yaml via latency_mode):
  "80ms"   — 8 encoder frames per step, 80ms look-ahead
  "480ms"  — 48 encoder frames per step, 480ms look-ahead  (DEFAULT)
  "1040ms" — 104 encoder frames per step, 1040ms look-ahead

STREAMING: Cache-aware encoder, identical mechanism to nemo_fastconformer/.
  CTC decoder for fast partial results.
  RNNT decoder for final result (more accurate).

Required interface (called by ingestion program):
  __init__()                       — Load model weights (once)
  set_partial_callback(fn) -> None — Register partial result callback
  reset()             -> None      — Reset state per audio file
  accept_chunk(buf)   -> str       — Feed 100ms audio chunk
  input_finished()    -> str       — Signal end of audio, return text

To change settings, edit config.yaml (not this file).
To switch latency mode, change config.model.latency_mode.
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

# Map human-readable latency mode → encoder chunk size (frames per step, non-first chunk)
# These values correspond to the multi-latency training configurations.
_LATENCY_TO_CHUNK_FRAMES = {
    "80ms": 8,      # 8 frames × 10ms = 80ms
    "480ms": 48,    # 48 frames × 10ms = 480ms
    "1040ms": 104,  # 104 frames × 10ms = 1040ms
}


# =====================================================================
# Section 4: Helpers
# =====================================================================

def _ctc_greedy_decode(
    log_probs: torch.Tensor, vocabulary: List[str], blank_id: int
) -> str:
    """CTC greedy decode: argmax collapse + blank removal."""
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
    """Streaming ASR wrapping NeMo FastConformer Hybrid Large Multi.

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
        latency_mode = _config.model.latency_mode
        print(f"Loading NeMo FastConformer Multi: {model_name} (latency={latency_mode}) …")

        self._model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name=model_name,
            map_location="cpu",
        )
        self._model.eval()
        self._model.to("cpu")

        # ── Multi-latency streaming configuration ────────────────────
        # The multi model supports multiple chunk sizes. We configure the
        # encoder to use the chunk size corresponding to the chosen latency mode.
        target_chunk_frames = _LATENCY_TO_CHUNK_FRAMES.get(latency_mode)
        if target_chunk_frames is None:
            valid = list(_LATENCY_TO_CHUNK_FRAMES.keys())
            raise ValueError(
                f"Unknown latency_mode {latency_mode!r}. Valid options: {valid}"
            )

        # setup_streaming_params() populates streaming_cfg with trained defaults.
        # For the multi model, we then override chunk_size to select the latency mode.
        self._model.encoder.setup_streaming_params()
        cfg = self._model.encoder.streaming_cfg

        # Override chunk_size for the selected latency mode.
        # chunk_size is a 2-element list: [first_chunk_frames, non_first_chunk_frames]
        # (some NeMo versions use [first, subsequent, subsequent] — both are handled)
        original_chunk = cfg.chunk_size
        cfg.chunk_size = [original_chunk[0], target_chunk_frames]
        self._model.encoder.streaming_cfg = cfg

        # Left context override (optional)
        if _config.model.left_context_frames is not None:
            cfg.cache_drop_size = _config.model.left_context_frames

        self._hop_length = 160  # 10ms at 16kHz
        self._enc_chunk_frames = cfg.chunk_size[1]
        self._samples_per_enc_chunk = self._enc_chunk_frames * self._hop_length

        # ── Decoder setup ─────────────────────────────────────────────
        self._decoder_type = _config.model.decoder_type
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
            f"NeMo FastConformer Multi loaded ({params:.1f}M params, "
            f"latency={latency_mode}, "
            f"chunk={self._enc_chunk_frames}fr/{self._samples_per_enc_chunk}smp, "
            f"decoder={self._decoder_type})"
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

        self._try_encode()

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
        flush_s = _config.streaming.flush_padding_s
        flush_samples = int(flush_s * self._sr)
        silence = np.zeros(flush_samples, dtype=np.float32)
        self._audio_buffer.append(silence)
        self._try_encode(flush=True)

        if not self._encoder_outputs:
            return ""

        text = self._decode_accumulated()
        if self._partial_callback is not None:
            self._partial_callback(text)
        return text

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _try_encode(self, flush: bool = False) -> None:
        """Run encoder steps on buffered audio."""
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

    def _decode_accumulated(self) -> str:
        """CTC-decode all accumulated encoder outputs."""
        if not self._encoder_outputs:
            return ""
        enc = torch.cat(self._encoder_outputs, dim=2)  # (1, D, T)
        log_probs = self._model.ctc_decoder(encoder_output=enc)
        return _ctc_greedy_decode(log_probs[0], self._vocabulary, self._blank_id)
