#!/usr/bin/env python3
"""
Parakeet CTC 1.1B (Fine-tuned on SAP dysarthric data) — SAPC2 Track 2
=======================================================================

Wraps a fine-tuned NeMo EncDecCTCModelBPE (Parakeet-CTC ~1B params) with the
5-method streaming interface required by the Track 2 ingestion program.

MODEL NOTES:
  - Architecture: FastConformer encoder (1.06B params) + ConvASRDecoder (CTC)
  - Trained with att_context_size=[-1,-1] (full attention, non-streaming)
  - Fine-tuned on SAPC2 dysarthric speech data (Track 1 submission)
  - Vocab: 1024 BPE tokens, blank_id=1024

STREAMING APPROACH:
  Because the model was trained with full attention (not cache-aware streaming),
  we use batch-accumulation mode:
    - accept_chunk(): buffer audio, periodically re-encode + CTC-decode for partials
    - input_finished(): encode full buffer, CTC-decode for final result

  This gives correct (full-context) CTC output at the cost of no true streaming.
  Partial results are emitted every partial_interval_chunks chunks by re-running
  the full encoder over accumulated audio.

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
) or _glob.glob(
    os.path.join(os.path.dirname(__file__), "venv", "*", "lib", "python3.*", "site-packages")
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

def _ctc_greedy_decode(log_probs: torch.Tensor, blank_id: int) -> str:
    """CTC greedy decode: argmax collapse + blank removal.

    Args:
        log_probs: (T, vocab_size+1) log probabilities (float32)
        blank_id:  index of the CTC blank token

    Returns decoded text string with BPE sentencepiece detokenization.
    """
    token_ids = torch.argmax(log_probs, dim=-1).tolist()
    prev = -1
    tokens = []
    for tid in token_ids:
        if tid != prev:
            if tid != blank_id:
                tokens.append(tid)
            prev = tid
    return tokens


def _ids_to_text(token_ids: list, vocabulary: List[str]) -> str:
    """Convert BPE token ids to text using sentencepiece-style detokenization."""
    if not token_ids:
        return ""
    pieces = []
    for t in token_ids:
        if t < len(vocabulary):
            pieces.append(vocabulary[t])
    # Sentencepiece uses ▁ as word-start marker
    text = "".join(pieces).replace("▁", " ").strip()
    return text


# =====================================================================
# Section 5: Model — Public Interface for the Ingestion Program
# =====================================================================
class Model:
    """Streaming ASR wrapping fine-tuned NeMo Parakeet CTC.

    Uses batch-accumulation mode since the model was trained with full attention.

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
        local_nemo = _DIR / "weights" / "final.nemo"
        if local_nemo.exists():
            print(f"Loading Parakeet CTC from local file: {local_nemo}")
            self._model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
                restore_path=str(local_nemo),
                map_location="cpu",
            )
        else:
            model_name = _config.model.name
            print(f"Loading Parakeet CTC: {model_name} (from HuggingFace) ...")
            self._model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                model_name=model_name,
                map_location="cpu",
            )
        self._model.eval()
        self._model.to("cpu")

        # ── Vocabulary & blank ────────────────────────────────────────
        # ConvASRDecoder.vocabulary: list of 1024 BPE pieces (no blank entry)
        # blank is at index num_classes_with_blank - 1 = 1024
        self._vocabulary: List[str] = list(self._model.decoder.vocabulary)
        self._blank_id: int = self._model.decoder.num_classes_with_blank - 1

        # ── Feature extraction (NeMo's built-in preprocessor) ────────
        self._preprocessor = self._model.preprocessor
        self._preprocessor.eval()

        # ── Audio buffer (reset per utterance) ───────────────────────
        self._sr: int = _config.audio.sample_rate
        self._audio_buffer: List[np.ndarray] = []
        self._chunks_since_partial: int = 0
        self._total_chunks: int = 0
        self._last_partial: str = ""

        params = sum(p.numel() for p in self._model.parameters()) / 1e6
        print(f"Parakeet CTC loaded ({params:.1f}M params, vocab={len(self._vocabulary)}, blank={self._blank_id})")

    # -----------------------------------------------------------------
    # Streaming Interface
    # -----------------------------------------------------------------

    def set_partial_callback(self, callback: Callable[[str], None]) -> None:
        self._partial_callback = callback

    def reset(self) -> None:
        """Reset state for a new audio file."""
        self._audio_buffer = []
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

        # Emit partial at configured interval
        if (
            interval > 0
            and self._chunks_since_partial >= interval
            and self._total_chunks >= min_chunks
        ):
            self._last_partial = self._decode_buffer()
            self._chunks_since_partial = 0
            if self._last_partial and self._partial_callback is not None:
                self._partial_callback(self._last_partial)

        return self._last_partial

    def input_finished(self) -> str:
        """Signal end of audio. Returns final transcription."""
        if not self._audio_buffer:
            return ""
        text = self._decode_buffer()
        if self._partial_callback is not None:
            self._partial_callback(text)
        return text

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _decode_buffer(self) -> str:
        """Encode accumulated audio and CTC-decode for a transcription."""
        if not self._audio_buffer:
            return ""

        all_audio = np.concatenate(self._audio_buffer)
        audio_tensor = torch.from_numpy(all_audio).unsqueeze(0)  # (1, T)
        length = torch.tensor([len(all_audio)], dtype=torch.long)

        ctx = torch.inference_mode if _config.inference.inference_mode else torch.no_grad

        with ctx():
            # Feature extraction: (1, 80, T_frames)
            features, feat_len = self._preprocessor(
                input_signal=audio_tensor,
                length=length,
            )
            # Encoder: (1, D, T_enc)
            encoded, enc_len = self._model.encoder(
                audio_signal=features,
                length=feat_len,
            )
            # CTC decoder: (1, T_enc, vocab+1)
            log_probs = self._model.decoder(encoder_output=encoded)

        # Greedy CTC decode on first (and only) sequence
        token_ids = _ctc_greedy_decode(log_probs[0], self._blank_id)
        return _ids_to_text(token_ids, self._vocabulary)
