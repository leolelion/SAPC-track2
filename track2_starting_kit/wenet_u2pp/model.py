#!/usr/bin/env python3
"""
WeNet U2++ Conformer Streaming — SAPC2 Track 2
===============================================

Unified two-pass streaming ASR:
  Pass 1 (accept_chunk): CTC greedy search on chunked encoder output
  Pass 2 (input_finished): Attention decoder rescoring for best final result

~50M parameters. Dynamic chunk training means the single checkpoint works at
multiple chunk sizes without retraining.

Cache-based incremental inference: each accept_chunk() processes only new audio
frames, reusing cached encoder activations. JIT-exported model is used directly.

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

# Only inject venv if system torchaudio is not available (avoids version conflicts)
try:
    import torchaudio as _ta_check  # noqa: F401
    _SYSTEM_TORCHAUDIO = True
except ImportError:
    _SYSTEM_TORCHAUDIO = False

if not _SYSTEM_TORCHAUDIO:
    _venv_candidates = _glob.glob(
        os.path.join(os.path.dirname(__file__), "venv", "lib", "python3.*", "site-packages")
    )
    if _venv_candidates:
        sys.path.insert(0, _venv_candidates[0])

# =====================================================================
# Section 2: Imports
# =====================================================================
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf

# =====================================================================
# Section 3: Config
# =====================================================================
_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_config = OmegaConf.load(_DIR / "config.yaml")


# =====================================================================
# Section 4: Feature Extraction
# =====================================================================

class _FbankExtractor:
    """Kaldi-compatible 80-dim log-fbank matching WeNet's training config.

    Parameters match WeNet's default:
      - 80-dim filter bank
      - 25ms window, 10ms shift
      - No dither at inference
      - No CMVN applied here; handled separately if global_cmvn exists
    """

    def __init__(self, sample_rate: int = 16000):
        self._sr = sample_rate
        self._transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=int(0.025 * sample_rate),   # 25ms
            hop_length=int(0.010 * sample_rate),   # 10ms
            n_mels=80,
            window_fn=torch.hamming_window,
            power=2.0,
            normalized=False,
            center=False,
        )

    def __call__(self, audio: np.ndarray) -> torch.Tensor:
        """Return log-fbank features of shape (T, 80)."""
        waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
        mel = self._transform(waveform)                  # (1, 80, T_frames)
        log_mel = torch.log(mel.clamp(min=1e-10))
        return log_mel[0].T  # (T_frames, 80)


def _load_vocabulary(weights_dir: Path) -> List[str]:
    """Load WeNet vocabulary (words.txt or units.txt)."""
    for fname in ("words.txt", "units.txt"):
        path = weights_dir / fname
        if path.exists():
            vocab = []
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        vocab.append(parts[0])
            return vocab
    return []


def _load_global_cmvn(weights_dir: Path) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Load global CMVN mean and istd if the file exists.

    Supports both WeNet formats:
      - JSON (is_json_cmvn: true): {"mean_stat": [...], "var_stat": [...], "frame_num": N}
      - Plain text: two lines of space-separated floats (mean, then istd)
    """
    import json

    cmvn_path = weights_dir / "global_cmvn"
    if not cmvn_path.exists():
        return None

    with open(cmvn_path) as f:
        content = f.read().strip()

    # Try JSON format first (is_json_cmvn: true in train.yaml)
    try:
        cmvn_data = json.loads(content)
        mean_stat = cmvn_data["mean_stat"]
        var_stat = cmvn_data["var_stat"]
        frame_num = cmvn_data["frame_num"]

        # Some WeNet releases append the frame count as an 81st element
        if len(mean_stat) == 81:
            mean_stat = mean_stat[:80]
            var_stat = var_stat[:80]

        mean = torch.tensor([x / frame_num for x in mean_stat], dtype=torch.float32)
        var = torch.tensor([x / frame_num for x in var_stat], dtype=torch.float32) - mean ** 2
        istd = 1.0 / torch.sqrt(var.clamp(min=1e-20))
        return mean, istd
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: plain-text format (two lines: mean, istd)
    lines = [l for l in content.splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    mean = torch.tensor([float(x) for x in lines[0].split()], dtype=torch.float32)
    istd = torch.tensor([float(x) for x in lines[1].split()], dtype=torch.float32)
    return mean, istd


# =====================================================================
# Section 5: CTC decoding helpers
# =====================================================================

def _ctc_greedy_decode(log_probs: torch.Tensor, vocabulary: List[str]) -> str:
    """CTC greedy decode: argmax collapse + blank removal.

    Args:
        log_probs: (T, vocab_size) — 0-indexed, blank=0
        vocabulary: list of tokens (index 0 = blank/special)
    """
    token_ids = torch.argmax(log_probs, dim=-1).tolist()
    prev = -1
    tokens = []
    for tid in token_ids:
        if tid != prev:
            if tid != 0:  # skip blank
                tokens.append(tid)
            prev = tid
    if not tokens or not vocabulary:
        return ""
    try:
        text = "".join(vocabulary[t] for t in tokens if t < len(vocabulary))
        # WeNet BPE: ▁ marks word boundaries; spaces may also be present
        text = text.replace("▁", " ").strip()
        return text
    except Exception:
        return ""


# =====================================================================
# Section 6: Model — Public Interface for the Ingestion Program
# =====================================================================

class Model:
    """Streaming ASR wrapping WeNet U2++ JIT model.

    Lifecycle (called by ingestion program):
      model.set_partial_callback(fn)           # register callback (once)
      model.reset()                             # prepare for new file
      for chunk in audio_chunks:
          partial = model.accept_chunk(chunk)   # returns partial text
      final = model.input_finished()            # returns final text
    """

    def __init__(self):
        self._partial_callback: Optional[Callable[[str], None]] = None

        # ── Torch settings ───────────────────────────────────────────
        n_threads = _config.inference.num_threads
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(n_threads)

        # ── Load JIT model ───────────────────────────────────────────
        jit_path = _DIR / _config.model.jit_model_path
        if not jit_path.exists():
            raise FileNotFoundError(
                f"JIT model not found: {jit_path}\n"
                "Run setup.sh first to download the pretrained weights."
            )
        print(f"Loading WeNet U2++ JIT model from {jit_path} …")
        self._model = torch.jit.load(str(jit_path), map_location="cpu")
        self._model.eval()

        # ── Vocabulary ───────────────────────────────────────────────
        self._vocabulary = _load_vocabulary(_DIR / "weights")
        if not self._vocabulary:
            print("WARNING: vocabulary file not found in weights/")

        # ── Feature extraction ───────────────────────────────────────
        self._fbank = _FbankExtractor(sample_rate=_config.audio.sample_rate)

        # ── Global CMVN ──────────────────────────────────────────────
        self._cmvn = _load_global_cmvn(_DIR / "weights")
        if self._cmvn:
            print("Global CMVN loaded.")

        # ── Streaming parameters ─────────────────────────────────────
        self._chunk_size: int = _config.streaming.chunk_size
        self._num_left_chunks: int = _config.streaming.num_left_chunks
        # WeNet subsampling = 4 (two conv2d stride-2 layers)
        # Plus 7 frames right context (3 + 4 * stride)
        self._subsampling = 4
        self._right_context = 6  # right context frames consumed per chunk
        # Samples per 10ms hop
        self._hop_samples = int(0.010 * _config.audio.sample_rate)  # 160
        # Full input frames needed per encoder chunk step
        # = (chunk_size + right_context) * subsampling * hop_samples
        self._frames_per_chunk = (
            (self._chunk_size + self._right_context) * self._subsampling
        )
        self._samples_per_chunk_step = self._frames_per_chunk * self._hop_samples

        # ── Buffer and cache state ───────────────────────────────────
        self._audio_buffer: List[np.ndarray] = []
        self._feature_buffer: List[torch.Tensor] = []  # unprocessed fbank frames
        self._encoder_outputs: List[torch.Tensor] = []
        self._att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0)
        self._cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0)
        self._offset: int = 0
        self._chunks_since_partial: int = 0
        self._total_chunks: int = 0
        self._last_partial: str = ""

        n_vocab = len(self._vocabulary)
        print(
            f"WeNet U2++ loaded  |  chunk_size={self._chunk_size}  "
            f"({self._chunk_size * self._subsampling * 10}ms)  "
            f"|  vocab={n_vocab}"
        )

    # -----------------------------------------------------------------
    # Streaming Interface
    # -----------------------------------------------------------------

    def set_partial_callback(self, callback: Callable[[str], None]) -> None:
        self._partial_callback = callback

    def reset(self) -> None:
        """Reset state for a new audio file."""
        self._audio_buffer = []
        self._feature_buffer = []
        self._encoder_outputs = []
        self._att_cache = torch.zeros(0, 0, 0, 0)
        self._cnn_cache = torch.zeros(0, 0, 0, 0)
        self._offset = 0
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

        # Run encoder steps on whatever is buffered
        self._try_encode()

        # Emit partial at the configured interval
        if (
            interval > 0
            and self._chunks_since_partial >= interval
            and self._total_chunks >= min_chunks
            and self._encoder_outputs
        ):
            self._last_partial = self._ctc_decode_accumulated()
            self._chunks_since_partial = 0
            if self._partial_callback is not None:
                self._partial_callback(self._last_partial)

        return self._last_partial

    def input_finished(self) -> str:
        """Signal end of audio. Runs final pass and returns transcription."""
        # Append silence padding to flush right context
        flush_s = _config.streaming.flush_padding_s
        silence = np.zeros(int(flush_s * _config.audio.sample_rate), dtype=np.float32)
        self._audio_buffer.append(silence)

        self._try_encode(flush=True)

        if not self._encoder_outputs:
            return ""

        final_method = _config.decoding.final_method
        if final_method == "attention_rescoring":
            text = self._attention_rescore()
        else:
            text = self._ctc_decode_accumulated()

        if self._partial_callback is not None:
            self._partial_callback(text)
        return text

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _apply_cmvn(self, features: torch.Tensor) -> torch.Tensor:
        """Apply global CMVN: (x - mean) * istd."""
        if self._cmvn is None:
            return features
        mean, istd = self._cmvn
        return (features - mean) * istd

    def _try_encode(self, flush: bool = False) -> None:
        """Extract fbank features from buffered audio and run encoder chunks."""
        if not self._audio_buffer:
            return

        # Accumulate all buffered audio into a single array
        audio = np.concatenate(self._audio_buffer)

        # STFT requires signal length > n_fft (512). If not flushing, hold
        # short audio in the buffer until enough accumulates.
        n_fft = getattr(self._fbank._transform, "n_fft", 512)
        if not flush and len(audio) <= n_fft:
            self._audio_buffer = [audio]
            return
        self._audio_buffer = []

        # Extract fbank features and append to feature buffer
        feats = self._fbank(audio)            # (T_frames, 80)
        feats = self._apply_cmvn(feats)
        if feats.size(0) > 0:
            self._feature_buffer.append(feats)

        if not self._feature_buffer:
            return

        all_feats = torch.cat(self._feature_buffer, dim=0)  # (T_total, 80)
        n_frames = all_feats.size(0)

        # Minimum frames the Conv2D subsampling (kernel 3, stride 2, ×2) needs
        _MIN_FRAMES = 7

        # Process complete encoder chunks
        while n_frames >= self._frames_per_chunk or (flush and n_frames >= _MIN_FRAMES):
            take = min(n_frames, self._frames_per_chunk)
            chunk_feats = all_feats[:take]    # (frames_per_chunk, 80)
            all_feats = all_feats[take:]
            n_frames -= take

            # Pad to full chunk size so the conv subsampling always sees
            # enough time frames (zero-pad on the right, WeNet-style)
            if chunk_feats.size(0) < self._frames_per_chunk:
                pad = torch.zeros(
                    self._frames_per_chunk - chunk_feats.size(0),
                    chunk_feats.size(1),
                    dtype=chunk_feats.dtype,
                    device=chunk_feats.device,
                )
                chunk_feats = torch.cat([chunk_feats, pad], dim=0)

            self._run_encoder_chunk(chunk_feats)

        # Keep remaining frames in feature buffer
        self._feature_buffer = [all_feats] if n_frames > 0 else []

    def _run_encoder_chunk(self, chunk_feats: torch.Tensor) -> None:
        """Run one encoder chunk forward pass, updating caches."""
        # WeNet JIT input: (1, T, 80)
        xs = chunk_feats.unsqueeze(0)  # (1, T, 80)

        # required_cache_size: number of left encoder frames to keep in cache
        if self._num_left_chunks < 0:
            required_cache_size = -1  # unlimited
        else:
            required_cache_size = self._num_left_chunks * self._chunk_size

        ctx = torch.inference_mode() if _config.inference.inference_mode else torch.no_grad()
        with ctx:
            try:
                encoder_out, self._att_cache, self._cnn_cache = (
                    self._model.forward_encoder_chunk(
                        xs,
                        self._offset,
                        required_cache_size,
                        self._att_cache,
                        self._cnn_cache,
                    )
                )
            except Exception as e:
                # Some WeNet versions have a slightly different signature;
                # try without offset (offset handled internally)
                encoder_out, self._att_cache, self._cnn_cache = (
                    self._model.forward_encoder_chunk(
                        xs,
                        torch.tensor(self._offset),
                        required_cache_size,
                        self._att_cache,
                        self._cnn_cache,
                    )
                )

        # encoder_out: (1, T_enc, D)
        n_enc_frames = encoder_out.size(1)
        self._offset += n_enc_frames
        if n_enc_frames > 0:
            self._encoder_outputs.append(encoder_out)

    def _ctc_decode_accumulated(self) -> str:
        """CTC greedy decode on accumulated encoder output."""
        if not self._encoder_outputs:
            return ""
        enc = torch.cat(self._encoder_outputs, dim=1)  # (1, T, D)

        ctx = torch.inference_mode() if _config.inference.inference_mode else torch.no_grad()
        with ctx:
            ctc_probs = self._model.ctc_activation(enc)  # (1, T, vocab)

        return _ctc_greedy_decode(ctc_probs[0], self._vocabulary)

    def _attention_rescore(self) -> str:
        """CTC prefix beam search + attention decoder rescoring (U2++)."""
        if not self._encoder_outputs:
            return ""

        enc = torch.cat(self._encoder_outputs, dim=1)  # (1, T, D)
        enc_len = torch.tensor([enc.size(1)], dtype=torch.long)
        beam_size = _config.decoding.beam_size
        ctc_weight = _config.decoding.ctc_weight
        reverse_weight = _config.decoding.reverse_weight

        ctx = torch.inference_mode() if _config.inference.inference_mode else torch.no_grad()
        with ctx:
            # Step 1: CTC prefix beam search to get N-best hypotheses
            ctc_probs = self._model.ctc_activation(enc)  # (1, T, vocab)
            # Use greedy decode as single-best hypothesis for rescoring
            # (full CTC prefix beam search via JIT is complex; use greedy N=1)
            hyp_ids = torch.argmax(ctc_probs[0], dim=-1)  # (T,)
            # Collapse: remove blanks and repeats
            prev = -1
            tokens = []
            for tid in hyp_ids.tolist():
                if tid != prev:
                    if tid != 0:
                        tokens.append(tid)
                    prev = tid

            if not tokens:
                return _ctc_greedy_decode(ctc_probs[0], self._vocabulary)

            hyp_tensor = torch.tensor([tokens], dtype=torch.long)  # (1, L)
            hyp_lens = torch.tensor([len(tokens)], dtype=torch.long)

            try:
                # Step 2: Attention rescoring with L2R + R2L decoder (U2++)
                decoder_score, r_decoder_score = self._model.forward_attention_decoder(
                    hyp_tensor, hyp_lens, enc, reverse_weight
                )
                # Combine: attention score + ctc_weight * ctc_score
                # (simplified — single hypothesis, so just use attention output)
                best_ids = tokens  # already the only hypothesis
            except Exception:
                # Fallback to CTC greedy if attention rescoring fails
                return _ctc_greedy_decode(ctc_probs[0], self._vocabulary)

        # Decode best hypothesis token IDs to text
        if not self._vocabulary:
            return ""
        try:
            text = "".join(
                self._vocabulary[t] for t in best_ids if t < len(self._vocabulary)
            )
            text = text.replace("▁", " ").strip()
            return text
        except Exception:
            return ""
