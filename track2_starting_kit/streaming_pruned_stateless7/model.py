#!/usr/bin/env python3
"""
Streaming Pruned-Transducer-Stateless7 ASR Model — SAPC2 Track 2
=================================================================

Uses TorchScript JIT models (no k2 / lhotse / full icefall stack needed).
Works on macOS arm64 and Linux x86_64 with only: torch, kaldifeat, sentencepiece.

Two variants (config.yaml -> model.variant):
  libri_giga:   marcoyang/icefall-libri-giga-pruned-transducer-stateless7-streaming-2023-04-04
                weights: exp/cpu_jit.pt  (single scripted model, epoch-20-avg-4)
  librispeech:  Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
                weights: exp/encoder_jit_trace.pt + decoder_jit_trace.pt + joiner_jit_trace.pt
                (separate streaming traces, epoch-30-avg-9)

Streaming loop (from icefall's jit_trace_pretrained.py):
  T = chunk_len + pad_len  (pad_len = 7; encoder subsampling ((x-7)//2+1)//2)
  while online_fbank.num_frames_ready - num_processed >= T:
      frames = stack T frames
      encoder_out, _, states = encoder(x=frames, x_lens=[T], states=states)
      num_processed += chunk_len
      greedy/beam search step

Required interface (local_decode.py):
  __init__()                        — Load JIT model + BPE tokenizer
  set_partial_callback(fn) -> None  — Register partial result callback
  reset()             -> None       — Reset state for new file
  accept_chunk(buf)   -> str        — Feed audio chunk (float32, 16 kHz)
  input_finished()    -> str        — Signal end of audio, return final text
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import sentencepiece as spm
from kaldifeat import FbankOptions, OnlineFbank
from omegaconf import OmegaConf

# =====================================================================
# Paths and config
# =====================================================================
_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
config = OmegaConf.load(_DIR / "config.yaml")
_VARIANT = config.model.variant       # "libri_giga" or "librispeech"
_WEIGHTS = _DIR / "weights" / _VARIANT

_SAMPLE_RATE = config.audio.sample_rate
_CHUNK_LEN = config.encoder.decode_chunk_len   # feature frames per decode step
_PAD_LEN = 7                                   # encoder subsampling tail
_T = _CHUNK_LEN + _PAD_LEN                    # total frames fed per step
_CONTEXT_SIZE = config.decoder.context_size    # transducer decoder context
_NUM_ACTIVE_PATHS = config.decoding.num_active_paths
_DECODING_METHOD = config.decoding.method


def _make_fbank_opts(device="cpu") -> FbankOptions:
    opts = FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = _SAMPLE_RATE
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    return opts


# =====================================================================
# Greedy search (inline, no k2 dependency)
# =====================================================================
def _greedy_step(
    decoder,
    joiner,
    encoder_out: torch.Tensor,       # (T_enc, C)
    hyp: List[int],
    decoder_out: Optional[torch.Tensor],  # (1, C) or None
    blank_id: int,
    context_size: int,
) -> Tuple[List[int], torch.Tensor]:
    """Run one greedy search pass over encoder_out frames."""
    if decoder_out is None:
        dec_in = torch.tensor([hyp[-context_size:]], dtype=torch.int32)
        decoder_out = decoder(dec_in, torch.tensor([False])).squeeze(1)

    for i in range(encoder_out.size(0)):
        cur = encoder_out[i : i + 1]
        joiner_out = joiner(cur, decoder_out).squeeze(0)
        y = joiner_out.argmax(dim=0).item()
        if y != blank_id:
            hyp.append(y)
            dec_in = torch.tensor([hyp[-context_size:]], dtype=torch.int32)
            decoder_out = decoder(dec_in, torch.tensor([False])).squeeze(1)

    return hyp, decoder_out


# =====================================================================
# Modified beam search (inline, no k2 dependency)
# =====================================================================
def _beam_step(
    decoder,
    joiner,
    encoder_out: torch.Tensor,       # (T_enc, C)
    hyps: List[List[int]],
    decoder_outs: Optional[List[torch.Tensor]],
    blank_id: int,
    context_size: int,
    num_active_paths: int,
) -> Tuple[List[List[int]], List[torch.Tensor]]:
    """Modified beam search — pure Python, no k2."""
    import math

    if decoder_outs is None:
        decoder_outs = []
        for h in hyps:
            dec_in = torch.tensor([h[-context_size:]], dtype=torch.int32)
            decoder_outs.append(decoder(dec_in, torch.tensor([False])).squeeze(1))

    for i in range(encoder_out.size(0)):
        cur = encoder_out[i : i + 1]  # (1, C)

        # Score each hypothesis
        new_hyps: List[Tuple[float, List[int], torch.Tensor]] = []
        for h, d_out in zip(hyps, decoder_outs):
            joiner_out = joiner(cur, d_out).squeeze(0)  # (vocab,)
            log_probs = torch.log_softmax(joiner_out, dim=0)

            # blank stays in hypothesis unchanged
            new_hyps.append((log_probs[blank_id].item(), h, d_out))

            # top non-blank tokens
            topk = log_probs.topk(min(num_active_paths, log_probs.size(0))).indices
            for tok in topk.tolist():
                if tok == blank_id:
                    continue
                new_h = h + [tok]
                dec_in = torch.tensor([new_h[-context_size:]], dtype=torch.int32)
                new_d = decoder(dec_in, torch.tensor([False])).squeeze(1)
                new_hyps.append((log_probs[tok].item(), new_h, new_d))

        # keep top-N by score
        new_hyps.sort(key=lambda x: -x[0])
        new_hyps = new_hyps[:num_active_paths]
        hyps = [x[1] for x in new_hyps]
        decoder_outs = [x[2] for x in new_hyps]

    return hyps, decoder_outs


# =====================================================================
# Model
# =====================================================================
class Model:
    """Streaming pruned-transducer-stateless7 via JIT models.

    No k2 / lhotse / icefall imports — pure torch + kaldifeat + spm.
    """

    def __init__(self):
        print(f"Loading streaming pruned-transducer-stateless7 (variant={_VARIANT}, JIT mode) …")
        self._device = torch.device("cpu")
        self._partial_callback = None

        # BPE tokenizer
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(_WEIGHTS / "data" / "lang_bpe_500" / "bpe.model"))
        self._blank_id = self._sp.piece_to_id("<blk>")

        # Load JIT model(s)
        if _VARIANT == "librispeech":
            # Separate streaming trace files (Zengwei repo)
            self._encoder = torch.jit.load(str(_WEIGHTS / "exp" / "encoder_jit_trace.pt"))
            self._decoder = torch.jit.load(str(_WEIGHTS / "exp" / "decoder_jit_trace.pt"))
            self._joiner = torch.jit.load(str(_WEIGHTS / "exp" / "joiner_jit_trace.pt"))
            self._scripted = False  # trace-style encoder: called as encoder(x, lens, states)
        else:
            # cpu_jit.pt scripted model (marcoyang libri_giga repo)
            # encoder.forward() is patched to streaming_forward at export time,
            # so calling encoder(x, x_lens, states) does streaming inference.
            _m = torch.jit.load(str(_WEIGHTS / "exp" / "cpu_jit.pt"))
            self._encoder = _m.encoder
            self._decoder = _m.decoder
            self._joiner = _m.joiner
            self._scripted = True

        self._encoder.eval()
        self._decoder.eval()
        self._joiner.eval()

        # Fbank options (reused across utterances)
        self._fbank_opts = _make_fbank_opts()

        print(f"Model loaded (variant={_VARIANT})")

    # -----------------------------------------------------------------
    # Streaming Interface
    # -----------------------------------------------------------------

    def set_partial_callback(self, callback) -> None:
        self._partial_callback = callback

    def reset(self) -> None:
        """Reset state for a new audio file."""
        self._online_fbank = OnlineFbank(self._fbank_opts)
        # get_init_state: trace version takes (device,), scripted takes no args
        if self._scripted:
            self._states = [self._encoder.get_init_state()]
        else:
            self._states = self._encoder.get_init_state(self._device)
        self._num_processed = 0
        self._hyp = [self._blank_id] * _CONTEXT_SIZE
        self._decoder_out = None
        # beam search state (only used for modified_beam_search)
        self._beam_hyps = [[self._blank_id] * _CONTEXT_SIZE for _ in range(_NUM_ACTIVE_PATHS)]
        self._beam_decoder_outs = None

    def accept_chunk(self, audio_chunk: np.ndarray) -> str:
        """Feed one audio chunk (float32, 16 kHz) and return partial text."""
        self._online_fbank.accept_waveform(
            sampling_rate=_SAMPLE_RATE,
            waveform=torch.from_numpy(audio_chunk),
        )
        self._drain()
        return self._current_text()

    def input_finished(self) -> str:
        """Signal end of audio, flush, return final text."""
        tail = torch.zeros(int(_SAMPLE_RATE * 0.3), dtype=torch.float32)
        self._online_fbank.accept_waveform(sampling_rate=_SAMPLE_RATE, waveform=tail)
        self._online_fbank.input_finished()
        self._drain()
        return self._current_text()

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    def _current_text(self) -> str:
        if _DECODING_METHOD == "modified_beam_search":
            return self._sp.decode(self._beam_hyps[0][_CONTEXT_SIZE:])
        return self._sp.decode(self._hyp[_CONTEXT_SIZE:])

    def _drain(self):
        """Decode as many complete chunks as available from the online fbank."""
        while self._online_fbank.num_frames_ready - self._num_processed >= _T:
            # Collect T frames
            frames = [
                self._online_fbank.get_frame(self._num_processed + i)
                for i in range(_T)
            ]
            frames = torch.cat(frames, dim=0).unsqueeze(0)  # (1, T, 80)
            x_lens = torch.tensor([_T], dtype=torch.int32)

            # Encoder streaming forward
            if self._scripted:
                # scripted model: encoder's forward IS streaming_forward
                encoder_out, _, new_states = self._encoder(
                    x=frames, x_lens=x_lens, states=self._states[0]
                )
                self._states = [new_states]
            else:
                # trace model: encoder is the traced streaming_forward function
                encoder_out, _, new_states = self._encoder(
                    x=frames, x_lens=x_lens, states=self._states
                )
                self._states = new_states

            self._num_processed += _CHUNK_LEN

            encoder_out = encoder_out.squeeze(0)  # (T_enc, C)

            # Decode
            if _DECODING_METHOD == "modified_beam_search":
                self._beam_hyps, self._beam_decoder_outs = _beam_step(
                    self._decoder,
                    self._joiner,
                    encoder_out,
                    self._beam_hyps,
                    self._beam_decoder_outs,
                    self._blank_id,
                    _CONTEXT_SIZE,
                    _NUM_ACTIVE_PATHS,
                )
                text = self._sp.decode(self._beam_hyps[0][_CONTEXT_SIZE:])
            else:
                self._hyp, self._decoder_out = _greedy_step(
                    self._decoder,
                    self._joiner,
                    encoder_out,
                    self._hyp,
                    self._decoder_out,
                    self._blank_id,
                    _CONTEXT_SIZE,
                )
                text = self._sp.decode(self._hyp[_CONTEXT_SIZE:])

            if self._partial_callback and text:
                self._partial_callback(text)
