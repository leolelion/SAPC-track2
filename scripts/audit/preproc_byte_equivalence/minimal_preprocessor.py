#!/usr/bin/env python3
"""Minimal mel preprocessor — pure torchaudio replacement for NeMo's
AudioToMelSpectrogramPreprocessor as configured in model.py.

Hard-coded to the exact Nemotron checkpoint args:
  sample_rate=16000, window_size=0.025, window_stride=0.010, window="hann",
  features=128, n_fft=512, dither=0.0, pad_to=0, normalize="NA",
  frame_splicing=1, log=True, pad_value=0.0

Replicates NeMo's FilterbankFeatures forward path (eval mode):
  - preemph=0.97 applied to waveform, masked beyond seq_len
  - torch.stft(center=True, pad_mode="constant", window=hann(periodic=False))
  - power = sqrt(re^2 + im^2) ** 2  (mag_power=2.0)
  - mel filterbank from torchaudio.functional.melscale_fbanks(
      mel_scale="slaney", norm="slaney")   (matches librosa htk=False, norm="slaney")
  - log(mel + 2**-24)                       (log_zero_guard_value)
  - normalize="NA" -> pass-through
  - frame_splicing=1 -> no-op
  - pad_to=0 -> no padding
  - pad_value=0 for frames beyond seq_len_out
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _slaney_hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Slaney's mel scale, htk=False. Mirrors librosa.hz_to_mel."""
    hz = np.asarray(hz, dtype=np.float64)
    # Linear part below 1000 Hz, log-spaced above.
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (hz - f_min) / f_sp

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    log_region = hz >= min_log_hz
    mels = np.where(
        log_region,
        min_log_mel + np.log(hz / min_log_hz) / logstep,
        mels,
    )
    return mels


def _slaney_mel_to_hz(mels: np.ndarray) -> np.ndarray:
    """Inverse of _slaney_hz_to_mel."""
    mels = np.asarray(mels, dtype=np.float64)
    f_min = 0.0
    f_sp = 200.0 / 3
    hz = f_min + f_sp * mels

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    log_region = mels >= min_log_mel
    hz = np.where(
        log_region,
        min_log_hz * np.exp(logstep * (mels - min_log_mel)),
        hz,
    )
    return hz


def _librosa_mel_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    """Slaney-normalized triangular mel filterbank in fp64, matching
    librosa.filters.mel(htk=False, norm='slaney'). Returns shape
    (n_mels, n_freqs)."""
    # FFT bin frequencies (fp64 throughout — this is where torchaudio's
    # default fp32 path loses precision in the higher mel bins).
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_freqs, dtype=np.float64)

    # Mel-spaced points covering [fmin, fmax] with n_mels + 2 endpoints
    # (the two extra are the outer triangle edges).
    mel_pts = np.linspace(
        _slaney_hz_to_mel(fmin),
        _slaney_hz_to_mel(fmax),
        n_mels + 2,
        dtype=np.float64,
    )
    hz_pts = _slaney_mel_to_hz(mel_pts)

    # Triangular weights: each filter has left/center/right at consecutive hz_pts.
    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for m in range(n_mels):
        f_left, f_center, f_right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        rising = (fft_freqs - f_left) / (f_center - f_left)
        falling = (f_right - fft_freqs) / (f_right - f_center)
        fb[m] = np.maximum(0.0, np.minimum(rising, falling))

    # Slaney normalization: divide each filter by half-width in Hz so each
    # filter has approximately constant area.
    enorm = 2.0 / (hz_pts[2 : n_mels + 2] - hz_pts[:n_mels])
    fb *= enorm[:, np.newaxis]
    return fb


class MinimalMelPreprocessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length = 400
        self.hop_length = 160
        self.n_mels = 128
        self.preemph = 0.97
        self.log_zero_guard_value = 2.0 ** -24
        self.pad_value = 0.0

        window = torch.hann_window(self.win_length, periodic=False)
        self.register_buffer("window", window)

        # Compute filterbank in numpy fp64 to match librosa exactly,
        # then cast to fp32 for matmul against power-spec.
        # Shape: (n_mels, n_freqs).
        fb_np = _librosa_mel_filterbank(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=0.0,
            fmax=self.sample_rate / 2.0,
        )
        fb_t = torch.from_numpy(fb_np.astype(np.float32))
        self.register_buffer("fb", fb_t)

    def get_seq_len(self, seq_len: torch.Tensor) -> torch.Tensor:
        # NeMo: pad_amount = n_fft//2 * 2 = n_fft when center=True.
        # seq_len_out = floor((seq_len + n_fft - n_fft) / hop) = floor(seq_len / hop)
        return torch.floor_divide(seq_len, self.hop_length).to(torch.long)

    @torch.no_grad()
    def forward(
        self, input_signal: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = input_signal
        seq_len_time = length

        seq_len_out_raw = self.get_seq_len(length)
        seq_len_out = torch.where(
            length == 0, torch.zeros_like(seq_len_out_raw), seq_len_out_raw
        )

        # Preemphasis (matches NeMo).
        timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
        x = torch.cat((x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
        x = x.masked_fill(~timemask, 0.0)

        # STFT (center=True, pad_mode="constant", window periodic=False).
        stft_out = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            window=self.window.to(dtype=x.dtype, device=x.device),
            return_complex=True,
            pad_mode="constant",
        )  # (B, n_freqs, T_frames), complex

        # NeMo computes sqrt(re^2 + im^2) then .pow(2.0); mirror that path
        # (skipping the shortcut to keep the float32 numerics identical).
        mag = torch.sqrt(stft_out.real.pow(2) + stft_out.imag.pow(2))
        power = mag.pow(2.0)  # (B, n_freqs, T_frames)

        # Mel filterbank. fb is (n_mels, n_freqs), same layout as NeMo's.
        # matmul((n_mels, n_freqs), (B, n_freqs, T)) -> (B, n_mels, T).
        mel = torch.matmul(self.fb.to(power.dtype), power)  # (B, n_mels, T_frames)

        # Log with NeMo's zero-guard.
        mel = torch.log(mel + self.log_zero_guard_value)

        # Mask beyond seq_len_out with pad_value=0.
        max_len = mel.size(-1)
        mask = torch.arange(max_len, device=mel.device).repeat(mel.size(0), 1) >= seq_len_out.unsqueeze(1)
        mel = mel.masked_fill(mask.unsqueeze(1), self.pad_value)

        return mel, seq_len_out
