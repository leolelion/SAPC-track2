#!/usr/bin/env python3
"""NeMo-free log-mel front-end for the parakeet_realtime ONNX submission.

Drop-in replacement for NeMo's AudioToMelSpectrogramPreprocessor, parameterised from
`weights/streaming_meta.json` (written by scripts/export_parakeet_onnx.py) instead of
hard-coded — the Nemotron localmel.py baked [n_fft=512, hop=160, win=400, preemph=0.97,
128 mel, normalize=NA] as constants and would silently produce wrong features for any
model whose preprocessor differs. Here every parameter comes from the exporting model's
own cfg, and anything we do not implement raises at construction (never a silent
fallback — repo house rule).

The mel filterbank and analysis window are exported as .npy from the real NeMo
preprocessor, so the filter shapes are NeMo's, not a re-derivation.

Parity with the NeMo preprocessor (dither=0, pad_to=0, normalize=None) is gated
numerically on-pod by scripts/parity_parakeet_onnx.py stage `feat`.
"""

import os

import numpy as np
import torch

_SUPPORTED_NORMALIZE = (None, "None", "NA", "na", "")


class LocalMel:
    """Config-driven log-mel extractor. __call__ mirrors the NeMo preprocessor signature
    (input_signal=[1, N] float32 tensor, length=[1] int) -> ([1, F, T], [1] lengths)."""

    def __init__(self, weights_dir: str, meta: dict):
        pp = meta["preprocessor"]

        # --- reject anything this implementation does not reproduce, loudly ---
        if pp.get("normalize") not in _SUPPORTED_NORMALIZE:
            raise RuntimeError(
                f"localmel: preprocessor normalize={pp.get('normalize')!r} is not implemented; "
                "this front-end only reproduces un-normalised features (normalize=NA/None). "
                "The exported model needs per-feature normalisation -> implement it before shipping."
            )
        if pp.get("mag_power", 2.0) != 2.0:
            raise RuntimeError(f"localmel: mag_power={pp.get('mag_power')} != 2.0 unsupported")
        if pp.get("window", "hann") != "hann":
            raise RuntimeError(f"localmel: window={pp.get('window')!r} unsupported")
        if not pp.get("center", True):
            raise RuntimeError("localmel: center=False unsupported (NeMo exact_pad path)")
        if pp.get("log", True) is not True:
            raise RuntimeError("localmel: log=False unsupported")

        self.n_fft = int(pp["n_fft"])
        self.hop = int(pp["hop_length"])
        self.win_length = int(pp["win_length"])
        self.preemph = pp.get("preemph", None)
        self.preemph = None if self.preemph in (None, 0, 0.0) else float(self.preemph)
        self.log_zero_guard = float(pp["log_zero_guard_value"])
        self.n_mels = int(pp["features"])

        fb = np.load(os.path.join(weights_dir, "filterbank.npy")).astype("float32")
        win = np.load(os.path.join(weights_dir, "window.npy")).astype("float32")
        if fb.shape != (self.n_mels, self.n_fft // 2 + 1):
            raise RuntimeError(f"localmel: filterbank shape {fb.shape} != {(self.n_mels, self.n_fft // 2 + 1)}")
        if win.shape != (self.win_length,):
            raise RuntimeError(f"localmel: window shape {win.shape} != {(self.win_length,)}")
        self.fb = torch.from_numpy(fb)
        self.win = torch.from_numpy(win)

    def __call__(self, input_signal, length=None):
        x = input_signal[0].float()
        if self.preemph is not None:
            x = torch.cat([x[:1], x[1:] - self.preemph * x[:-1]])
        spec = torch.stft(
            x,
            self.n_fft,
            self.hop,
            self.win_length,
            self.win,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spec.real ** 2 + spec.imag ** 2
        mel = self.fb @ power
        lm = torch.log(mel + self.log_zero_guard)
        return lm.unsqueeze(0), torch.tensor([lm.shape[1]])
