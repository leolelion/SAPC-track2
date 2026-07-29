#!/usr/bin/env python3
"""Localise a localmel-vs-NeMo mel mismatch. POD-ONLY.

The parity `feat` stage reports one number; this says WHERE it comes from — which mel
bins, which frames, and what the two sides actually hold there. A diff of ~16.63 is
|log(2**-24)|, i.e. one side hит the log-zero guard while the other did not, so the
interesting question is which bins are empty on which side.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_module(path: Path, name: str):
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--nemo-dir", type=Path, required=True)
    ap.add_argument("--wav", type=Path, required=True)
    args = ap.parse_args()

    import soundfile as sf
    import torch

    nm = load_module(args.nemo_dir / "model.py", "p_nemo").Model()
    om = load_module(args.onnx_dir / "model.py", "p_onnx").Model()

    audio, sr = sf.read(str(args.wav), dtype="float32")
    ours = om._raw_to_feats(np.ascontiguousarray(audio, dtype=np.float32))[0]
    with torch.inference_mode():
        ref, _ = nm._raw_preprocessor(
            input_signal=torch.from_numpy(audio).unsqueeze(0), length=torch.tensor([len(audio)])
        )
    ref = ref[0].numpy()

    print(f"shapes ours={ours.shape} nemo={ref.shape}")
    n = min(ours.shape[-1], ref.shape[-1])
    d = np.abs(ours[:, :n] - ref[:, :n])
    print(f"max={d.max():.6g} mean={d.mean():.6g} frac>1e-3={(d > 1e-3).mean():.4f}")

    bin_max = d.max(axis=1)
    worst_bins = np.argsort(-bin_max)[:8]
    print("worst mel bins (idx, maxdiff):", [(int(b), round(float(bin_max[b]), 4)) for b in worst_bins])
    print("bins with any diff>1:", int((bin_max > 1).sum()), "/", d.shape[0])

    i, j = np.unravel_index(np.argmax(d), d.shape)
    print(f"argmax at mel_bin={i} frame={j}: ours={ours[i, j]:.6f} nemo={ref[i, j]:.6f}")
    guard = float(om._meta["preprocessor"]["log_zero_guard_value"])
    lg = np.log(guard)
    print(f"log(guard)={lg:.6f}")
    print(f"ours==log(guard) count={int(np.isclose(ours[:, :n], lg, atol=1e-3).sum())}")
    print(f"nemo==log(guard) count={int(np.isclose(ref[:, :n], lg, atol=1e-3).sum())}")

    # Are the two just offset by a constant (a scaling/normalisation difference)?
    diff = (ours[:, :n] - ref[:, :n]).ravel()
    print(f"signed diff: mean={diff.mean():.6g} std={diff.std():.6g} "
          f"p1={np.percentile(diff,1):.4f} p99={np.percentile(diff,99):.4f}")

    # Per-frame: is the damage concentrated at the edges (padding) or everywhere?
    fr = d.max(axis=0)
    print("per-frame maxdiff: first5", np.round(fr[:5], 4), "last5", np.round(fr[-5:], 4),
          "middle max", round(float(fr[5:-5].max()), 6) if n > 10 else "n/a")

    # What does NeMo's own featurizer say its config is?
    f = nm._raw_preprocessor.featurizer
    for attr in ("log_zero_guard_value", "log_zero_guard_type", "preemph", "n_fft", "hop_length",
                 "win_length", "mag_power", "normalize", "dither", "pad_to", "exact_pad", "use_grads",
                 "nfilt", "lowfreq", "highfreq", "log"):
        print(f"  featurizer.{attr} = {getattr(f, attr, '<absent>')}")
    fb = getattr(f, "filter_banks", None)
    if fb is not None:
        fb = fb.detach().cpu().numpy().squeeze()
        ours_fb = om._preproc.fb.numpy()
        print(f"filterbank: nemo{fb.shape} ours{ours_fb.shape} "
              f"maxdiff={np.abs(fb - ours_fb).max():.6g} nemo_rowsum0={(fb.sum(1) == 0).sum()}")


if __name__ == "__main__":
    main()
