#!/usr/bin/env python3
"""Byte-equivalence test: MinimalMelPreprocessor vs NeMo
AudioToMelSpectrogramPreprocessor with the exact Nemotron args.

Pass criterion: max abs diff over all (mel, frame) cells < 1e-4.
Also reports rel-diff and per-frame stats so we can debug if the gross
match is off.
"""
import sys
import wave
import math

import numpy as np
import torch

sys.path.insert(0, "/work")
from minimal_preprocessor import MinimalMelPreprocessor  # noqa: E402

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor  # noqa: E402


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as f:
        assert f.getframerate() == 16000, f.getframerate()
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def build_nemo():
    pp = AudioToMelSpectrogramPreprocessor(
        sample_rate=16000,
        window_size=0.025,
        window_stride=0.010,
        window="hann",
        features=128,
        n_fft=512,
        dither=0.0,
        pad_to=0,
        normalize="NA",
        frame_splicing=1,
        log=True,
        pad_value=0.0,
    )
    pp.eval()
    return pp


def compare(audio_path: str) -> dict:
    audio = load_wav(audio_path)
    x = torch.from_numpy(audio).unsqueeze(0)
    seq_len = torch.tensor([len(audio)], dtype=torch.long)

    pp_nemo = build_nemo()
    pp_ours = MinimalMelPreprocessor()
    pp_ours.eval()

    with torch.inference_mode():
        mel_n, len_n = pp_nemo(input_signal=x, length=seq_len)
        mel_o, len_o = pp_ours(input_signal=x, length=seq_len)

    info = {
        "audio_samples": int(len(audio)),
        "audio_seconds": round(len(audio) / 16000, 3),
        "mel_n_shape": tuple(mel_n.shape),
        "mel_o_shape": tuple(mel_o.shape),
        "len_n": int(len_n.item()),
        "len_o": int(len_o.item()),
    }

    if mel_n.shape != mel_o.shape:
        info["shape_match"] = False
        return info
    info["shape_match"] = True

    diff = (mel_n - mel_o).abs()
    info["max_abs_diff"] = float(diff.max().item())
    info["mean_abs_diff"] = float(diff.mean().item())
    info["max_rel_diff"] = float(
        (diff / (mel_n.abs() + 1e-12)).max().item()
    )

    # Frame-by-frame max abs diff so we can see if it's localized
    per_frame = diff.max(dim=1).values.squeeze(0)  # (T_frames,)
    info["per_frame_max_first10"] = [round(v, 6) for v in per_frame[:10].tolist()]
    info["per_frame_max_last10"] = [round(v, 6) for v in per_frame[-10:].tolist()]
    info["per_frame_max_argmax"] = int(per_frame.argmax().item())
    info["per_frame_max_overall"] = round(per_frame.max().item(), 6)

    # Per-mel-bin max
    per_mel = diff.max(dim=2).values.squeeze(0)  # (n_mels,)
    info["per_mel_max_first10"] = [round(v, 6) for v in per_mel[:10].tolist()]
    info["per_mel_max_last10"] = [round(v, 6) for v in per_mel[-10:].tolist()]
    info["per_mel_max_argmax"] = int(per_mel.argmax().item())

    return info


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print("usage: test_equiv.py /path/to/audio.wav [...]")
        sys.exit(2)
    import json
    for p in paths:
        print(f"\n=== {p} ===")
        info = compare(p)
        print(json.dumps(info, indent=2))
        ok = info.get("shape_match") and info.get("max_abs_diff", 1.0) < 1e-4
        print("PASS" if ok else "FAIL", "(threshold 1e-4)")
