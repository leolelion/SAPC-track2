#!/usr/bin/env python3
"""Single-stream benchmark for the Nemotron SAPC2 submission wrapper.

This intentionally stays outside `track2_starting_kit/local_decode.py` so it
cannot perturb organizer scoring semantics. It imports a submission directory,
feeds one fixed audio file in 100 ms chunks, and reports timing distributions
plus a coarse frontend/encoder/decoder breakdown.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load audio as mono float32, preferring torchaudio when available."""
    try:
        import torch
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0).to(torch.float32).numpy(), int(sr)
    except Exception:
        pass

    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float32), int(sr)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {path}; install torchaudio or soundfile"
        ) from exc


def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio.astype(np.float32, copy=False)

    try:
        import torch
        import torchaudio.functional as F

        wav = torch.from_numpy(audio.astype(np.float32, copy=False))
        out = F.resample(wav, sr, 16000)
        return out.numpy().astype(np.float32, copy=False)
    except Exception:
        pass

    try:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(sr, 16000)
        return resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Could not resample from {sr} Hz to 16000 Hz") from exc


class SessionRunTimer:
    def __init__(self, session: Any):
        self.session = session
        self.calls = 0
        self.seconds = 0.0

    def reset(self) -> None:
        self.calls = 0
        self.seconds = 0.0

    def run(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = self.session.run(*args, **kwargs)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self.session, name)


def _load_model(submission_dir: Path):
    sys.path.insert(0, str(submission_dir))
    spec = importlib.util.spec_from_file_location(
        "bench_submission_model", submission_dir / "model.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model.py from {submission_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model()


def _instrument_model(model: Any) -> dict[str, Any]:
    model._enc_sess = SessionRunTimer(model._enc_sess)
    model._dec_sess = SessionRunTimer(model._dec_sess)

    original_ensure = model._ensure_features
    feature_stats = {
        "calls": 0,
        "seconds": 0.0,
        "samples_processed": 0,
        "frames_after": 0,
    }

    def ensure_wrapper() -> Any:
        before = model._cached_n_samples
        t0 = time.perf_counter()
        out = original_ensure()
        dt = time.perf_counter() - t0
        if model._cached_n_samples != before:
            feature_stats["calls"] += 1
            feature_stats["seconds"] += dt
            feature_stats["samples_processed"] += model._total_samples
            feature_stats["frames_after"] = model._cached_feat_len
        return out

    model._ensure_features = ensure_wrapper
    return feature_stats


def _run_once(model: Any, audio: np.ndarray, realtime: bool) -> dict[str, Any]:
    feature_stats = model._bench_feature_stats
    model.reset()
    model._enc_sess.reset()
    model._dec_sess.reset()
    feature_stats.update(calls=0, seconds=0.0, samples_processed=0, frames_after=0)

    partials: list[tuple[float, str]] = []
    t0 = time.perf_counter()
    model.set_partial_callback(lambda text: partials.append((time.perf_counter() - t0, text)))

    for chunk_idx, start in enumerate(range(0, len(audio), 1600)):
        if realtime:
            target = chunk_idx * 0.1
            elapsed = time.perf_counter() - t0
            if target > elapsed:
                time.sleep(target - elapsed)
        model.accept_chunk(audio[start : start + 1600])

    final_text = model.input_finished()
    wall = time.perf_counter() - t0
    audio_sec = len(audio) / 16000.0
    return {
        "wall_sec": wall,
        "audio_sec": audio_sec,
        "rtf_wall": wall / audio_sec,
        "compute_time_sec": float(getattr(model, "compute_time_sec", float("nan"))),
        "rtf_compute": float(getattr(model, "compute_time_sec", float("nan"))) / audio_sec,
        "first_partial_sec": partials[0][0] if partials else None,
        "last_partial_sec": partials[-1][0] if partials else None,
        "partials": len(partials),
        "final_len": len(final_text),
        "encoder_calls": model._enc_sess.calls,
        "encoder_sec": model._enc_sess.seconds,
        "decoder_calls": model._dec_sess.calls,
        "decoder_sec": model._dec_sess.seconds,
        "feature_calls": feature_stats["calls"],
        "feature_sec": feature_stats["seconds"],
        "feature_samples_processed": feature_stats["samples_processed"],
        "feature_over_actual": (
            feature_stats["samples_processed"] / len(audio) if len(audio) else None
        ),
        "feature_frames_after": feature_stats["frames_after"],
        "final_preview": final_text[:160],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--jsonl", type=Path, default=None)
    args = parser.parse_args()

    if args.threads is not None:
        os.environ["SAPC2_THREADS"] = str(args.threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

    model = _load_model(args.submission_dir)
    model._bench_feature_stats = _instrument_model(model)

    audio, sr = _load_audio(args.audio)
    audio = _resample_to_16k(audio, sr)
    audio = audio.astype(np.float32, copy=False)

    for _ in range(args.warmup_runs):
        _run_once(model, audio, realtime=False)

    rows = []
    out_f = args.jsonl.open("w", encoding="utf-8") if args.jsonl else None
    try:
        for idx in range(args.runs):
            row = {"run": idx, **_run_once(model, audio, args.realtime)}
            rows.append(row)
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            if out_f:
                out_f.write(line + "\n")
                out_f.flush()
    finally:
        if out_f:
            out_f.close()

    summary = {"runs": len(rows), "realtime": args.realtime}
    for key in (
        "wall_sec",
        "rtf_wall",
        "compute_time_sec",
        "rtf_compute",
        "first_partial_sec",
        "encoder_sec",
        "decoder_sec",
        "feature_sec",
    ):
        values = [r[key] for r in rows if r.get(key) is not None]
        summary[key] = {
            "p50": _percentile(values, 50),
            "p90": _percentile(values, 90),
            "p99": _percentile(values, 99),
            "mean": sum(values) / len(values) if values else None,
        }

    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
