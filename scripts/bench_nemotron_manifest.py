#!/usr/bin/env python3
"""Benchmark a Nemotron SAPC2 submission on a manifest sample.

This is the multi-file companion to `bench_nemotron_single_stream.py`.
It loads the model once, optionally warms it, then measures selected
manifest rows while keeping the exact SAPC2 100 ms chunk interface.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_RATE = 16000
CHUNK_SIZE = 1600


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _read_wave(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        if f.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {f.getframerate()}")
        if f.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono, got {f.getnchannels()} channels")
        if f.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {f.getsampwidth() * 8}-bit")
        samples = f.readframes(f.getnframes())
    return np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0


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


def _bucket(duration: float) -> str:
    if duration < 5.0:
        return "short"
    if duration < 15.0:
        return "medium"
    return "long"


def _select_rows(manifest_csv: Path, limit_per_bucket: int, max_rows: int | None) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    counts = {"short": 0, "medium": 0, "long": 0}
    with manifest_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur = float(row.get("duration", "0") or 0)
            b = _bucket(dur)
            if limit_per_bucket > 0 and counts[b] >= limit_per_bucket:
                continue
            row = dict(row)
            row["duration_bucket"] = b
            selected.append(row)
            counts[b] += 1
            if max_rows is not None and len(selected) >= max_rows:
                break
            if limit_per_bucket > 0 and all(v >= limit_per_bucket for v in counts.values()):
                break
    return selected


def _run_utterance(model: Any, feature_stats: dict[str, Any], audio: np.ndarray, realtime: bool) -> dict[str, Any]:
    model.reset()
    model._enc_sess.reset()
    model._dec_sess.reset()
    feature_stats.update(calls=0, seconds=0.0, samples_processed=0, frames_after=0)
    partials: list[tuple[float, str]] = []
    t0 = time.perf_counter()
    model.set_partial_callback(lambda text: partials.append((time.perf_counter() - t0, text)))

    for chunk_idx, start in enumerate(range(0, len(audio), CHUNK_SIZE)):
        if realtime:
            target = chunk_idx * 0.1
            elapsed = time.perf_counter() - t0
            if target > elapsed:
                time.sleep(target - elapsed)
        model.accept_chunk(audio[start : start + CHUNK_SIZE])

    final_text = model.input_finished()
    wall = time.perf_counter() - t0
    audio_sec = len(audio) / SAMPLE_RATE
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


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows), "buckets": {}}
    for bucket in ("short", "medium", "long", "all"):
        group = rows if bucket == "all" else [r for r in rows if r["duration_bucket"] == bucket]
        if not group:
            continue
        out["buckets"][bucket] = {"n": len(group)}
        for key in (
            "audio_sec",
            "rtf_wall",
            "rtf_compute",
            "first_partial_sec",
            "encoder_sec",
            "decoder_sec",
            "feature_sec",
            "feature_over_actual",
            "decoder_calls",
        ):
            vals = [r[key] for r in group if r.get(key) is not None]
            out["buckets"][bucket][key] = {
                "p50": _percentile(vals, 50),
                "p90": _percentile(vals, 90),
                "mean": sum(vals) / len(vals) if vals else None,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--limit-per-bucket", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--warmup-utterances", type=int, default=1)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    if args.threads is not None:
        os.environ["SAPC2_THREADS"] = str(args.threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

    rows = _select_rows(args.manifest_csv, args.limit_per_bucket, args.max_rows)
    if not rows:
        raise RuntimeError(f"No rows selected from {args.manifest_csv}")

    model = _load_model(args.submission_dir)
    feature_stats = _instrument_model(model)

    for row in rows[: args.warmup_utterances]:
        audio = _read_wave(args.data_root / row["audio_filepath"])
        _run_utterance(model, feature_stats, audio, realtime=False)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    measured: list[dict[str, Any]] = []
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            audio_path = args.data_root / row["audio_filepath"]
            audio = _read_wave(audio_path)
            result = {
                "index": idx,
                "id": row["id"],
                "speaker": row.get("speaker"),
                "etiology": row.get("etiology"),
                "duration_manifest": float(row["duration"]),
                "duration_bucket": row["duration_bucket"],
                "audio_path": str(audio_path),
                **_run_utterance(model, feature_stats, audio, realtime=args.realtime),
            }
            measured.append(result)
            line = json.dumps(result, sort_keys=True)
            print(line, flush=True)
            f.write(line + "\n")
            f.flush()

    print("SUMMARY " + json.dumps(_summary(measured), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
