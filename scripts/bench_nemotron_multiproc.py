#!/usr/bin/env python3
"""Multiprocess batch benchmark for the Nemotron SAPC2 submission.

This approximates the Track 2 accuracy pass topology: N worker processes,
each loading one Model instance, decoding assigned utterances as fast as
possible, with callback disabled. It is meant for offline stress tests such
as "20 Codabench workers x per-process thread count".
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
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


def _load_rows(manifest_csv: Path, data_root: Path, max_rows: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dur = float(row.get("duration", "0") or 0)
            rows.append(
                {
                    "id": row["id"],
                    "audio_path": str(data_root / row["audio_filepath"]),
                    "duration": dur,
                    "speaker": row.get("speaker"),
                    "etiology": row.get("etiology"),
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _load_model(submission_dir: Path):
    sys.path.insert(0, str(submission_dir))
    spec = importlib.util.spec_from_file_location(
        f"bench_submission_model_{os.getpid()}", submission_dir / "model.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model.py from {submission_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model()


def _decode_one(model: Any, item: dict[str, Any]) -> dict[str, Any]:
    audio = _read_wave(Path(item["audio_path"]))
    t0 = time.perf_counter()
    model.reset()
    model.set_partial_callback(lambda _text: None)
    for start in range(0, len(audio), CHUNK_SIZE):
        model.accept_chunk(audio[start : start + CHUNK_SIZE])
    text = model.input_finished() or ""
    wall = time.perf_counter() - t0
    audio_sec = len(audio) / SAMPLE_RATE
    return {
        **item,
        "audio_sec": audio_sec,
        "decode_wall_sec": wall,
        "decode_rtf_wall": wall / audio_sec,
        "compute_time_sec": float(getattr(model, "compute_time_sec", float("nan"))),
        "compute_rtf": float(getattr(model, "compute_time_sec", float("nan"))) / audio_sec,
        "text_len": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text_preview": text[:120],
        "ok": True,
    }


def _worker_main(
    worker_id: int,
    submission_dir: str,
    task_q: mp.Queue,
    result_q: mp.Queue,
    threads: int,
) -> None:
    os.environ["SAPC2_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    load_t0 = time.perf_counter()
    model = _load_model(Path(submission_dir))
    load_sec = time.perf_counter() - load_t0
    result_q.put(
        {
            "event": "worker_loaded",
            "worker_id": worker_id,
            "pid": os.getpid(),
            "load_sec": load_sec,
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        }
    )
    while True:
        try:
            item = task_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if item is None:
            break
        try:
            result = _decode_one(model, item)
            result_q.put({"event": "result", "worker_id": worker_id, "pid": os.getpid(), **result})
        except Exception as exc:
            result_q.put(
                {
                    "event": "result",
                    "worker_id": worker_id,
                    "pid": os.getpid(),
                    "id": item.get("id"),
                    "ok": False,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )


def _summary(results: list[dict[str, Any]], total_wall_sec: float, worker_loads: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if r.get("ok")]
    audio_total = sum(float(r["audio_sec"]) for r in ok)
    empty = [r for r in ok if int(r.get("text_len", 0)) == 0]
    out: dict[str, Any] = {
        "n_results": len(results),
        "n_ok": len(ok),
        "n_failed": len(results) - len(ok),
        "n_empty_text": len(empty),
        "total_audio_sec": audio_total,
        "total_wall_sec": total_wall_sec,
        "aggregate_rtf_wall": total_wall_sec / audio_total if audio_total else None,
        "throughput_audio_sec_per_wall_sec": audio_total / total_wall_sec if total_wall_sec else None,
        "worker_loads": worker_loads,
    }
    for key in ("decode_rtf_wall", "compute_rtf", "decode_wall_sec", "compute_time_sec", "text_len"):
        vals = [float(r[key]) for r in ok if r.get(key) is not None]
        out[key] = {
            "p50": _percentile(vals, 50),
            "p90": _percentile(vals, 90),
            "p95": _percentile(vals, 95),
            "mean": sum(vals) / len(vals) if vals else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--start-method", choices=["spawn", "fork"], default="spawn")
    args = parser.parse_args()

    rows = _load_rows(args.manifest_csv, args.data_root, args.max_rows)
    if not rows:
        raise RuntimeError(f"No rows loaded from {args.manifest_csv}")

    ctx = mp.get_context(args.start_method)
    task_q: mp.Queue = ctx.Queue(maxsize=args.workers * 4)
    result_q: mp.Queue = ctx.Queue()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    workers = [
        ctx.Process(
            target=_worker_main,
            args=(i, str(args.submission_dir), task_q, result_q, args.threads),
        )
        for i in range(args.workers)
    ]

    t0 = time.perf_counter()
    for proc in workers:
        proc.start()
    for row in rows:
        task_q.put(row)
    for _ in workers:
        task_q.put(None)

    results: list[dict[str, Any]] = []
    loads: list[dict[str, Any]] = []
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        while len(results) < len(rows):
            event = result_q.get()
            event["elapsed_sec"] = time.perf_counter() - t0
            line = json.dumps(event, sort_keys=True)
            print(line, flush=True)
            f.write(line + "\n")
            f.flush()
            if event.get("event") == "worker_loaded":
                loads.append(event)
            elif event.get("event") == "result":
                results.append(event)

    for proc in workers:
        proc.join()
    total_wall = time.perf_counter() - t0
    summary = _summary(results, total_wall, loads)
    summary.update(
        {
            "workers": args.workers,
            "threads": args.threads,
            "max_rows": args.max_rows,
            "start_method": args.start_method,
            "affinity_parent": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        }
    )
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
