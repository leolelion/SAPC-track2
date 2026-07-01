#!/usr/bin/env python3
"""ONNX Runtime op profile for one Nemotron SAPC2 manifest row."""

from __future__ import annotations

import argparse
import collections
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


def _read_wave(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        samples = f.readframes(f.getnframes())
    return np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0


def _load_model(submission_dir: Path):
    sys.path.insert(0, str(submission_dir))
    spec = importlib.util.spec_from_file_location(
        "profile_submission_model", submission_dir / "model.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model.py from {submission_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.Model()


def _select_row(manifest_csv: Path, bucket: str, index: int) -> dict[str, str]:
    rows: list[dict[str, str]] = []
    with manifest_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dur = float(row.get("duration", "0") or 0)
            row_bucket = "short" if dur < 5.0 else "medium" if dur < 15.0 else "long"
            if row_bucket == bucket:
                rows.append(dict(row, duration_bucket=row_bucket))
    if not rows:
        raise RuntimeError(f"No rows found for bucket {bucket}")
    return rows[index]


def _make_profile_session(ort: Any, path: Path, threads: int, prefix: Path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_profiling = True
    so.profile_file_prefix = str(prefix)
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _summarize_profile(path: str) -> dict[str, Any]:
    events = json.load(open(path, "r", encoding="utf-8"))
    by_op: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0])
    by_provider: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0])
    nodes: list[tuple[float, str, str, str]] = []
    for event in events:
        if event.get("cat") != "Node":
            continue
        dur = float(event.get("dur", 0.0))
        args = event.get("args", {})
        op = args.get("op_name", "?")
        provider = args.get("provider", "?")
        name = event.get("name", "")
        by_op[op][0] += dur
        by_op[op][1] += 1
        by_provider[provider][0] += dur
        by_provider[provider][1] += 1
        nodes.append((dur, op, provider, name))
    return {
        "path": path,
        "node_total_ms": sum(v[0] for v in by_op.values()) / 1000.0,
        "node_count": len(nodes),
        "providers": [
            {"provider": k, "ms": v[0] / 1000.0, "count": v[1]}
            for k, v in sorted(by_provider.items(), key=lambda item: -item[1][0])
        ],
        "top_ops": [
            {"op": k, "ms": v[0] / 1000.0, "count": v[1], "avg_us": v[0] / v[1]}
            for k, v in sorted(by_op.items(), key=lambda item: -item[1][0])[:20]
        ],
        "top_nodes": [
            {"ms": d / 1000.0, "op": op, "provider": provider, "name": name[:160]}
            for d, op, provider, name in sorted(nodes, reverse=True)[:20]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--bucket", choices=["short", "medium", "long"], default="long")
    parser.add_argument("--bucket-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    os.environ["SAPC2_THREADS"] = str(args.threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

    row = _select_row(args.manifest_csv, args.bucket, args.bucket_index)
    audio = _read_wave(args.data_root / row["audio_filepath"])
    module, model = _load_model(args.submission_dir)
    ort = module.ort
    weights = args.submission_dir / "weights"
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    enc_prefix = args.out_json.parent / f"{args.out_json.stem}_encoder"
    dec_prefix = args.out_json.parent / f"{args.out_json.stem}_decoder"
    model._enc_sess = _make_profile_session(
        ort, weights / "encoder_model.onnx", args.threads, enc_prefix
    )
    model._dec_sess = _make_profile_session(
        ort, weights / "decoder_model.onnx", args.threads, dec_prefix
    )
    model._enc_out_names = [o.name for o in model._enc_sess.get_outputs()]
    model._dec_out_names = [o.name for o in model._dec_sess.get_outputs()]

    t0 = time.perf_counter()
    model.reset()
    for start in range(0, len(audio), CHUNK_SIZE):
        model.accept_chunk(audio[start : start + CHUNK_SIZE])
    final = model.input_finished()
    wall = time.perf_counter() - t0
    enc_path = model._enc_sess.end_profiling()
    dec_path = model._dec_sess.end_profiling()

    result = {
        "row": row,
        "audio_sec": len(audio) / SAMPLE_RATE,
        "wall_sec": wall,
        "rtf_wall": wall / (len(audio) / SAMPLE_RATE),
        "compute_time_sec": float(getattr(model, "compute_time_sec", float("nan"))),
        "rtf_compute": float(getattr(model, "compute_time_sec", float("nan")))
        / (len(audio) / SAMPLE_RATE),
        "final_len": len(final),
        "final_preview": final[:160],
        "encoder_profile": _summarize_profile(enc_path),
        "decoder_profile": _summarize_profile(dec_path),
    }
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
