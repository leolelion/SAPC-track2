#!/usr/bin/env python3
"""Faster alternative to local_decode.py — batch decode only, no streaming pass.

Usage:
  python direct_decode.py \
    --manifest /path/to/Dev_100_local.csv \
    --data-root /path/to/dev100_bundle \
    --out-csv /path/to/output.csv

This is for local benchmarking only. The official submission still uses
local_decode.py.
"""
import argparse
import csv
import sys
import time
import wave
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True, type=Path)
parser.add_argument("--data-root", required=True, type=Path)
parser.add_argument("--out-csv", required=True, type=Path)
args = parser.parse_args()

# Load model
sys.path.insert(0, str(Path(__file__).parent))
from model import Model  # noqa: E402

print("Loading model...", flush=True)
t0 = time.perf_counter()
model = Model()
model.set_partial_callback(lambda _: None)
print(f"Model loaded in {time.perf_counter()-t0:.1f}s", flush=True)

# Load manifest
entries = []
with open(args.manifest, newline="") as f:
    for row in csv.DictReader(f):
        entries.append((row["id"], args.data_root / row["audio_filepath"]))
print(f"Decoding {len(entries)} files...", flush=True)

def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

CHUNK = 1600
ids, preds = [], []
t_start = time.perf_counter()
for i, (uid, audio_path) in enumerate(entries, 1):
    samples = read_wav(audio_path)
    model.reset()
    for start in range(0, len(samples), CHUNK):
        model.accept_chunk(samples[start : start + CHUNK])
    result = model.input_finished()
    ids.append(uid)
    preds.append(result)
    if i % 10 == 0:
        elapsed = time.perf_counter() - t_start
        print(f"  [{i}/{len(entries)}] {elapsed:.0f}s elapsed", flush=True)

args.out_csv.parent.mkdir(parents=True, exist_ok=True)
with open(args.out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "raw_hypos"])
    for uid, pred in zip(ids, preds):
        w.writerow([uid, pred])

elapsed = time.perf_counter() - t_start
print(f"\nDone. {len(entries)} files in {elapsed:.0f}s. Saved to {args.out_csv}")
