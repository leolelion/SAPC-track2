#!/usr/bin/env python3
"""Build an organizer-shaped manifest CSV for the 2k speaker-disjoint val set.

The val set exists only as NeMo jsonl (/workspace/nemo_ft/val.json). The real harness
(local_decode.py + evaluate.sh) needs a CSV with the organizers' columns, including the two
reference columns the two-ref scorer uses. Recovered by joining val.json's audio paths back
to Train.csv on the wav stem.
"""
import csv, json, os, sys

VAL = "/workspace/nemo_ft/val.json"
# val.json turned out to be drawn from the organizers' Dev split, not Train (verified
# 2026-07-29: paths under processed/Dev/, ids present in Dev.csv, absent from Train.csv).
SRC_CSVS = ["/workspace/SAPC2/manifest/Dev.csv", "/workspace/SAPC2/manifest/Train.csv"]
OUT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/SAPC2/manifest/Val2k.csv"

want = set()
for line in open(VAL):
    r = json.loads(line)
    want.add(os.path.splitext(os.path.basename(r["audio_filepath"]))[0])
print(f"[val] {len(want)} unique stems")

rows, hdr, seen = [], None, set()
for src in SRC_CSVS:
    with open(src, newline="", errors="replace") as f:
        rd = csv.DictReader(f)
        hdr = hdr or rd.fieldnames
        for r in rd:
            if r["id"] in want and r["id"] not in seen:
                seen.add(r["id"])
                rows.append(r)
    print(f"[join] after {src}: {len(rows)}")
print(f"[join] matched {len(rows)}/{len(want)}")
assert len(rows) == len(want), "stem join incomplete — do not score a partial set"

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr)
    w.writeheader()
    w.writerows(rows)
print("VAL2K_CSV_DONE", OUT, len(rows))
