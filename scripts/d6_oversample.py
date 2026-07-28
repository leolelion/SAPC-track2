#!/usr/bin/env python3
"""D6 — short-command oversampling, derived from the EXISTING train.json.

Why not prep_nemo_manifest_v2.py --oversample-short-words: that script rebuilds the
speaker split from Train.csv, which would change the val set and break apples-to-apples
with the Arm A / rung-1 baselines already measured on /workspace/nemo_ft/val.json.
This pre-pass touches ONLY the train side; val is untouched by construction.

Rationale: RNN-T predictors over-delete rare short phrases (internal-LM bias,
arXiv 2108.10752); 22/48 of the severe empties are <=3 words.

Usage: d6_oversample.py <in_train.json> <out_train.json> [max_words=3] [mult=3]
"""
import json, sys, collections

IN = sys.argv[1]
OUT = sys.argv[2]
MAXW = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MULT = int(sys.argv[4]) if len(sys.argv) > 4 else 3

recs = [json.loads(l) for l in open(IN)]
wc = collections.Counter()
short = 0
with open(OUT, "w") as out:
    for r in recs:
        n = len((r.get("text") or "").split())
        wc[min(n, 20)] += 1
        reps = MULT if 0 < n <= MAXW else 1
        if reps > 1:
            short += 1
        for _ in range(reps):
            out.write(json.dumps(r) + "\n")

total_out = sum(1 for _ in open(OUT))
print(f"[d6] in={len(recs)} short(<= {MAXW} words)={short} ({100.0*short/len(recs):.2f}%) "
      f"mult={MULT} out={total_out} (+{total_out-len(recs)} dup rows)")
print("[d6] word-count histogram (1..10):", {k: wc[k] for k in range(1, 11)})
print("D6_OVERSAMPLE_DONE", OUT)
