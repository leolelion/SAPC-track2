#!/usr/bin/env python3
# Build a stratified diagnostic manifest from Dev.csv for the full-picture study:
#   (1) ALL utts (capped) of the known worst speakers -> characterize the failure fully
#   (2) length-stratified random sample across ALL speakers -> test the duration hypothesis
#   (3) good-speaker controls come for free from (2)
# Usage (pod): python3 build_diag_manifest.py /workspace/SAPC2/manifest/Dev.csv /workspace/SAPC2/manifest/Dev_diag.csv
import csv, sys, random
random.seed(42)
src, out = sys.argv[1], sys.argv[2]
# worst speakers (8-char prefixes) found in Dev_rand300 (empties concentrated here)
WORST = ["55c1784a", "4a9f71ab", "7c58626d", "8926bff8", "54618732", "4902e925", "e42558d6"]
CAP_PER_WORST = 25                      # cap so a prolific speaker doesn't dominate
BINS = [(0,3),(3,8),(8,15),(15,30),(30,1e9)]
K_PER_BIN = 50                          # random utts per duration bin (any speaker)

rows = list(csv.DictReader(open(src)))
hdr = rows[0].keys()
def dur(r):
    try: return float(r["duration"])
    except: return 0.0

picked = {}                             # id -> row (dedup)
# (1) worst speakers
wcount = {w:0 for w in WORST}
random.shuffle(rows)
for r in rows:
    for w in WORST:
        if r["speaker"].startswith(w) and wcount[w] < CAP_PER_WORST:
            picked[r["id"]] = r; wcount[w]+=1
# (2) length strata
for lo,hi in BINS:
    pool = [r for r in rows if lo<=dur(r)<hi and r["id"] not in picked]
    random.shuffle(pool)
    for r in pool[:K_PER_BIN]:
        picked[r["id"]] = r

sel = list(picked.values())
with open(out,"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(hdr)); w.writeheader()
    for r in sel: w.writerow(r)

# composition report
from collections import Counter
def binname(d):
    for lo,hi in BINS:
        if lo<=d<hi: return f"{lo}-{hi if hi<1e9 else 'inf'}s"
print(f"wrote {len(sel)} utts to {out}")
print("worst-speaker utts:", {w:wcount[w] for w in WORST})
print("by duration bin:", dict(Counter(binname(dur(r)) for r in sel)))
print("by etiology:", dict(Counter(r["etiology"] for r in sel)))
print("dur: min=%.1f median=%.1f max=%.1f" % (
    min(dur(r) for r in sel),
    sorted(dur(r) for r in sel)[len(sel)//2],
    max(dur(r) for r in sel)))
