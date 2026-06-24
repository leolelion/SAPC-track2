#!/usr/bin/env python3
# Direct two-ref char-CER (official EnglishTextNormalizer) + empty/distribution analysis.
# Usage (on pod): python3 nemotron_repro_analyze.py <manifest.csv> <predict.csv>
# Purpose: discriminate "clean unimodal ~25%" (Nemotron just weaker zero-shot) from
# "bimodal w/ empties" (a subpopulation fails -> CER clipped to 1.0 -> inflated mean).
import csv, sys
sys.path.insert(0, "/workspace/sapc-nemotron/utils")
from normalizer.text_normalizer_hf import EnglishTextNormalizer
norm = EnglishTextNormalizer()

man, hyp = sys.argv[1], sys.argv[2]
refs = {}
with open(man) as f:
    for r in csv.DictReader(f):
        refs[r["id"]] = (r.get("norm_text_with_disfluency", ""),
                         r.get("norm_text_without_disfluency", ""))

def cer(h, r):
    h, r = list(h), list(r)
    if not r:
        return 0.0 if not h else 1.0
    dp = list(range(len(r) + 1))
    for i in range(1, len(h) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(r) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (h[i - 1] != r[j - 1]))
            prev = cur
    return min(1.0, dp[len(r)] / len(r))

dist = []; empties = 0; raw_empty = 0
with open(hyp) as f:
    for r in csv.DictReader(f):
        uid = r["id"]
        if uid not in refs:
            continue
        raw = r["raw_hypos"]
        h = norm.norm(raw)
        c = min(cer(h, norm.norm(refs[uid][0])), cer(h, norm.norm(refs[uid][1])))
        if not raw.strip():
            raw_empty += 1
        if not h.strip():
            empties += 1
        dist.append(c)

n = len(dist); dist.sort()
mean = sum(dist) / n * 100
p = lambda q: dist[min(n - 1, int(n * q))] * 100
print(f"N={n}  meanCER={mean:.2f}%  raw_empty_hyps={raw_empty}  normed_empty={empties}")
print(f"  CER dist: p10={p(.10):.1f} p50={p(.50):.1f} p90={p(.90):.1f} p99={p(.99):.1f} max={dist[-1]*100:.1f}")
print(f"  buckets:  <20%: {sum(c<.2 for c in dist)}   20-50%: {sum(.2<=c<.5 for c in dist)}"
      f"   50-80%: {sum(.5<=c<.8 for c in dist)}   >80%(near-fail): {sum(c>=.8 for c in dist)}")
