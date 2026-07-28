#!/usr/bin/env python3
"""GATE-TRAIN measurement: {val CER, val empty-count, val insertion-rate} for ONE .nemo ckpt.

Same script for Arm A and every Arm B rung, on the SAME val json, at the SAME pinned
att_context -> the comparison is apples-to-apples (the runbook's explicit requirement).

PROXY WARNING: punct-stripped single-ref CER, NOT the organizers' two-ref sclite scorer.
Passing this NEVER authorizes a ship claim (GATE-SHIP only). Nemotron scar.

Usage: val_metrics.py <ckpt.nemo> <val.json> <out.json> [limit] [batch_size]
"""
import json, re, sys, os

CKPT, VAL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 0
BS = int(sys.argv[5]) if len(sys.argv) > 5 else 32
CTX = [70, 1]

_PUNCT = re.compile(r"[^\w\s']")


def norm(s):
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", (s or "").lower())).strip()


def edit_counts(hyp, ref):
    """Levenshtein over word lists -> (sub, ins, del). ins = words the hyp added."""
    h, r = hyp, ref
    n, m = len(h), len(r)
    # dp[i][j] = (cost, sub, ins, del)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [None] * m
        for j in range(1, m + 1):
            # i walks hyp, j walks ref.
            # insertion = hyp word i consumed with no ref word  -> come from prev row, same j
            c_ins = (prev[j][0] + 1, prev[j][1], prev[j][2] + 1, prev[j][3])
            # deletion  = ref word j consumed with no hyp word  -> come from same row, j-1
            c_del = (cur[j - 1][0] + 1, cur[j - 1][1], cur[j - 1][2], cur[j - 1][3] + 1)
            if h[i - 1] == r[j - 1]:
                c_sub = (prev[j - 1][0], prev[j - 1][1], prev[j - 1][2], prev[j - 1][3])
            else:
                c_sub = (prev[j - 1][0] + 1, prev[j - 1][1] + 1, prev[j - 1][2], prev[j - 1][3])
            cur[j] = min(c_del, c_ins, c_sub, key=lambda t: t[0])
        prev = cur
    return prev[m][1], prev[m][2], prev[m][3]


def cer(h, r):
    h, r = list(h), list(r)
    if not r:
        return 0.0 if not h else 1.0
    dp = list(range(len(r) + 1))
    for i in range(1, len(h) + 1):
        p = dp[0]
        dp[0] = i
        for j in range(1, len(r) + 1):
            c = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, p + (h[i - 1] != r[j - 1]))
            p = c
    return min(1.0, dp[len(r)] / len(r))


import torch
import nemo.collections.asr as nemo_asr

recs = [json.loads(l) for l in open(VAL)]
if LIMIT:
    recs = recs[:LIMIT]
print(f"[val] {len(recs)} utts from {VAL}")

m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(CKPT, map_location="cpu")
m = m.cuda().eval() if torch.cuda.is_available() else m.eval()
m.encoder.set_default_att_context_size(CTX)
print(f"[ctx] {m.encoder.att_context_size}")

with torch.no_grad():
    hyps = m.transcribe([r["audio_filepath"] for r in recs], batch_size=BS)
hyps = [(h.text if hasattr(h, "text") else h) for h in hyps]

n_empty = 0
cers, ins_tot, ref_tot, sub_tot, del_tot = [], 0, 0, 0, 0
empties = []
for h, r in zip(hyps, recs):
    nh, nr = norm(h), norm(r["text"])
    if not nh:
        n_empty += 1
        empties.append({"audio_filepath": r["audio_filepath"], "ref": nr})
    cers.append(cer(nh, nr))
    s, i, d = edit_counts(nh.split(), nr.split())
    sub_tot += s
    ins_tot += i
    del_tot += d
    ref_tot += len(nr.split())

res = {
    "ckpt": CKPT, "val": VAL, "n": len(recs), "att_context": CTX,
    "cer_pct": 100.0 * sum(cers) / len(cers),
    "empty_count": n_empty,
    "empty_pct": 100.0 * n_empty / len(recs),
    "insertion_rate_pct": 100.0 * ins_tot / max(1, ref_tot),
    "substitution_rate_pct": 100.0 * sub_tot / max(1, ref_tot),
    "deletion_rate_pct": 100.0 * del_tot / max(1, ref_tot),
    "ref_words": ref_tot,
    "empties_sample": empties[:25],
}
json.dump(res, open(OUT, "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "empties_sample"}, indent=2))
print("VAL_METRICS_DONE", OUT)
