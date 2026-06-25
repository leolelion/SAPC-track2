#!/usr/bin/env python3
# Evaluate a finetuned .nemo on the severe-enriched Dev_diag set (HELD-OUT: Dev speakers are disjoint from
# Train). Per-etiology two-ref CER + empties via NeMo transcribe at the deploy att_context. This is the
# Gate-2 decision signal (vs zero-shot Nemotron 47.5% CER / 25% empty on the same 425 utts, research/15).
#   python nemo_eval_diag.py <ft.nemo> <Dev_diag.csv> <data_root> "[70,1]"
import sys, csv, os, ast
sys.path.insert(0, "/workspace/sapc-nemotron/utils")
from normalizer.text_normalizer_hf import EnglishTextNormalizer
import torch, nemo.collections.asr as nemo_asr
nm = EnglishTextNormalizer()

nemo_path, man_csv, data_root, att = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
ATT = ast.literal_eval(att)
m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(nemo_path, map_location="cuda")
try: m.encoder.set_default_att_context_size(ATT)
except Exception as e: print("att set failed:", e)
m.eval()

rows = list(csv.DictReader(open(man_csv)))
wavs = [os.path.join(data_root, r["audio_filepath"]) for r in rows]
print(f"transcribing {len(wavs)} utts at att={ATT} ...")
with torch.no_grad():
    hyps = m.transcribe(wavs, batch_size=16)
hyps = [(h.text if hasattr(h, "text") else h) for h in hyps]

def cer(h, r):
    h, r = list(h), list(r)
    if not r: return 0.0 if not h else 1.0
    dp = list(range(len(r)+1))
    for i in range(1, len(h)+1):
        p = dp[0]; dp[0] = i
        for j in range(1, len(r)+1):
            c = dp[j]; dp[j] = min(dp[j]+1, dp[j-1]+1, p+(h[i-1]!=r[j-1])); p = c
    return min(1.0, dp[len(r)]/len(r))

from collections import defaultdict
agg = defaultdict(lambda: {"n": 0, "cer": 0.0, "empty": 0}); tot = {"n": 0, "cer": 0.0, "empty": 0}
for r, h in zip(rows, hyps):
    hn = nm.norm(h or "")
    c = min(cer(hn, nm.norm(r.get("norm_text_with_disfluency", ""))),
            cer(hn, nm.norm(r.get("norm_text_without_disfluency", ""))))
    e = 1 if not (h or "").strip() else 0
    for b in (agg[r["etiology"]], tot):
        b["n"] += 1; b["cer"] += c; b["empty"] += e
def line(name, b): print(f"  {name:20s} n={b['n']:4d}  CER={100*b['cer']/max(1,b['n']):5.1f}%  empty={b['empty']:3d} ({100*b['empty']/max(1,b['n']):.0f}%)")
print(f"\n===== {os.path.basename(nemo_path)} on {os.path.basename(man_csv)} =====")
line("ALL", tot)
for eti in sorted(agg): line(eti, agg[eti])
print("EVAL_DONE")
