#!/usr/bin/env python3
# Full-picture cross-tab: per-utt two-ref char-CER (official normalizer) for BOTH models,
# broken down by duration bin and etiology, plus the head-to-head on the SAME utterances.
# Usage (pod):
#   python3 diag_crosstab.py <manifest.csv> nemo=<nemo_predict.csv> zf=<zipformer_predict.csv>
import csv, sys
sys.path.insert(0, "/workspace/sapc-nemotron/utils")
from normalizer.text_normalizer_hf import EnglishTextNormalizer
nm = EnglishTextNormalizer()

man = sys.argv[1]
preds = {}
for a in sys.argv[2:]:
    k, p = a.split("=", 1); preds[k] = {r["id"]: r["raw_hypos"] for r in csv.DictReader(open(p))}
M = {r["id"]: r for r in csv.DictReader(open(man))}

def cer(h, r):
    h, r = list(h), list(r)
    if not r: return 0.0 if not h else 1.0
    dp = list(range(len(r)+1))
    for i in range(1, len(h)+1):
        p = dp[0]; dp[0] = i
        for j in range(1, len(r)+1):
            c = dp[j]; dp[j] = min(dp[j]+1, dp[j-1]+1, p+(h[i-1]!=r[j-1])); p = c
    return min(1.0, dp[len(r)]/len(r))

def score(uid, raw):
    m = M[uid]
    h = nm.norm(raw)
    return min(cer(h, nm.norm(m.get("norm_text_with_disfluency",""))),
               cer(h, nm.norm(m.get("norm_text_without_disfluency",""))))

def dbin(d):
    d = float(d)
    for lo,hi,nm_ in [(0,3,"0-3s"),(3,8,"3-8s"),(8,15,"8-15s"),(15,30,"15-30s"),(30,9e9,">30s")]:
        if lo<=d<hi: return nm_

from collections import defaultdict
def agg():
    return {"n":0,"cer":0.0,"empty":0}
for k in preds:
    print(f"\n########## MODEL: {k} ##########")
    by_bin=defaultdict(agg); by_eti=defaultdict(agg); tot=agg()
    for uid in M:
        if uid not in preds[k]: continue
        raw=preds[k][uid]; c=score(uid,raw); e=1 if not raw.strip() else 0
        for bucket in (by_bin[dbin(M[uid]["duration"])], by_eti[M[uid]["etiology"]], tot):
            bucket["n"]+=1; bucket["cer"]+=c; bucket["empty"]+=e
    def line(name,a): print(f"  {name:14s} n={a['n']:4d}  meanCER={100*a['cer']/max(1,a['n']):5.1f}%  empty={a['empty']:3d} ({100*a['empty']/max(1,a['n']):.0f}%)")
    line("ALL",tot)
    print("  -- by duration --")
    for b in ["0-3s","3-8s","8-15s","15-30s",">30s"]:
        if by_bin[b]["n"]: line(b,by_bin[b])
    print("  -- by etiology --")
    for e in sorted(by_eti): line(e,by_eti[e])

# head-to-head on SAME utts (only where both models ran)
if "nemo" in preds and "zf" in preds:
    print("\n########## HEAD-TO-HEAD (same utts): does ZF rescue NEMO failures? ##########")
    both=[u for u in M if u in preds["nemo"] and u in preds["zf"]]
    nemo_fail=[u for u in both if score(u,preds["nemo"][u])>0.8 or not preds["nemo"][u].strip()]
    zf_ok=sum(1 for u in nemo_fail if score(u,preds["zf"][u])<0.5)
    zf_alsofail=sum(1 for u in nemo_fail if score(u,preds["zf"][u])>0.8 or not preds["zf"][u].strip())
    print(f"  Nemo failures (>0.8 or empty): {len(nemo_fail)} / {len(both)}")
    print(f"    -> ZF good (<0.5 CER) on those: {zf_ok}  (DOMAIN: finetuning would fix)")
    print(f"    -> ZF also fails (>0.8/empty):  {zf_alsofail}  (audio genuinely hard)")
    print("  sample (dur | eti | nemoCER | zfCER | zf hyp):")
    for u in sorted(nemo_fail, key=lambda u:-float(M[u]['duration']))[:12]:
        print(f"    {float(M[u]['duration']):5.1f}s {M[u]['etiology'][:14]:14s} "
              f"nemo={100*score(u,preds['nemo'][u]):3.0f} zf={100*score(u,preds['zf'][u]):3.0f}  zf={preds['zf'][u][:42]!r}")
