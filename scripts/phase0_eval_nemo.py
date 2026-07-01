#!/usr/bin/env python3
# Phase-0 companion: eval a full .nemo on the SAME deterministic 3k subset as phase0_eval_curve.py.
# Used to compare the shipped 5-ckpt AVERAGE (ft_smoke_encoder_only.nemo) vs ep3-single on the same
# proxy -> tests the averaging-contamination hypothesis (avg diluted by 2x epoch-0 ckpts).
#   python3 phase0_eval_nemo.py --nemo .../ft_smoke_encoder_only.nemo \
#     --train-csv /workspace/SAPC2/manifest/Train.csv --data-root /workspace/SAPC2
import argparse, csv, os, random, collections, sys, ast, json
ap = argparse.ArgumentParser()
ap.add_argument("--nemo", required=True)
ap.add_argument("--train-csv", required=True)
ap.add_argument("--data-root", required=True)
ap.add_argument("--ctx", default="[70,1]")
ap.add_argument("--subset-utts", type=int, default=3000)
ap.add_argument("--dev-speaker-frac", type=float, default=0.08)
ap.add_argument("--split-seed", type=int, default=13)
ap.add_argument("--max-duration", type=float, default=40.0)
ap.add_argument("--ref1-col", default="norm_text_with_disfluency")
ap.add_argument("--ref2-col", default="norm_text_without_disfluency")
ap.add_argument("--batch-size", type=int, default=32)
a = ap.parse_args()

# ---- identical subset construction to phase0_eval_curve.py (deterministic) ----
rows = list(csv.DictReader(open(a.train_csv)))
for r in rows:
    r["_abs"] = os.path.join(a.data_root, r["audio_filepath"])
    try: r["_dur"] = float(r["duration"])
    except Exception: r["_dur"] = 0.0
random.seed(a.split_seed)
speakers = sorted({r["speaker"] for r in rows}); random.shuffle(speakers)
n_dev = max(1, int(len(speakers) * a.dev_speaker_frac)); dev_spk = set(speakers[:n_dev])
dev = [r for r in rows if r["speaker"] in dev_spk and r["_dur"] > 0 and os.path.exists(r["_abs"])]
def ref_of(r, col): return (r.get(col) or "").strip()
dev = [r for r in dev if ref_of(r, a.ref1_col) or ref_of(r, a.ref2_col)]
dev = [r for r in dev if r["_dur"] <= a.max_duration]
by_spk = collections.defaultdict(list)
for r in sorted(dev, key=lambda x: x["id"]): by_spk[r["speaker"]].append(r)
subset, idx, order = [], 0, sorted(by_spk.keys())
while len(subset) < a.subset_utts:
    prog = False
    for spk in order:
        if idx < len(by_spk[spk]):
            subset.append(by_spk[spk][idx]); prog = True
            if len(subset) >= a.subset_utts: break
    if not prog: break
    idx += 1
print(f"[subset] n={len(subset)} speakers={len({r['speaker'] for r in subset})}", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalizer.text_normalizer_hf import EnglishTextNormalizer
NORM = EnglishTextNormalizer()
def norm(s): return NORM.norm(s or "").strip()
def ed(a_, b_):
    if len(a_) < len(b_): a_, b_ = b_, a_
    prev = list(range(len(b_)+1))
    for i, ca in enumerate(a_, 1):
        cur=[i]
        for j, cb in enumerate(b_,1): cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]
def min2(h, r1, r2, unit):
    seq=(lambda s:list(s)) if unit=="char" else (lambda s:s.split())
    H,R1,R2=seq(h),seq(r1),seq(r2); e1,n1=ed(H,R1),len(R1); e2,n2=ed(H,R2),len(R2)
    if n1>0: e1=min(e1,n1)
    if n2>0: e2=min(e2,n2)
    c1=e1/n1 if n1 else 9e9; c2=e2/n2 if n2 else 9e9
    if c1<c2: return e1,n1
    if c2<c1: return e2,n2
    return 0.5*(e1+e2), 0.5*(n1+n2)

import torch, nemo.collections.asr as nemo_asr
CTX=ast.literal_eval(a.ctx)
print(f"[model] restoring {a.nemo}", flush=True)
m=nemo_asr.models.EncDecRNNTBPEModel.restore_from(a.nemo, map_location="cpu")
if torch.cuda.is_available(): m=m.cuda()
m.eval()
try: m.encoder.set_default_att_context_size(CTX); print(f"[ctx] {m.encoder.att_context_size}", flush=True)
except Exception as e: print("[ctx] failed", e, flush=True)
paths=[r["_abs"] for r in subset]
ref1=[norm(ref_of(r,a.ref1_col) or ref_of(r,a.ref2_col)) for r in subset]
ref2=[norm(ref_of(r,a.ref2_col) or ref_of(r,a.ref1_col)) for r in subset]
etio=[r.get("etiology","?") for r in subset]
with torch.no_grad():
    hyps=m.transcribe(paths, batch_size=a.batch_size)
hyps=[norm(h.text if hasattr(h,"text") else h) for h in hyps]
ce=ct=we=wt=0.0; ee=collections.defaultdict(float); et=collections.defaultdict(float); n_empty=0
for h,r1,r2,e in zip(hyps,ref1,ref2,etio):
    if not h.strip(): n_empty+=1
    x,y=min2(h,r1,r2,"char"); ce+=x; ct+=y; ee[e]+=x; et[e]+=y
    x,y=min2(h,r1,r2,"word"); we+=x; wt+=y
print(json.dumps({"nemo": os.path.basename(a.nemo), "n": len(subset), "n_empty": n_empty,
    "cer": round(100*ce/ct,4), "wer": round(100*we/wt,4),
    "cer_by_etiology": {k: round(100*ee[k]/et[k],2) for k in sorted(ee) if et[k]}}, indent=2), flush=True)
print("PHASE0_NEMO_DONE", flush=True)
