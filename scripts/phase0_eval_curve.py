#!/usr/bin/env python3
# Phase-0 (v2 workup): re-test v1 under-convergence on a RELIABLE val — fast but accurate.
#
# WHY: v1's in-training selection used limit_val_batches=30 + shuffle=False => the SAME 480 of
# 27,135 internal-dev utts, few speakers. The independent reviewer flagged the under-convergence
# claim as resting on that noisy curve. This re-evaluates all v1 checkpoints (ep0..ep3) on a
# SPEAKER-STRATIFIED subset that covers ALL internal-dev speakers, scored with the OFFICIAL
# normalizer + min-over-two-refs + clip@1 (the real scorer's CER definition).
#
# This is an OFFLINE-transcribe PROXY (not 100 ms streaming) — valid for the curve SHAPE
# (descending vs flat), which is the only decision here. Any re-ship/submit still goes through
# the faithful local_decode.py + evaluate.sh streaming harness (house rule, learned 2026-06-24).
#
#   source /workspace/nemoenv/bin/activate
#   python3 phase0_eval_curve.py \
#     --train-csv /workspace/SAPC2/manifest/Train.csv --data-root /workspace/SAPC2 \
#     --base-nemo /workspace/finetune/nemo_ft/nemotron-speech-streaming-en-0.6b.nemo \
#     --ckpt-dir  /workspace/finetune/nemo_ft/full_enc \
#     --out-json  /workspace/finetune/nemo_ft/phase0_curve.json
import argparse, csv, json, os, random, glob, collections, sys

ap = argparse.ArgumentParser()
ap.add_argument("--train-csv", required=True)
ap.add_argument("--data-root", required=True)
ap.add_argument("--base-nemo", required=True)
ap.add_argument("--ckpt-dir", required=True)
ap.add_argument("--ckpt-glob", default="ft-*.ckpt")
ap.add_argument("--ctx", default="[70,1]")
ap.add_argument("--subset-utts", type=int, default=3000)
ap.add_argument("--dev-speaker-frac", type=float, default=0.08)  # MUST match prep_nemo_manifest.py
ap.add_argument("--split-seed", type=int, default=13)            # MUST match prep_nemo_manifest.py
ap.add_argument("--max-duration", type=float, default=40.0)      # match train-time filter
ap.add_argument("--ref1-col", default="norm_text_with_disfluency")
ap.add_argument("--ref2-col", default="norm_text_without_disfluency")
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--out-json", required=True)
a = ap.parse_args()

# ---------- reproduce the EXACT v1 speaker-disjoint dev split (prep_nemo_manifest.py) ----------
rows = list(csv.DictReader(open(a.train_csv)))
for r in rows:
    r["_abs"] = os.path.join(a.data_root, r["audio_filepath"])
    try: r["_dur"] = float(r["duration"])
    except Exception: r["_dur"] = 0.0
random.seed(a.split_seed)
speakers = sorted({r["speaker"] for r in rows})
random.shuffle(speakers)
n_dev = max(1, int(len(speakers) * a.dev_speaker_frac))
dev_spk = set(speakers[:n_dev])
dev = [r for r in rows if r["speaker"] in dev_spk and r["_dur"] > 0 and os.path.exists(r["_abs"])]
print(f"[split] total_speakers={len(speakers)} dev_speakers={len(dev_spk)} dev_utts={len(dev)}", flush=True)

# keep only rows with a non-empty ref; respect the train-time max_duration so the subset matches
def ref_of(r, col): return (r.get(col) or "").strip()
dev = [r for r in dev if ref_of(r, a.ref1_col) or ref_of(r, a.ref2_col)]
dev_in_dur = [r for r in dev if r["_dur"] <= a.max_duration]
print(f"[split] dev_with_ref={len(dev)} within_{a.max_duration}s={len(dev_in_dur)}", flush=True)

# ---------- speaker-stratified subset: round-robin across ALL dev speakers (deterministic) ----------
by_spk = collections.defaultdict(list)
for r in sorted(dev_in_dur, key=lambda x: x["id"]):
    by_spk[r["speaker"]].append(r)
subset, idx = [], 0
order = sorted(by_spk.keys())
while len(subset) < a.subset_utts:
    progressed = False
    for spk in order:
        if idx < len(by_spk[spk]):
            subset.append(by_spk[spk][idx]); progressed = True
            if len(subset) >= a.subset_utts: break
    if not progressed: break
    idx += 1
print(f"[subset] n={len(subset)} speakers_covered={len({r['speaker'] for r in subset})}/{len(dev_spk)}",
      flush=True)

# ---------- metric: official normalizer + min-over-two-refs CER/WER, clip@1 ----------
# normalizer/ is shipped as a package next to this script (it uses relative imports).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalizer.text_normalizer_hf import EnglishTextNormalizer
NORM = EnglishTextNormalizer()
def norm(s): return NORM.norm(s or "").strip()

def edit_distance(a_, b_):
    if len(a_) < len(b_): a_, b_ = b_, a_
    prev = list(range(len(b_) + 1))
    for i, ca in enumerate(a_, 1):
        cur = [i]
        for j, cb in enumerate(b_, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def min2(h, r1, r2, unit):  # unit: 'char' or 'word' -> returns (errors, total)
    seq = (lambda s: list(s)) if unit == "char" else (lambda s: s.split())
    H, R1, R2 = seq(h), seq(r1), seq(r2)
    e1, n1 = edit_distance(H, R1), len(R1)
    e2, n2 = edit_distance(H, R2), len(R2)
    if n1 > 0: e1 = min(e1, n1)
    if n2 > 0: e2 = min(e2, n2)
    c1 = e1 / n1 if n1 else float("inf")
    c2 = e2 / n2 if n2 else float("inf")
    if c1 < c2:   return e1, n1
    if c2 < c1:   return e2, n2
    return 0.5 * (e1 + e2), 0.5 * (n1 + n2)

# ---------- model: restore base once, swap checkpoint state_dict per epoch ----------
import ast, torch
import nemo.collections.asr as nemo_asr
CTX = ast.literal_eval(a.ctx)
print(f"[model] restoring base {a.base_nemo}", flush=True)
m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(a.base_nemo, map_location="cpu")
if torch.cuda.is_available(): m = m.cuda()
m.eval()
try:
    m.encoder.set_default_att_context_size(CTX)
    print(f"[ctx] att_context = {m.encoder.att_context_size}", flush=True)
except Exception as e:
    print("[ctx] set failed:", e, flush=True)

paths = [r["_abs"] for r in subset]
ref1 = [norm(ref_of(r, a.ref1_col) or ref_of(r, a.ref2_col)) for r in subset]
ref2 = [norm(ref_of(r, a.ref2_col) or ref_of(r, a.ref1_col)) for r in subset]
etio = [r.get("etiology", "?") for r in subset]

ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, a.ckpt_glob)))
print(f"[ckpts] {len(ckpts)} found:", [os.path.basename(c) for c in ckpts], flush=True)

def epoch_of(path):
    import re
    mt = re.search(r"epoch=(\d+)", os.path.basename(path))
    return int(mt.group(1)) if mt else -1

results = []
for c in ckpts:
    sd = torch.load(c, map_location="cpu", weights_only=False).get("state_dict", {})
    miss = m.load_state_dict(sd, strict=False)
    n_miss = len(getattr(miss, "missing_keys", []))
    n_unexp = len(getattr(miss, "unexpected_keys", []))
    m.eval()
    try: m.encoder.set_default_att_context_size(CTX)
    except Exception: pass
    with torch.no_grad():
        hyps = m.transcribe(paths, batch_size=a.batch_size)
    hyps = [norm(h.text if hasattr(h, "text") else h) for h in hyps]

    ce = ct = we = wt = 0.0
    et_err = collections.defaultdict(float); et_tot = collections.defaultdict(float)
    n_empty = 0
    for h, r1, r2, et in zip(hyps, ref1, ref2, etio):
        if not h.strip(): n_empty += 1
        e, t = min2(h, r1, r2, "char"); ce += e; ct += t
        et_err[et] += e; et_tot[et] += t
        e, t = min2(h, r1, r2, "word"); we += e; wt += t
    row = {
        "ckpt": os.path.basename(c), "epoch": epoch_of(c),
        "missing_keys": n_miss, "unexpected_keys": n_unexp,
        "n": len(subset), "n_empty": n_empty,
        "cer": round(100 * ce / ct, 4) if ct else None,
        "wer": round(100 * we / wt, 4) if wt else None,
        "cer_by_etiology": {k: round(100 * et_err[k] / et_tot[k], 2) for k in sorted(et_err) if et_tot[k]},
    }
    results.append(row)
    print(f"[eval] {row['ckpt']} ep{row['epoch']} CER={row['cer']}% WER={row['wer']}% "
          f"empty={n_empty}/{len(subset)} miss={n_miss} unexp={n_unexp}", flush=True)

results.sort(key=lambda x: (x["epoch"], x["ckpt"]))
out = {"subset_utts": len(subset), "subset_speakers": len({r["speaker"] for r in subset}),
       "dev_speakers": len(dev_spk), "ctx": a.ctx, "normalizer": "EnglishTextNormalizer",
       "metric": "min-over-two-refs, clip@1", "results": results}
json.dump(out, open(a.out_json, "w"), indent=2)
print("WROTE", a.out_json, flush=True)
print("PHASE0_CURVE_DONE", flush=True)
