#!/usr/bin/env python3
# v2 manifests: speaker-disjoint, with CAPPED severity oversampling (guarded ride-along, research/37 §6)
# + a fixed representative val3k (CER selection) + a PD forgetting probe + Gate-0 smoke + guardrail subset.
# Severity weights are ETIOLOGY-ONLY (no model-error proxy -> avoids circularity). Cap 2x.
#   python3 prep_nemo_manifest_v2.py --train-csv .../Train.csv --data-root /workspace/SAPC2 \
#       --out-dir .../manifests_v2 --target text
import argparse, csv, json, os, random, collections
random.seed(13)

ap = argparse.ArgumentParser()
ap.add_argument("--train-csv", required=True)
ap.add_argument("--data-root", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--target", default="text")  # v1 trained on cased 'text'; scorer normalizes both sides
ap.add_argument("--dev-speaker-frac", type=float, default=0.08)
ap.add_argument("--val-utts", type=int, default=3000)
ap.add_argument("--smoke-utts", type=int, default=4000)
ap.add_argument("--guardrail-utts", type=int, default=50000)
ap.add_argument("--sev-cap", type=float, default=2.0)
# --- D6 short-command oversampling (default OFF = byte-identical output to before).
# RNN-T predictors over-rely on consecutive-word dependencies -> elevated deletion on rare/short/OOD
# phrases (arXiv 2108.10752). 22 of parakeet's 48 Dev_diag empties are <=3 words (wake-words/commands:
# alexa, cortana, hey siri, dog, football). Multiplies the etiology weight, then the SAME sev-cap
# applies, so the total duplication factor can never exceed --sev-cap. ---
ap.add_argument("--oversample-short-words", type=int, default=0,
                help="D6: utts with <= this many words get --oversample-short-mult. 0 = OFF (default)")
ap.add_argument("--oversample-short-mult", type=float, default=1.0,
                help="D6: weight multiplier for short utts. 1.0 = no-op")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

# etiology -> oversample weight (capped at sev-cap). PD=1.0 (already strong, must not regress).
WEIGHTS = {"ALS": 2.0, "Down Syndrome": 2.0, "Cerebral Palsy": 1.5, "Stroke": 1.5, "Parkinson's Disease": 1.0}

rows = list(csv.DictReader(open(a.train_csv)))
for r in rows:
    r["_abs"] = os.path.join(a.data_root, r["audio_filepath"])
    try: r["_dur"] = float(r["duration"])
    except Exception: r["_dur"] = 0.0

speakers = sorted({r["speaker"] for r in rows}); random.shuffle(speakers)
n_dev = max(1, int(len(speakers) * a.dev_speaker_frac))
dev_spk = set(speakers[:n_dev]); train_spk = set(speakers[n_dev:])
def ok(r): return r["_dur"] > 0 and os.path.exists(r["_abs"]) and (r.get(a.target) or "").strip()
train = [r for r in rows if r["speaker"] in train_spk and ok(r)]
dev   = [r for r in rows if r["speaker"] in dev_spk   and ok(r)]

def rec(r): return {"audio_filepath": r["_abs"], "duration": r["_dur"], "text": (r.get(a.target) or "").strip()}
def write(path, recs):
    with open(path, "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    return len(recs)

# --- severity oversampling (capped), fractional via per-utt probability ---
def is_short(r):
    """D6: short-command utterance (wake-word slice). OFF unless --oversample-short-words > 0."""
    if a.oversample_short_words <= 0:
        return False
    return 0 < len((r.get(a.target) or "").split()) <= a.oversample_short_words

def expand(recs, rng):
    out = []
    for r in recs:
        # etiology weight, capped as before; D6 short-multiplier applies on top, so the maximum
        # total duplication is sev_cap * oversample_short_mult (both 1.0-defaulted -> unchanged).
        w = min(a.sev_cap, WEIGHTS.get(r["etiology"], 1.0))
        if is_short(r):
            w *= a.oversample_short_mult
        k = int(w)
        if rng.random() < (w - k): k += 1
        out.extend([rec(r)] * max(1, k))
    return out

rng = random.Random(13)
train_sevw = expand(train, rng); random.Random(17).shuffle(train_sevw)
n_sevw = write(os.path.join(a.out_dir, "train_sevw.json"), train_sevw)
# guardrail subset of the WEIGHTED train (mirrors full-run distribution)
write(os.path.join(a.out_dir, "guardrail_train.json"), train_sevw[:a.guardrail_utts])
# Gate-0 wiring smoke (unweighted, small)
sm = [rec(r) for r in train]; random.Random(23).shuffle(sm)
write(os.path.join(a.out_dir, "smoke4k.json"), sm[:a.smoke_utts])

# --- fixed representative val3k (speaker-stratified round-robin over dev speakers) ---
by_spk = collections.defaultdict(list)
for r in sorted(dev, key=lambda x: x["id"]): by_spk[r["speaker"]].append(r)
val, idx, order = [], 0, sorted(by_spk)
while len(val) < a.val_utts:
    prog = False
    for s in order:
        if idx < len(by_spk[s]):
            val.append(by_spk[s][idx]); prog = True
            if len(val) >= a.val_utts: break
    if not prog: break
    idx += 1
write(os.path.join(a.out_dir, "val3k.json"), [rec(r) for r in val])
# PD forgetting probe (held-out PD utts) + full dev_internal
write(os.path.join(a.out_dir, "pd_probe.json"), [rec(r) for r in dev if r["etiology"] == "Parkinson's Disease"])
write(os.path.join(a.out_dir, "dev_internal.json"), [rec(r) for r in dev])

n_pd = sum(1 for r in dev if r["etiology"] == "Parkinson's Disease")
print(f"target={a.target} | speakers total={len(speakers)} train={len(train_spk)} dev={len(dev_spk)}")
print(f"train(raw)={len(train)} -> train_sevw={n_sevw} (+{100*(n_sevw/max(1,len(train))-1):.0f}% from oversampling)")
print(f"by etiology raw train: {dict(collections.Counter(r['etiology'] for r in train))}")
print(f"val3k={len(val)}  pd_probe={n_pd}  "
      f"guardrail={min(a.guardrail_utts, n_sevw)}  smoke4k={min(a.smoke_utts, len(train))}  cap={a.sev_cap}")
if a.oversample_short_words > 0:
    n_short = sum(1 for r in train if is_short(r))
    print(f"[D6] short-command oversampling ON: <={a.oversample_short_words} words x"
          f"{a.oversample_short_mult} | {n_short}/{len(train)} train utts qualify "
          f"({100*n_short/max(1,len(train)):.1f}%) | max total dup = "
          f"{a.sev_cap * a.oversample_short_mult:.2f}x")
else:
    print("[D6] short-command oversampling OFF (default) -> manifests identical to pre-D6 behavior")
print("wrote:", sorted(os.listdir(a.out_dir)))
