#!/usr/bin/env python3
# Build NeMo JSON manifests from the SAP CSV manifests, SPEAKER-DISJOINT.
# Produces: train.json, dev_internal.json (held-out speakers), overfit.json (~30 hard utts for Gate 1),
# smoke_train.json (small subset for Gate 2). NeMo line format: {"audio_filepath","duration","text"}.
#
#   python3 prep_nemo_manifest.py --train-csv /workspace/SAPC2/manifest/Train.csv \
#       --data-root /workspace/SAPC2 --out-dir /workspace/finetune/nemo_ft/manifests \
#       --target norm_text_without_disfluency   # or: text (raw cased+punct) | norm_text_with_disfluency
import argparse, csv, json, os, random, collections
random.seed(13)

ap = argparse.ArgumentParser()
ap.add_argument("--train-csv", required=True)
ap.add_argument("--data-root", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--target", default="norm_text_without_disfluency",
                help="CSV column to use as the training transcript")
ap.add_argument("--dev-speaker-frac", type=float, default=0.08, help="fraction of speakers held out for internal dev")
ap.add_argument("--smoke-utts", type=int, default=4000)
ap.add_argument("--hard-etiologies", default="ALS,Cerebral Palsy")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

rows = list(csv.DictReader(open(a.train_csv)))
for r in rows:
    r["_abs"] = os.path.join(a.data_root, r["audio_filepath"])
    try: r["_dur"] = float(r["duration"])
    except Exception: r["_dur"] = 0.0

# speaker-disjoint split (anti-overfit; mirrors unseen Test1 speakers)
speakers = sorted({r["speaker"] for r in rows})
random.shuffle(speakers)
n_dev = max(1, int(len(speakers) * a.dev_speaker_frac))
dev_spk = set(speakers[:n_dev]); train_spk = set(speakers[n_dev:])
train = [r for r in rows if r["speaker"] in train_spk and r["_dur"] > 0 and os.path.exists(r["_abs"])]
dev   = [r for r in rows if r["speaker"] in dev_spk   and r["_dur"] > 0 and os.path.exists(r["_abs"])]

def write(path, recs):
    with open(path, "w") as f:
        for r in recs:
            txt = (r.get(a.target) or "").strip()
            if not txt: continue
            f.write(json.dumps({"audio_filepath": r["_abs"], "duration": r["_dur"], "text": txt}) + "\n")
    return sum(1 for r in recs if (r.get(a.target) or "").strip())

n_tr = write(os.path.join(a.out_dir, "train.json"), train)
n_dv = write(os.path.join(a.out_dir, "dev_internal.json"), dev)

# Gate-1 overfit set: ~30 short-ish utts from the hard etiologies (the empties we must learn)
hard = {e.strip() for e in a.hard_etiologies.split(",")}
hard_rows = [r for r in train if r["etiology"] in hard]
random.shuffle(hard_rows)
write(os.path.join(a.out_dir, "overfit.json"), hard_rows[:30])

# Gate-2 smoke train subset
sm = train[:]; random.shuffle(sm)
write(os.path.join(a.out_dir, "smoke_train.json"), sm[:a.smoke_utts])

# stats
def hours(recs): return sum(r["_dur"] for r in recs) / 3600.0
print(f"target column: {a.target}")
print(f"speakers: total={len(speakers)} train={len(train_spk)} dev_internal={len(dev_spk)}")
print(f"train: {n_tr} utts / {hours(train):.1f} h   dev_internal: {n_dv} utts / {hours(dev):.1f} h")
print(f"overfit.json: 30 hard utts ({a.hard_etiologies})   smoke_train.json: {min(a.smoke_utts,len(train))} utts")
print("by etiology (train):", dict(collections.Counter(r["etiology"] for r in train)))
print("wrote:", os.listdir(a.out_dir))
