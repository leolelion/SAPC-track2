#!/usr/bin/env python3
"""D2 prep — build a Train-provenance-only, hour-capped kNN-VC manifest for fine-tuning.

Implements the two constraints D0 established (2026-07-29):

  1. LEAKAGE FILTER (binding). kNN-VC contains 18,797 wavs derived from 41 *Dev* speakers and
     40,339 wavs whose source id matches neither Train nor Dev. Training on either would
     contaminate the Dev gate that is our only ship authority. Keep Train-provenance only;
     drop `unknown` rather than guessing.
  2. CAP + WEIGHTING. kNN-VC carries new acoustics but ZERO new lexical information (G-PROV
     100% exact SAP-Train text match), so it is augmentation: subsample to --hours, weight the
     high-empty etiologies (ALS, Down), and let the caller keep it under 25% of training steps
     and finish on real data.

Text comes from the SAP source utterance (kNN-VC is voice conversion of real SAP audio);
filenames are `{SAP_id}__v-{voice}.wav`, with a `_16kHz` variant.

Usage:
  build_knnvc_train_manifest.py --out /workspace/nemo_ft/knnvc_train.json --hours 60
"""
import argparse, csv, collections, json, os, random

ap = argparse.ArgumentParser()
ap.add_argument("--knnvc-root", default="/workspace/data/processed/SAPC2_v3_knnvc")
ap.add_argument("--train-csv", default="/workspace/SAPC2/manifest/Train.csv")
ap.add_argument("--dev-csv", default="/workspace/SAPC2/manifest/Dev.csv")
ap.add_argument("--out", required=True)
ap.add_argument("--hours", type=float, default=60.0, help="target total hours after weighting")
ap.add_argument("--seed", type=int, default=13)
# ALS and Down are the two highest-empty etiologies (24.5% / 7.5%); PD and Stroke had 0% empties.
ap.add_argument("--weights", default="als=2.0,ds=2.0,cp=1.5,stroke=1.5,pd=1.0")
a = ap.parse_args()

rng = random.Random(a.seed)
weights = dict(kv.split("=") for kv in a.weights.split(","))
weights = {k: float(v) for k, v in weights.items()}


def load(path, want_dur=False):
    ids, dur = {}, {}
    with open(path, newline="", errors="replace") as f:
        for r in csv.DictReader(f):
            if r.get("id"):
                ids[r["id"]] = r.get("text", "")
                if want_dur:
                    try:
                        dur[r["id"]] = float(r.get("duration") or 0.0)
                    except ValueError:
                        dur[r["id"]] = 0.0
    return ids, dur


train_text, train_dur = load(a.train_csv, want_dur=True)
dev_text, _ = load(a.dev_csv)
print(f"[sap] Train ids={len(train_text)} Dev ids={len(dev_text)}")


def source_id(fn):
    stem = fn[:-4].split("__v-")[0]
    if stem in train_text or stem in dev_text:
        return stem
    if stem.endswith("_16kHz"):
        return stem[: -len("_16kHz")]
    return stem


by_etio = collections.defaultdict(list)
counts = collections.Counter()
for dirpath, _d, files in os.walk(a.knnvc_root):
    rel = os.path.relpath(dirpath, a.knnvc_root)
    parts = rel.split(os.sep)
    etio = parts[1] if len(parts) > 1 and parts[0] == "wav" else "unknown_etio"
    for fn in files:
        if not fn.endswith(".wav"):
            continue
        sid = source_id(fn)
        if sid in train_text:
            counts["train"] += 1
            by_etio[etio].append((os.path.join(dirpath, fn), train_text[sid], train_dur.get(sid, 0.0)))
        elif sid in dev_text:
            counts["dev_DROPPED"] += 1
        else:
            counts["unknown_DROPPED"] += 1

print("[filter]", dict(counts))
for e, v in sorted(by_etio.items()):
    print(f"  {e:12s} train-provenance wavs={len(v)}")

# weighted target hours per etiology
wsum = sum(weights.get(e, 1.0) for e in by_etio)
picked = []
for e, items in sorted(by_etio.items()):
    share = weights.get(e, 1.0) / wsum
    budget_s = a.hours * 3600.0 * share
    rng.shuffle(items)
    acc = 0.0
    for path, text, dur in items:
        if acc >= budget_s:
            break
        if not text.strip():
            continue
        d = dur if dur > 0 else 5.44  # kNN-VC sampled median duration (D0)
        picked.append({"audio_filepath": path, "duration": d, "text": text.strip().lower()})
        acc += d
    print(f"  {e:12s} weight={weights.get(e,1.0):.1f} target={budget_s/3600:.2f}h picked={acc/3600:.2f}h")

rng.shuffle(picked)
with open(a.out, "w") as f:
    for r in picked:
        f.write(json.dumps(r) + "\n")
total_h = sum(r["duration"] for r in picked) / 3600.0
print(f"[out] {len(picked)} utts, {total_h:.2f} h -> {a.out}")
print("KNNVC_TRAIN_MANIFEST_DONE")
