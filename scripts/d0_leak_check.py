#!/usr/bin/env python3
"""D0 addendum — split-provenance / leakage audit of the two synthetic corpora.

Unplanned but binding: kNN-VC filenames join to SAP ids, and a spot check found ids that
live in Dev.csv, not Train.csv. If synthetic training data is derived from Dev, any Dev
gate we run afterwards is contaminated. Quantify exactly, per corpus:

  kNN-VC: source utterance id -> {Train, Dev, unknown}
  F5-TTS: speaker-UUID bucket -> {Train speaker, Dev speaker, unknown}
"""
import csv, collections, json, os, sys

KNN_ROOT = "/workspace/data/processed/SAPC2_v3_knnvc"
F5_ROOT = "/workspace/data/processed/SAPC2_v3_synth"
MAN = "/workspace/SAPC2/manifest"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/d0_leak_check.json"


def load(split):
    ids, spks = set(), set()
    with open(os.path.join(MAN, f"{split}.csv"), newline="", errors="replace") as f:
        for r in csv.DictReader(f):
            if r.get("id"):
                ids.add(r["id"])
            if r.get("speaker"):
                spks.add(r["speaker"])
    return ids, spks


train_ids, train_spk = load("Train")
dev_ids, dev_spk = load("Dev")
print(f"[sap] Train ids={len(train_ids)} spk={len(train_spk)} | Dev ids={len(dev_ids)} spk={len(dev_spk)}")
print(f"[sap] speaker overlap Train&Dev = {len(train_spk & dev_spk)}")


def classify_id(sap_id):
    for cand in (sap_id, sap_id[:-len("_16kHz")] if sap_id.endswith("_16kHz") else sap_id):
        if cand in train_ids:
            return "train"
        if cand in dev_ids:
            return "dev"
    return "unknown"


knn = collections.Counter()
knn_spk = collections.defaultdict(collections.Counter)
for dirpath, _d, files in os.walk(KNN_ROOT):
    for fn in files:
        if not fn.endswith(".wav"):
            continue
        sap_id = fn[:-4].split("__v-")[0]
        cls = classify_id(sap_id)
        knn[cls] += 1
        knn_spk[cls][sap_id.split("_")[0]] += 1
print("[knnvc] source-split counts:", dict(knn))
print("[knnvc] distinct source speakers per class:", {k: len(v) for k, v in knn_spk.items()})

f5 = collections.Counter()
f5_buckets = {}
for dirpath, dirs, files in os.walk(F5_ROOT):
    wavs = [f for f in files if f.endswith(".wav")]
    if not wavs:
        continue
    spk = os.path.basename(dirpath)
    cls = "train" if spk in train_spk else ("dev" if spk in dev_spk else "unknown")
    f5[cls] += len(wavs)
    f5_buckets.setdefault(cls, set()).add(spk)
print("[f5] speaker-bucket split counts (wavs):", dict(f5))
print("[f5] distinct speaker buckets per class:", {k: len(v) for k, v in f5_buckets.items()})

json.dump({
    "sap": {"train_ids": len(train_ids), "dev_ids": len(dev_ids),
            "train_speakers": len(train_spk), "dev_speakers": len(dev_spk),
            "speaker_overlap_train_dev": len(train_spk & dev_spk)},
    "knnvc": {"wavs_by_source_split": dict(knn),
              "distinct_source_speakers": {k: len(v) for k, v in knn_spk.items()}},
    "f5tts": {"wavs_by_speaker_split": dict(f5),
              "distinct_speaker_buckets": {k: len(v) for k, v in f5_buckets.items()}},
}, open(OUT, "w"), indent=2)
print("D0_LEAK_CHECK_DONE", OUT)
