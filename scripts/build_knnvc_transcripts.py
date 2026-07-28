#!/usr/bin/env python3
"""D0 helper — recover kNN-VC transcripts by joining wav filenames to SAP Train.csv.

kNN-VC filenames are `{SAP_id}__v-{voice}.wav`; kNN-VC is voice conversion of REAL SAP
audio, so the transcript is the source utterance's. F5-TTS filenames (`sev__slotNNNN__llm.wav`)
carry no such join key — their LLM-generated text is not on this pod, so F5's text half of
G-COVER/G-PROV stays unanswerable here (reported N/A, never guessed).

Writes a NeMo-style jsonl consumable by d0_synth_forensics.py --knnvc-text.
"""
import csv, json, os, sys

KNN_ROOT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/processed/SAPC2_v3_knnvc"
TRAIN_CSV = sys.argv[2] if len(sys.argv) > 2 else "/workspace/SAPC2/manifest/Train.csv"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/workspace/knnvc_transcripts.jsonl"

id2text = {}
with open(TRAIN_CSV, newline="", errors="replace") as f:
    for r in csv.DictReader(f):
        if r.get("id"):
            id2text[r["id"]] = r.get("text", "")
print(f"[sap] {len(id2text)} Train ids")

n_wav = n_join = n_miss = 0
missing_examples = []
with open(OUT, "w") as out:
    for dirpath, _dirs, files in os.walk(KNN_ROOT):
        for fn in files:
            if not fn.endswith(".wav"):
                continue
            n_wav += 1
            stem = fn[:-4]
            sap_id = stem.split("__v-")[0]
            # some kNN-VC sources were resampled copies: `<sap_id>_16kHz`
            text = id2text.get(sap_id)
            if text is None and sap_id.endswith("_16kHz"):
                text = id2text.get(sap_id[: -len("_16kHz")])
            if text is None:
                n_miss += 1
                if len(missing_examples) < 5:
                    missing_examples.append(fn)
                continue
            n_join += 1
            out.write(json.dumps({"audio_filepath": os.path.join(dirpath, fn), "text": text}) + "\n")

print(f"[knnvc] wavs={n_wav} joined={n_join} missing={n_miss} "
      f"join_rate={100.0*n_join/max(1,n_wav):.2f}%")
if missing_examples:
    print("[knnvc] unjoined examples:", missing_examples)
print("KNNVC_TRANSCRIPTS_DONE", OUT)
