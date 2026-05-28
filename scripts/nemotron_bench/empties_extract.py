#!/usr/bin/env python3
"""Step 1 — extract empty-prediction set from saved Dev_10k Nemotron run.

For every utterance whose final transcription is empty/whitespace, write
out the manifest fields needed for the Step 2 characterizations.
"""
import argparse
import csv
import json
from pathlib import Path


def is_empty(text: str) -> bool:
    return not (text or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-csv", required=True)
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--timing-json", required=True, help="phase4b dev_10k_nemotron.timing.json")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--report-csv", default=None)
    args = ap.parse_args()

    # Load predictions
    hyps = {}
    with open(args.predictions_csv) as f:
        for r in csv.DictReader(f):
            hyps[r["id"]] = r.get("raw_hypos", "")

    # Load timing (per-utt compute time + audio sec)
    timing_by_id = {}
    with open(args.timing_json) as f:
        td = json.load(f)
    for r in td["per_utt"]:
        timing_by_id[r["id"]] = r

    # Iterate manifest, project the empty rows
    fields = [
        "id", "speaker", "etiology", "audio_filepath", "duration",
        "text", "norm_text_with_disfluency", "norm_text_without_disfluency",
        "mfa_speech_start", "mfa_speech_end",
        "vad_speech_start", "vad_speech_end",
    ]

    total = 0
    empty = 0
    empty_rows = []
    with open(args.manifest_csv) as f:
        for r in csv.DictReader(f):
            total += 1
            uid = r["id"]
            if uid not in hyps:
                continue  # not in predictions (shouldn't happen)
            if not is_empty(hyps[uid]):
                continue
            empty += 1
            ref_text = (r.get("norm_text_without_disfluency") or r.get("text") or "").strip()
            row = {k: r.get(k, "") for k in fields}
            row["reference_word_count"] = len(ref_text.split()) if ref_text else 0
            t = timing_by_id.get(uid)
            row["compute_sec"] = t["compute_sec"] if t else None
            row["audio_sec_from_timing"] = t["audio_sec"] if t else None
            empty_rows.append(row)

    print(f"total predictions: {len(hyps)}")
    print(f"manifest utts: {total}")
    print(f"empties: {empty} ({empty/max(1,total)*100:.2f}%)")

    # Save
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        if not empty_rows:
            return
        w = csv.DictWriter(f, fieldnames=list(empty_rows[0].keys()))
        w.writeheader()
        for r in empty_rows:
            w.writerow(r)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
