#!/usr/bin/env python3
"""
prepare_sapc_lhotse.py — Convert SAPC manifest CSVs to lhotse CutSets.

Usage:
    python3 finetuning/prepare_sapc_lhotse.py \
        --data-root /workspace/data \
        --output-dir finetuning/data

Reads:
    <data-root>/manifest/Train.csv
    <data-root>/manifest/Dev.csv

Writes:
    <output-dir>/cuts_train.jsonl.gz
    <output-dir>/cuts_dev.jsonl.gz
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from lhotse import CutSet, RecordingSet, SupervisionSet
from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment


def build_cutset(csv_path: Path, split_name: str) -> CutSet:
    df = pd.read_csv(csv_path)
    print(f"[{split_name}] {len(df)} utterances loaded from {csv_path}")

    recordings = []
    supervisions = []

    missing = 0
    for _, row in df.iterrows():
        audio_path = str(row["audio_filepath"])
        if not os.path.isfile(audio_path):
            missing += 1
            continue

        rec = Recording.from_file(audio_path, recording_id=str(row["id"]))
        recordings.append(rec)

        text = str(row["norm_text_without_disfluency"]).strip()
        sup = SupervisionSegment(
            id=str(row["id"]),
            recording_id=str(row["id"]),
            start=0.0,
            duration=float(row["duration"]),
            channel=0,
            text=text,
            speaker=str(row["speaker"]),
            custom={"etiology": str(row.get("etiology", ""))},
        )
        supervisions.append(sup)

    if missing:
        print(f"  WARNING: {missing} audio files not found — skipped.")

    rec_set = RecordingSet.from_recordings(recordings)
    sup_set = SupervisionSet.from_segments(supervisions)
    cuts = CutSet.from_manifests(recordings=rec_set, supervisions=sup_set)

    # Print statistics
    total_dur = sum(c.duration for c in cuts)
    hours = total_dur / 3600
    avg_dur = total_dur / len(cuts) if cuts else 0
    print(f"  {len(cuts)} cuts | {hours:.2f} hours | avg {avg_dur:.2f}s per utt")

    return cuts


def main():
    parser = argparse.ArgumentParser(description="Prepare SAPC lhotse CutSets")
    parser.add_argument("--data-root", required=True, help="Root of SAPC dataset")
    parser.add_argument("--output-dir", default="finetuning/data",
                        help="Where to write cuts_train/dev.jsonl.gz")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, fname in [("Train", "cuts_train.jsonl.gz"), ("Dev", "cuts_dev.jsonl.gz")]:
        csv_path = data_root / "manifest" / f"{split}.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found — skipping {split}")
            continue
        cuts = build_cutset(csv_path, split)
        out_path = out_dir / fname
        cuts.to_file(str(out_path))
        print(f"  Saved: {out_path}")

    print("\nDone. Next: python3 finetuning/finetune.py --config finetuning/finetune_config.yaml")


if __name__ == "__main__":
    main()
