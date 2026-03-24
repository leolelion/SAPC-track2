#!/usr/bin/env python3
"""Convert SAPC2 manifest CSVs to Lhotse CutSets for Icefall training."""

import argparse
import csv
from pathlib import Path

from lhotse import CutSet, RecordingSet, SupervisionSet
from lhotse import Recording, SupervisionSegment


def load_manifest(csv_path: Path, data_root: Path):
    recordings = []
    supervisions = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = data_root / row["audio_filepath"]
            if not audio_path.exists():
                print(f"  WARNING: Audio not found: {audio_path}")
                continue

            utt_id = row["id"]
            duration = float(row["duration"])

            # Use Recording.from_file() to auto-detect sampling_rate and num_samples
            try:
                rec = Recording.from_file(str(audio_path), recording_id=utt_id)
            except Exception as e:
                print(f"  WARNING: Could not read {audio_path}: {e}")
                continue

            recordings.append(rec)

            # Use norm_text_without_disfluency — matches evaluation scoring
            text = row.get("norm_text_without_disfluency") or row.get("text", "")
            text = text.strip()

            supervisions.append(SupervisionSegment(
                id=utt_id,
                recording_id=utt_id,
                start=0.0,
                duration=rec.duration,
                text=text,
                speaker=row.get("speaker", "unknown"),
                language="en",
            ))

    return (
        RecordingSet.from_recordings(recordings),
        SupervisionSet.from_segments(supervisions),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Lhotse CutSets from SAPC2 manifest CSVs"
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root directory that audio_filepath values are relative to")
    parser.add_argument("--train-csv", type=Path, required=True,
                        help="Path to Train.csv")
    parser.add_argument("--dev-csv", type=Path, required=True,
                        help="Path to Dev.csv")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write sapc2_{train,dev}_cuts.jsonl.gz")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split, csv_path in [("train", args.train_csv), ("dev", args.dev_csv)]:
        print(f"\nProcessing {split} from {csv_path} ...")
        recs, sups = load_manifest(csv_path, args.data_root)
        cuts = CutSet.from_manifests(recordings=recs, supervisions=sups)
        out_path = args.output_dir / f"sapc2_{split}_cuts.jsonl.gz"
        cuts.to_file(str(out_path))
        total_hours = sum(c.duration for c in cuts) / 3600
        print(f"  Saved {len(cuts)} cuts ({total_hours:.1f} h) → {out_path}")

    print("\nDone. Verify with:")
    print(f"  python3 -c \"from lhotse import CutSet; c=CutSet.from_file('{args.output_dir}/sapc2_train_cuts.jsonl.gz'); print(len(c))\"")


if __name__ == "__main__":
    main()
