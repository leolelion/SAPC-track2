#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_speaker_disjoint_dev.py — build a SPEAKER-DISJOINT Dev split (anti-overfit gate).

Why this exists
---------------
SAPC2 Track 2's binding methodology risk is Dev->Test transfer: a model can look
great on a Dev ruler that shares the TRAIN speaker pool, then lose on the held-out
Test speakers (research/37: Nemotron +18 Dev->Test vs Zipformer +1.8). AGENTS.md
section 5 says the `speaker` column exists precisely to build "speaker-disjoint dev
splits (anti-overfit)". This script operationalizes that: it partitions a manifest
into a HELDOUT set (speakers never seen in the POOL set) so evaluate.sh becomes an
honest Test proxy.

It does NOT touch audio, does NOT retrain, and does NOT modify scoring semantics —
it only re-partitions manifest CSVs (schema per utils/manifest.py:
id,speaker,etiology,audio_filepath,duration,text). Stdlib only.

Outputs (next to --out-prefix):
  <prefix>_heldout.csv   rows whose speaker is in the held-out speaker set
  <prefix>_pool.csv      the remaining rows (speaker-disjoint from heldout)
and, if --streaming-csv is given, the id-matched subsets:
  <prefix>_heldout_streaming.csv
  <prefix>_pool_streaming.csv

Usage:
  python3 scripts/make_speaker_disjoint_dev.py \
    --manifest-csv $DATA_ROOT/manifest/Dev.csv \
    --streaming-csv $DATA_ROOT/manifest/Dev_streaming.csv \
    --out-prefix $DATA_ROOT/manifest/Dev \
    --heldout-frac 0.3 --seed 0
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED_COLS = ["id", "speaker", "etiology", "audio_filepath", "duration", "text"]


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"[ERR] empty/headerless CSV: {path}")
        rows = list(reader)
    return reader.fieldnames, rows


def write_rows(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] wrote {len(rows):>6} rows -> {path}")


def speaker_of(row):
    # Prefer explicit column; fall back to id/wav prefix before first '_'.
    spk = (row.get("speaker") or "").strip()
    if spk:
        return spk
    key = (row.get("id") or row.get("audio_filepath") or "").strip()
    stem = Path(key).name
    return stem.split("_")[0] if "_" in stem else stem


def choose_heldout_speakers(rows, heldout_frac, seed):
    """Stratify by etiology so the heldout set covers every disease, then pick
    whole speakers (never split a speaker across pool/heldout)."""
    by_eti = defaultdict(set)
    for r in rows:
        by_eti[(r.get("etiology") or "UNK").strip()].add(speaker_of(r))
    rng = random.Random(seed)
    heldout = set()
    for eti, spks in sorted(by_eti.items()):
        spks = sorted(spks)
        rng.shuffle(spks)
        k = max(1, round(len(spks) * heldout_frac)) if spks else 0
        heldout.update(spks[:k])
    return heldout


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-csv", type=Path, required=True)
    ap.add_argument("--streaming-csv", type=Path, default=None,
                    help="Optional matching *_streaming.csv to subset by id.")
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--heldout-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout-speakers-out", type=Path, default=None,
                    help="Optional path to write the held-out speaker ids "
                         "(one per line, sorted) so the split can be audited.")
    ap.add_argument("--summary-json", type=Path, default=None,
                    help="Optional path to write a JSON audit of the split "
                         "(row/speaker/etiology counts + leakage assertion).")
    args = ap.parse_args()

    fieldnames, rows = read_rows(args.manifest_csv)
    if "speaker" not in fieldnames and "id" not in fieldnames:
        raise SystemExit(f"[ERR] manifest lacks 'speaker'/'id' columns: {fieldnames}")

    heldout_spk = choose_heldout_speakers(rows, args.heldout_frac, args.seed)
    heldout_rows = [r for r in rows if speaker_of(r) in heldout_spk]
    pool_rows = [r for r in rows if speaker_of(r) not in heldout_spk]

    # Hard invariant: no speaker appears in both sides.
    overlap = {speaker_of(r) for r in heldout_rows} & {speaker_of(r) for r in pool_rows}
    assert not overlap, f"speaker leakage: {sorted(overlap)[:5]}"

    write_rows(Path(f"{args.out_prefix}_heldout.csv"), fieldnames, heldout_rows)
    write_rows(Path(f"{args.out_prefix}_pool.csv"), fieldnames, pool_rows)
    pool_spk = {speaker_of(r) for r in pool_rows}
    print(f"[INFO] speakers: {len(heldout_spk)} heldout / "
          f"{len(pool_spk)} pool; "
          f"utts: {len(heldout_rows)} heldout / {len(pool_rows)} pool")

    stream_stats = None
    if args.streaming_csv:
        sfields, srows = read_rows(args.streaming_csv)
        held_ids = {r.get("id") for r in heldout_rows}
        s_held = [r for r in srows if r.get("id") in held_ids]
        s_pool = [r for r in srows if r.get("id") not in held_ids]
        write_rows(Path(f"{args.out_prefix}_heldout_streaming.csv"), sfields, s_held)
        write_rows(Path(f"{args.out_prefix}_pool_streaming.csv"), sfields, s_pool)
        stream_stats = {
            "heldout_streaming_rows": len(s_held),
            "pool_streaming_rows": len(s_pool),
            # parity check: streaming heldout ids must equal the manifest heldout ids
            # that are present in the streaming manifest (no orphan/missing streaming rows).
            "heldout_id_parity": sorted(held_ids & {r.get("id") for r in srows})
            == sorted({r.get("id") for r in s_held}),
        }

    if args.heldout_speakers_out:
        args.heldout_speakers_out.parent.mkdir(parents=True, exist_ok=True)
        args.heldout_speakers_out.write_text(
            "\n".join(sorted(heldout_spk)) + "\n", encoding="utf-8")
        print(f"[OK] wrote {len(heldout_spk):>6} heldout speakers -> "
              f"{args.heldout_speakers_out}")

    if args.summary_json:
        def eti_counts(rs):
            return dict(Counter((r.get("etiology") or "UNK").strip() for r in rs))
        summary = {
            "manifest_csv": str(args.manifest_csv),
            "seed": args.seed,
            "heldout_frac": args.heldout_frac,
            "speaker_leakage": sorted(heldout_spk & pool_spk),  # must be []
            "no_speaker_leakage": not (heldout_spk & pool_spk),
            "heldout": {
                "rows": len(heldout_rows),
                "speakers": len(heldout_spk),
                "etiology_rows": eti_counts(heldout_rows),
            },
            "pool": {
                "rows": len(pool_rows),
                "speakers": len(pool_spk),
                "etiology_rows": eti_counts(pool_rows),
            },
            "streaming": stream_stats,
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[OK] wrote split summary -> {args.summary_json}")


if __name__ == "__main__":
    main()
