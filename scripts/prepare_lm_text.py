#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_lm_text.py — extract an LM training corpus from SAPC2 manifest(s).

Why this exists
---------------
RNN-LM shallow fusion (research/42 R1, the highest-EV zipformer accuracy lever) needs
a plain-text corpus to train the LM. The LM MUST share the ASR model's BPE tokenizer
(weights/data/lang_bpe_500/bpe.model), so this script does NOT tokenize — it only emits
one clean transcript per line. Tokenization with the shared bpe.model happens on the pod
during icefall rnn_lm training (see scripts/pod_train_rnnlm.sh).

This is deliberately pod-independent and stdlib-only: it reads the manifest schema from
utils/manifest.py (id,speaker,etiology,audio_filepath,duration,text) and writes text.
It does NOT touch audio, weights, or scoring. Runnable + testable on the darwin box.

Design choices (documented so the pod step is reproducible):
  - Source column is `text` (the manifest transcript), taken as-is except for
    whitespace collapse. We do NOT lowercase or strip punctuation here — the zipformer
    BPE was trained on LibriSpeech-style upper-case text; matching the ASR tokenizer's
    expectations is safer than imposing our own normalization. If the pod's bpe.model
    turns out to be lower-case, re-run with --lowercase.
  - By default we keep only manifest rows (the in-domain SAPC2 text). Pass extra
    --manifest-csv files to pool multiple splits (e.g. Train + Dev_pool), but NEVER
    include Dev_heldout or any Test text — that would leak the eval set into the LM.
  - Duplicate lines are kept by default (frequency is signal for an LM); pass --dedup
    to collapse them.

Usage:
  python3 scripts/prepare_lm_text.py \
    --manifest-csv $DATA_ROOT/manifest/Train.csv \
    --manifest-csv $DATA_ROOT/manifest/Dev_pool.csv \
    --out $DATA_ROOT/lm_text/corpus.txt \
    --stats-json $DATA_ROOT/lm_text/corpus.stats.json
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

WS = re.compile(r"\s+")


def read_texts(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"[ERR] empty/headerless CSV: {path}")
        if "text" not in reader.fieldnames:
            raise SystemExit(f"[ERR] manifest lacks 'text' column: {path} "
                             f"(has {reader.fieldnames})")
        for row in reader:
            yield row.get("text") or ""


def clean(line: str, lowercase: bool) -> str:
    line = WS.sub(" ", line).strip()
    if lowercase:
        line = line.lower()
    return line


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-csv", type=Path, action="append", required=True,
                    help="Manifest CSV(s) to draw text from. Repeatable. "
                         "Do NOT pass Dev_heldout/Test — that leaks the eval set.")
    ap.add_argument("--out", type=Path, required=True, help="Output corpus .txt")
    ap.add_argument("--stats-json", type=Path, default=None)
    ap.add_argument("--lowercase", action="store_true",
                    help="Lower-case all text (only if the bpe.model is lower-case).")
    ap.add_argument("--dedup", action="store_true",
                    help="Collapse duplicate lines (default keeps duplicates).")
    ap.add_argument("--min-chars", type=int, default=1,
                    help="Drop lines shorter than this after cleaning.")
    args = ap.parse_args()

    kept, dropped_empty = [], 0
    seen = set()
    for man in args.manifest_csv:
        for raw in read_texts(man):
            line = clean(raw, args.lowercase)
            if len(line) < args.min_chars:
                dropped_empty += 1
                continue
            if args.dedup:
                if line in seen:
                    continue
                seen.add(line)
            kept.append(line)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"[OK] wrote {len(kept)} lines -> {args.out} "
          f"(dropped {dropped_empty} empty/short)", file=sys.stderr)

    if args.stats_json:
        words = [w for ln in kept for w in ln.split()]
        stats = {
            "sources": [str(m) for m in args.manifest_csv],
            "lines": len(kept),
            "dropped_empty_or_short": dropped_empty,
            "deduped": bool(args.dedup),
            "lowercase": bool(args.lowercase),
            "total_words": len(words),
            "unique_words": len(set(words)),
        }
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"[OK] wrote stats -> {args.stats_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
