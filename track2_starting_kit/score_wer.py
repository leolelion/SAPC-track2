#!/usr/bin/env python3
"""Score all result CSVs against dev100 references."""
import pandas as pd
import os
from difflib import SequenceMatcher

ref = pd.read_csv('/Users/o/Development/SAPC-template/dev100_bundle/Dev_100_local.csv')
ref = ref.set_index('id')

results_dir = '/Users/o/Development/SAPC-template/track2_starting_kit/local_results'

rows = []
for fname in sorted(os.listdir(results_dir)):
    if not fname.endswith('.csv'):
        continue
    path = os.path.join(results_dir, fname)
    try:
        pred = pd.read_csv(path).set_index('id')
        merged = ref.join(pred, how='inner')
        if len(merged) == 0:
            continue
        total_words, total_errors = 0, 0
        for _, row in merged.iterrows():
            r = str(row.get('norm_text_without_disfluency', '')).lower().split()
            h = str(row.get('raw_hypos', '')).lower().split()
            sm = SequenceMatcher(None, r, h)
            edits = sum(
                max(x[4] - x[3], x[2] - x[1])
                for x in sm.get_opcodes()
                if x[0] != 'equal'
            )
            total_words += len(r)
            total_errors += edits
        wer = total_errors / total_words * 100 if total_words > 0 else 0
        rows.append((fname, len(merged), wer))
    except Exception as e:
        rows.append((fname, -1, -1))
        print(f"ERROR {fname}: {e}")

rows.sort(key=lambda x: x[2])
print(f"{'Model':<50s}  {'N':>4}  {'WER%':>7}")
print("-" * 65)
for name, n, wer in rows:
    print(f"{name:<50s}  {n:>4d}  {wer:>7.1f}%")
