# D0 Run-book — synthetic dysarthric corpus forensics (2026-07-28)

CPU-only, no GPU, no NeMo, minutes. Runs on the pod **because the data lives there** — nothing else
about it needs a pod. Answers whether the two synthetic corpora contain parakeet's failure region
before any GPU is paid for.

Companion: `experiments/PLANNED.md` (registry + gates) · `investigations/parakeet_improvement_framework.md` (why).

---

## 0. Pre-registered gates — written before the run

| Gate | Measure | PASS | Consequence of FAIL |
|---|---|---|---|
| **G-COVER** | fraction of utts ≤3 words · onset (first 100 ms) dBFS p25 | ≥5% short **AND** p25 ≤ −45 dBFS | corpus lacks the failure region; D2/D3 demote from "empty fix" to generic robustness, expected gain on the 11.3-pt lever ≈ 0 |
| **G-PROV** | exact normalized-text match rate vs SAP `Train.csv` | < 80% | synth is a re-weighting of SAP, not new data; cap mix fraction hard, expect small gains |
| **G-EOS** | \|median trailing silence F5 − kNN-VC\| | < 300 ms | trailing-silence clash is the mechanical candidate for the observed EOS collapse; fixable by normalization, and the D7 mixing ban could be revisited |

Reference the gates are calibrated against (Dev_diag severe, 48 empties):
**22/48 are ≤3 words** · onset median **−58.5 dBFS** · full-utterance RMS median **−29.6 dBFS**.

---

## 1. Upload (two files, nothing else)

```bash
POD=<user>@<host> ; PORT=<port>            # fill from runpodctl
scp -P $PORT scripts/d0_synth_forensics.py $POD:/workspace/
scp -P $PORT experiments/exp_parakeet_ft_empties/empty_slice_refs.json $POD:/workspace/
```

`empty_slice_refs.json` is the 48-utterance empty slice (47 unique refs, 151 vocabulary items),
extracted from `zf_on_parakeet_empties.json`. It drives the empty-slice vocabulary coverage counts.

## 2. Step 1 — DISCOVER (always first; measures nothing)

The on-disk transcript layout is **unverified**. Discover reports it and exits.

```bash
ssh -p $PORT $POD 'cd /workspace && python3 d0_synth_forensics.py --mode discover \
  --f5-root    /workspace/data/processed/SAPC2_v3_synth \
  --knnvc-root /workspace/data/processed/SAPC2_v3_knnvc'
```

Read the output before continuing. Expect ~139,500 F5 files across `wav/{etiology}/dys/` and
~203,427 kNN-VC files across `wav/{etiology}/{severity}/`.

**Rule 0 applies.** If it prints `!! NO transcript files found under this root`, **stop** — G-PROV and
G-COVER's text half cannot run. Locate the transcripts and pass them explicitly via `--f5-text` /
`--knnvc-text` (the reader handles `.json`, `.jsonl`, `.csv`, `.tsv`, `.txt`). Do not guess.

## 3. Step 2 — AUDIT

```bash
ssh -p $PORT $POD 'cd /workspace && python3 d0_synth_forensics.py --mode audit \
  --f5-root    /workspace/data/processed/SAPC2_v3_synth \
  --knnvc-root /workspace/data/processed/SAPC2_v3_knnvc \
  --sap-train-csv $DATA_ROOT/manifest/Train.csv \
  --empty-refs /workspace/empty_slice_refs.json \
  --n-per-bucket 400 \
  --out /workspace/d0_forensics.json 2>&1 | tee /workspace/d0_forensics.log'
```

`$DATA_ROOT` is VERIFY-ON-POD — confirm the SAP manifest path before running. Without
`--sap-train-csv` the script still runs; G-PROV reports `N/A` and the provenance question stays open.

Audio is **sampled** (`--n-per-bucket 400`, seeded) — 5 etiologies × 2 corpora is a few thousand file
reads, not 343k. Per-bucket and aggregate distributions both land in the JSON.
`soundfile` is used when present, otherwise stdlib `wave` (16-bit PCM only).

Ends with `D0_FORENSICS_DONE` and a printed PASS/FAIL/N-A block per gate.

## 4. Copy back + stop

```bash
mkdir -p experiments/exp_d0_synth_forensics
scp -P $PORT $POD:/workspace/d0_forensics.json experiments/exp_d0_synth_forensics/
scp -P $PORT $POD:/workspace/d0_forensics.log  experiments/exp_d0_synth_forensics/
```

Then write the `experiments/summary.csv` row + `EXPERIMENT_LOG.md` entry and flip D0 to `done` in
`experiments/PLANNED.md`.

**Stop condition.** The three verdicts are the entire deliverable. Once they print, D0 is over — do
not explore further on a running pod. **If D1 (Arm B) is not running in the same session, stop the
pod.** D0 needs no GPU, so it should never be the reason a GPU pod stays up.

## 5. How each outcome changes the plan

| Outcome | Action |
|---|---|
| G-COVER passes for a corpus | that corpus contains the failure region → it is a live D2/D3 candidate |
| G-COVER fails for both | **no synthetic corpus can fix the empties.** Data expansion drops behind D1 + D5 entirely; do not spend GPU on D2/D3 |
| G-PROV fails (≥80% SAP text) | synth carries no new lexical information → treat purely as augmentation, cap at ≤25% of steps, expect small gains |
| G-EOS fails (large gap) | EOS collapse has a mechanical explanation → normalize trailing silence; D7's mixing ban becomes re-testable rather than permanent |
| kNN-VC passes, F5 fails | run **D2 only**, skip D3 — saves ~3 h GPU |

## 6. What this run is NOT
Not a training run. Not a decode. Touches no model, no scorer file, no submission. Reads audio
headers and samples; writes one JSON. Zero blast radius.
