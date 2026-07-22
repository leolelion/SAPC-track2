# OPUS_STEP_BY_STEP_PLAN.md — SAPC2 Track 2 Next-Agent Execution Plan

> Purpose: give Opus a precise, step-by-step plan to advance the SAPC2 Track 2
> research project from the current local state. This plan assumes the repo root is
> `/Users/o/Downloads/SAPC-template`.
>
> Current date of plan: 2026-07-21.
>
> Critical rule: **do not start RunPod or any paid pod until the local prep checklist
> is complete and the user explicitly approves the exact files/commands/artifacts to
> transfer.** The next paid session should be run-only, copy-results-back, then stop.

---

## 0. Operating Constraints Opus Must Preserve

1. **Do not edit organizer scoring semantics.** Do not modify `track2_starting_kit/local_decode.py`,
   `evaluate.sh`, `steps/eval/*`, `utils/compute_metrics.py`, or `utils/compute_latency.py` except by
   wrappers/scripts that call them as-is.
2. **Do not mutate the pristine baseline.** Keep `track2_starting_kit/streaming_zipformer/` as the reference.
   Every variant must be a sibling dir under `track2_starting_kit/`.
3. **Track every experiment.** Each run gets `experiments/exp_XXX/` with config snapshot, git hash, logs,
   and metrics; add a row to `experiments/summary.csv`.
4. **No `git add .`.** Stage files individually if committing.
5. **No pod for coding.** Write/validate scripts locally first. Pod is for Linux-only scoring/training only.
6. **Do not claim a reproduction unless actually run through faithful harness:**
   `track2_starting_kit/local_decode.py` + `evaluate.sh`.

---

## 1. Current State to Verify First

Run:

```bash
cd /Users/o/Downloads/SAPC-template
git status --short
python3 -m py_compile scripts/make_speaker_disjoint_dev.py
bash -n scripts/pod_run_lm_and_latency_sweep.sh
python3 - <<'PY'
import csv
rows=list(csv.DictReader(open('experiments/summary.csv')))
print('summary rows', len(rows), 'fields', len(rows[0]))
for r in rows:
    if r.get('cer'):
        print(r['exp_id'], r['cer'], r['status'])
PY
```

Expected relevant local changes from Fugu:

- `research/42_fugu_independent_review.md` exists and is the canonical review/handoff.
- `scripts/make_speaker_disjoint_dev.py` exists and compiles.
- `scripts/pod_run_lm_and_latency_sweep.sh` exists and passes `bash -n`.
- `experiments/summary.csv` has backfilled accuracy rows:
  - `exp_a1_greedy_test1`
  - `exp_a1_beam4_test1`
  - `exp_nemotron_int8_test1`
  - `exp_v1_nemotron_devdiag`
  - `exp_armA_nemotron_devdiag`
  - `exp_peft_armC_nemotron_devdiag`

If any of these fail, fix them before moving on.

---

## 2. Read Only These Orientation Files

Do not reread the whole repo. Read these in order:

```bash
sed -n '1,260p' RESEARCH_SUMMARY_FOR_FABLE.md
sed -n '1,320p' research/42_fugu_independent_review.md
sed -n '1,220p' research/09_decoding_results.md
sed -n '1,180p' research/37_test1_gap_diagnosis_and_empties.md
sed -n '1,120p' research/41_v2_peft_armC_FAILED.md
cat track2_starting_kit/streaming_zipformer/config.yaml
```

Mental model to carry forward:

- Ship candidate today = fine-tuned streaming Zipformer A1, beam-4.
- Nemotron is stopped unless doing H7 closure only.
- Next high-EV work is Zipformer-only:
  1. speaker-disjoint gate,
  2. RNN-LM shallow fusion,
  3. latency Pareto sweep + beam-8 latency,
  4. possibly severity-aware Zipformer retrain later.

---

## 3. Local Prep Task A — Harden the Speaker-Disjoint Split Script

File: `scripts/make_speaker_disjoint_dev.py`

Goal: make it robust enough for real SAP manifests before the pod run.

Checklist:

1. Confirm it handles:
   - `speaker` column present,
   - fallback to prefix before `_` if `speaker` missing,
   - optional `--streaming-csv`,
   - exact id-matched streaming subsets.
2. Add `--heldout-speakers-out` optional output if useful, e.g. `<prefix>_heldout_speakers.txt`, so we can audit the split.
3. Add a small `--summary-json` optional output if useful, containing:
   - row counts,
   - speaker counts,
   - etiology row counts for pool/heldout,
   - leakage assertion result.
4. Re-run synthetic self-test.

Suggested validation command:

```bash
python3 -m py_compile scripts/make_speaker_disjoint_dev.py
# create synthetic CSVs under /tmp, run script, assert zero speaker overlap and streaming-id parity
```

Do not require pandas; keep it stdlib-only.

---

## 4. Local Prep Task B — Create Config-Only Zipformer Variants

These are local sibling dirs. Do **not** edit `streaming_zipformer/`.

Create:

```text
track2_starting_kit/streaming_zipformer_beam8/
track2_starting_kit/streaming_zipformer_cs8_lc128/
track2_starting_kit/streaming_zipformer_cs16_lc64/
track2_starting_kit/streaming_zipformer_cs16_lc256/
track2_starting_kit/streaming_zipformer_cs32_lc128/
```

Implementation:

```bash
cd /Users/o/Downloads/SAPC-template/track2_starting_kit
cp -a streaming_zipformer streaming_zipformer_beam8
cp -a streaming_zipformer streaming_zipformer_cs8_lc128
cp -a streaming_zipformer streaming_zipformer_cs16_lc64
cp -a streaming_zipformer streaming_zipformer_cs16_lc256
cp -a streaming_zipformer streaming_zipformer_cs32_lc128
```

Then edit each `config.yaml` only:

| dir | chunk_size | left_context_frames | decoding.method | num_active_paths |
|---|---:|---:|---|---:|
| `streaming_zipformer_beam8` | 16 | 128 | `modified_beam_search` | 8 |
| `streaming_zipformer_cs8_lc128` | 8 | 128 | `modified_beam_search` | 4 |
| `streaming_zipformer_cs16_lc64` | 16 | 64 | `modified_beam_search` | 4 |
| `streaming_zipformer_cs16_lc256` | 16 | 256 | `modified_beam_search` | 4 |
| `streaming_zipformer_cs32_lc128` | 32 | 128 | `modified_beam_search` | 4 |

Validation:

```bash
python3 - <<'PY'
from pathlib import Path
for d in Path('track2_starting_kit').glob('streaming_zipformer*'):
    cfg=d/'config.yaml'
    if cfg.exists():
        print('\n###', d)
        for line in cfg.read_text().splitlines():
            if any(k in line for k in ['chunk_size:', 'left_context_frames:', 'method:', 'num_active_paths:']):
                print(line)
PY
```

Do **not** attempt to import/run the model on macOS if Linux-only wheels are absent.

---

## 5. Local Prep Task C — Decide Whether to Build `earlyemit` Variant

`local_decode.py` counts TTFT from the first non-empty partial callback. Current `model.py:_decode_available()`
invokes the callback after each decoded chunk. It may already be as early as possible given `is_ready()`.

Before creating `streaming_zipformer_earlyemit_cs16/`, inspect whether the baseline suppresses early non-empty
partials anywhere. Search:

```bash
grep -R "partial_callback\|decoding_result\|is_ready\|done" -n track2_starting_kit/streaming_zipformer/model.py
```

If no suppression exists, **do not create an earlyemit variant**; mark it as not applicable in
`research/42_fugu_independent_review.md` or a new `research/43_...` note. The current pod script references
`earlyemit_cs16` only if the dir exists, so it is safe to skip.

If there is a suppression/flush policy that delays first non-empty partial, create a sibling dir and document
the exact one-line policy change. Keep it isolated.

---

## 6. Local Prep Task D — Prepare RNN-LM Shallow Fusion Skeleton

This is the only part that may need code beyond config clones. Do not overbuild locally if the icefall recipe
requires Linux-only dependencies; just stage a clean plan/script.

Target sibling:

```text
track2_starting_kit/streaming_zipformer_lm/
```

Initial setup:

```bash
cd /Users/o/Downloads/SAPC-template/track2_starting_kit
cp -a streaming_zipformer streaming_zipformer_lm
```

Then add placeholders/config in `streaming_zipformer_lm/config.yaml`:

```yaml
decoding:
  method: modified_beam_search
  num_active_paths: 4
  # TODO(opus/pod): enable once LM artifact exists and model.py wiring is verified
  lm_scale: 0.0
  lm_path: lm/epoch-best.pt
```

Opus must inspect the exact icefall/sherpa API available after `setup.sh` on Linux before wiring code. Search on pod:

```bash
find track2_starting_kit/streaming_zipformer/icefall -iname '*lm*' | head -50
grep -R "shallow\|lm_scale\|modified_beam" -n \
  track2_starting_kit/streaming_zipformer/icefall/egs/librispeech/ASR/zipformer \
  | head -100
```

Expected approach:

1. Train a small LM from SAPC2 manifest text only or SAPC2+allowed generic text.
2. Sweep `lm_scale` over a small grid, e.g. `0.05, 0.1, 0.2, 0.3`.
3. Gate on faithful `Dev_heldout` CER and latency.
4. If LM improves easy speakers but hurts ALS/DS severe tail, stop or try density-ratio/ILM later; do not ship a tail-regressing LM.

Important: If the icefall online code does not support shallow fusion cleanly, do **not** hack scoring or baseline semantics. Create a research note explaining the blocker and proceed with the latency sweep.

---

## 7. Local Prep Task E — Make Pod Script Match Actual Variants

File: `scripts/pod_run_lm_and_latency_sweep.sh`

After creating/skipping dirs above, update the script so its variant list is accurate.

Current expected dirs:

- required: `streaming_zipformer`
- optional if built: `streaming_zipformer_lm`
- optional if built: `streaming_zipformer_beam8`
- optional if built: latency sweep dirs listed in Task B

Make the script robust:

- If `streaming_zipformer_lm/` does not exist, print `[SKIP] LM dir missing` and continue.
- If `evaluate.sh` needs `DATA_ROOT`, `PROJ_ROOT`, or `SCTK_DIR` edited internally, document that explicitly at top of script.
- Ensure every run creates a unique `experiments/exp_.../` directory.

Validate:

```bash
bash -n scripts/pod_run_lm_and_latency_sweep.sh
```

---

## 8. Local Prep Task F — Research Note for This Prep

Create a new note:

```text
research/43_opus_pod_prep.md
```

Include:

- What variants were created.
- Any variant intentionally skipped and why.
- Exact files changed.
- Exact pod command to run after approval.
- Expected runtime and stop condition.
- What artifacts must be copied back.

Do not claim any CER/latency results.

---

## 9. Cost Gate Before Any Pod

Before asking the user to approve a pod, Opus must present:

1. Exact file list to upload/sync, e.g.:
   - `scripts/make_speaker_disjoint_dev.py`
   - `scripts/pod_run_lm_and_latency_sweep.sh`
   - `track2_starting_kit/streaming_zipformer_beam8/`
   - `track2_starting_kit/streaming_zipformer_cs*/`
   - `track2_starting_kit/streaming_zipformer_lm/` if created
   - `research/42_fugu_independent_review.md`
   - `research/43_opus_pod_prep.md`
   - `experiments/summary.csv`
2. Exact run command:

```bash
cd /workspace/SAPC-template
export DATA_ROOT=/workspace/SAPC2_DATA   # adjust to actual pod path
export PROJ_ROOT=/workspace/SAPC-template
bash scripts/pod_run_lm_and_latency_sweep.sh
```

3. Stop condition:
   - once `experiments/exp_*/accuracy.log`, `latency.log`, predictions, partial JSON, config snapshots, and `git_commit.txt` exist for all attempted variants,
   - copy `/workspace/SAPC-template/experiments/exp_*` back,
   - stop pod immediately.
4. Minimum experiment sequence:
   - smoke: run first 20-50 heldout utterances if easy to subset without changing scoring semantics;
   - guardrail: full `Dev_heldout` for baseline beam-4;
   - then variants.
5. Expected decision metric:
   - A variant must be non-dominated vs beam-4 on `(CER, mean(TTFT, TTLT))` on `Dev_heldout`.
   - Do not submit a variant that only improves in-pool Dev but regresses `Dev_heldout`.

Only after user approval should any pod start or any external transfer occur.

---

## 10. Pod Execution Plan After Approval

On the pod:

```bash
cd /workspace/SAPC-template
export DATA_ROOT=/workspace/SAPC2_DATA
export PROJ_ROOT=/workspace/SAPC-template
bash scripts/pod_run_lm_and_latency_sweep.sh 2>&1 | tee experiments/pod_run_lm_and_latency_sweep.log
```

After completion:

```bash
find experiments -maxdepth 2 -type f \( -name 'accuracy.log' -o -name 'latency.log' -o -name '*.csv' -o -name '*.json' -o -name '*.yaml' -o -name 'git_commit.txt' \) -print
```

Copy back:

```text
experiments/exp_000_zipformer_beam4_heldout/
experiments/exp_lm_shallowfusion_heldout/       # if run
experiments/exp_beam8_heldout/                  # if run
experiments/exp_latsweep_*/                     # if run
experiments/pod_run_lm_and_latency_sweep.log
$DATA_ROOT/manifest/Dev_heldout*.csv            # or copy generated manifests into experiment artifact dirs
```

Then stop the pod immediately.

---

## 11. Post-Pod Analysis Plan

Back on local machine:

1. Parse `accuracy.log` and `latency.log` into a table.
2. Add rows to `experiments/summary.csv`.
3. Create `research/44_zipformer_lm_latency_results.md` with:
   - table of CER/WER/TTFT/TTLT/latency mean,
   - Pareto plot or sorted Pareto table,
   - per-etiology if available,
   - clear ship/no-ship recommendation.
4. If a variant dominates beam-4 on heldout gate, prepare Codabench submission package.
5. If no variant wins:
   - keep beam-4 as live submission,
   - next candidate is severity-aware Zipformer retrain (medium-term GPU bet), not Nemotron.

---

## 12. What Not To Do

- Do not submit/revalidate old Nemotron runtime-fix unless the user explicitly asks; next action for Nemotron was Codabench submission of an already-built runtime zip, but research direction has pivoted away.
- Do not start a pod just to write scripts.
- Do not modify scoring code.
- Do not trust proxy scorers for a ship decision.
- Do not infer that a Dev-in-pool win transfers to Test; use `Dev_heldout`.
- Do not leave a paid pod running after the decision metric is known.

---

## 13. Success Criteria for Opus

Opus succeeds if it leaves the repo with:

1. Robust speaker-disjoint split tooling.
2. Config-only Zipformer variants staged.
3. Optional LM skeleton staged or a documented blocker.
4. A complete pod-run script that is syntax-clean and matches the actual variants.
5. `research/43_opus_pod_prep.md` documenting exact next commands and stop conditions.
6. No fabricated metrics and no unauthorized pod/external transfer.

If pod approval is granted and run completes, success becomes:

1. Faithful-harness results copied back.
2. `experiments/summary.csv` updated.
3. Pareto recommendation written in `research/44_zipformer_lm_latency_results.md`.
4. Pod stopped.
