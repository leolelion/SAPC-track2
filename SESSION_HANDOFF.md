# SESSION HANDOFF — 2026-07-30 (S0/S1/S2 decode-lever sessions)

Read this, then `PLAN.md` "Current State Update (2026-07-30)". Supersedes nothing; `HANDOFF.md` and
`STATE_OF_WORK.md` are from 2026-07-01 and describe the zipformer era.

---

## 1. Where the competition stands

**Shipped and scored:** parakeet Arm A, ONNX fp32, greedy — **Test1 CER 19.01% / 416.9 ms**, on the
**Pareto frontier**, owning the low-latency corner. Dominates our old beam-4 zipformer (21.28 / 729 ms)
and takagi ×2. Non-dominated against yac3xn (18.10 / 592 ms). Artifact `sha256 b17ad0d8`,
480,939,149 B. Dev_diag severe gate 18.733% → Test1 19.01% = **+0.28 pt transfer**.

**Nothing was submitted today.** Both decode levers tried today failed their pre-registered gates.

## 2. What today established

### The error budget, measured through the official scorer for the first time
`scripts/error_decomposition.py` self-gates: it refuses to print unless it reconstructs
`utils/compute_metrics.py:parse_sgml_csdi` (Gate A) and reproduces `CharErrorRateMinTwoRefs` /
`WordErrorRateMinTwoRefs` to 1e-6 (Gate B). On Dev_diag severe, n=425, 28,986 ref chars, 5,430 errors:

| | count | CER pts |
|---|---|---|
| **deletions** | 3780 | **13.04** |
| substitutions | 1294 | 4.46 |
| insertions | 401 | 1.38 |
| empties (48 utts) | 1129 | **3.89** |

`3780 char-del / 760 word-del = 4.97 chars each` against a **4.91-char mean word** → the deletions are
**whole words**, ~13% of the reference. Emitted words are ~2 chars off. **Coverage failure, not phonetic.**

### CER tracks speaking RATE, not utterance length
| | spearman vs CER |
|---|---|
| speaking rate (ref chars/s) | **−0.254** |
| duration | +0.144 |
| word count | **+0.027** |

Rate quartiles: Q1 slowest **40.14%** → Q4 fastest **10.31%** (4× gradient). Inside the 13+ word
bucket, length held fixed: slow 19.76% vs fast **9.19%**. Longest-*duration* quartile has the
*lowest* CER (15.17%).

> **Correction to an earlier claim in this repo.** S0's "51% of error mass is 13+ word utterances" is
> a **mass** statistic that was read as a **difficulty** statistic. Long utterances carry half the
> error because they carry half the characters. **The axis is slow speech, not long speech.**

### Deletions are positionally flat
21.8 / 20.6 / 16.5 / 21.6 / 19.4 across normalised reference position; flat again in the 13+ bucket
alone. Runs 62.7% singletons. **Falsifies prediction-net drift and within-utterance left-context
exhaustion** — both predict a rising histogram.

## 3. What is now ruled out

| lever | verdict | evidence |
|---|---|---|
| empty tail as the target | **3.89 pts only** | S0 decomposition |
| joint unfreeze (D1) | falsified | empties 48→50, CER +0.05 |
| `input_gain` | falsified | 0/48 empties recovered |
| blank-logit penalty | **NO-GO, grid never run** | empties are MORE confidently blank (p10 margin 6.35 vs 2.61); β≈6 needed, flips 27% of all blanks |
| **RNN-T beam search** | **FALSIFIED** | every width worse; `dDel` positive for all |
| `max_symbols_per_step` | not binding | resolves to 10; slow speech has low symbols/frame |
| `input_finished` tail drop | real but ~80 ms | bounded by subsampling scale; cannot be 760 words |

### The beam grid (Dev_diag, official scorer)
| config | CER% | dCER | **dDel** | empties | wall × |
|---|---|---|---|---|---|
| **greedy control** | **18.733** | — | — | 48 | 1.00 |
| beam4 · mean | 20.256 | +1.52 | **+192** | 28 | 1.24 |
| beam2 · mean | 21.182 | +2.45 | +637 | 34 | 1.05 |
| beam2 · none | 21.912 | +3.18 | +927 | 47 | 1.05 |
| beam4 · none | 21.979 | +3.25 | +961 | 52 | 1.28 |
| beam2 · max_sym1 | 80.108 | +61.4 | +19018 | 224 | 1.05 |

`beam=1` parity verified first: **0/40 utterances differ** from the banked shipped decode; full-set
decomposition reproduces CER 18.7332% / charD 3780 exactly.

**Why it failed — the transferable part.** Greedy has no score at all; it takes a per-step argmax and
never pays for blank. A correctly-scored beam does pay, and RNN-T path probability is structurally
biased toward short output (each extra token multiplies in another sub-1 probability). Searching
harder found **higher-probability paths that are worse transcripts**. Greedy is not an approximation
to the max-probability path here — it is a better estimator than the thing it approximates.

**⇒ The path posterior is miscalibrated toward blank on dysarthric speech, and no decode-time search
fixes a miscalibrated posterior.** `length_norm=mean` worked exactly as designed (−1.72 pt at beam 4,
empties 48→28) and still lost, because the deletions it adds inside non-empty output outweigh the
empties it saves — and the empties were only 3.89 points.

**Decode-time levers on the CER axis are exhausted.** Four independent attacks. Do not propose a
fifth without a mechanism that survives the miscalibration argument above.

## 4. Open / not diagnosed

1. **`beam_max_symbols=1` collapses 4×** (27% of greedy's words, 224 empties, `dDel +19018`).
   Direction explicable (blank preference compounded by a 1-emission-per-frame cap against a model
   that emits in bursts; each step decodes ~2 encoder frames given `chunk=[9,16]`, 8× subsampling,
   `valid_out_len=2`). **Magnitude is not.** Ruled out: ORT output-buffer aliasing across hypotheses
   — it would have broken every beam config, not one. The local test asserts the cap at 1 vs 2
   emissions and passes. Config is dead either way; recorded as an honest unknown.
2. **Dev_clean2k was never measured for the beam configs** — the grid died at the Dev_diag gate, so
   `score_final` / `latency` never ran. Not a gap in the verdict (nothing cleared B1a), but the beam
   has no broad-slice or latency number attached to it.
3. `55c1784a` is **14.1% of all errors** (21 of 48 empties, char-deletion 85.4%). Never investigated
   as a speaker-specific phenomenon.

## 5. Recommendations, in order

1. **Time-compress slow audio before the front end** so the model sees a familiar frame-per-word
   ratio. Decode-time, no retraining, one CPU pod session. This is the only untried cheap lever and
   it attacks the 4× rate gradient directly instead of the search. **Untested — no code written.**
2. **The GPU session (the real answer).** The remaining CER is a training problem, now with a
   correctly identified target: sample/augment toward **slow** speech, not toward long utterances or
   short commands. Resume the un-converged Arm B curve (still descending at epoch 3), sweep FastEmit
   λ (never swept; base ships 0.03), top-k checkpoint averaging, speaker-disjoint val. This reframes
   D2–D5, which have sat on "needs re-justification" since D1 failed.
3. Leave the shipped submission alone. It is on the frontier and nothing today beat it.

## 6. Files touched today

**Code (all shipping-inert at defaults):**
- `track2_starting_kit/parakeet_onnx/model.py` — `_BeamHyp`, `_log_softmax`, `_prune`,
  `_decode_frames_beam`, dispatch in `_stream_step`, beam state in `reset()`, knobs in
  `_init_runtime_cfg`. **Ships `beam: 1` → dispatches to the untouched greedy path.** Also the
  blank-penalty hook from the earlier session, shipping `0.0`.
- `track2_starting_kit/parakeet_onnx/config.yaml` — `beam`, `beam_max_symbols`, `beam_expand`,
  `beam_prune_logp`, `beam_length_norm`, `blank_penalty`, all at inert defaults.
- `scripts/error_decomposition.py` — `_dp_table` extracted, `seq_align_trace`, `deletion_profile`,
  `spearman`, `report_deletion_localization`.
- `scripts/probe_blank_margin.py`, `scripts/run_blank_penalty_sweep.sh`, `scripts/run_deletion_beam.sh`
- `scripts/test_beam_decode.py`, `scripts/test_deletion_localization.py` — run with plain `python3`,
  no weights / SGML / SAP data needed. **Both pass.**

**Docs:** `investigations/step01_runbook.md`, `investigations/step02_runbook.md`,
`EXPERIMENT_LOG.md` (`exp_s0_s1_probe`, `exp_s2_beam`), `experiments/summary.csv`,
`experiments/PLANNED.md`, `experiments/exp_s0_s1_probe/`, `experiments/exp_s2_beam/`.

**Commits:** `6d275f9` → `9e0e1d9` → `dc6ae97` → `abc42c0` (S0/S1) → `92ee96d` (S2 prep) →
`fd400ae` (S2 results). All pushed to `fork/parakeet-onnx-ship`.

## 7. Pod / environment notes for next session

- Pod `3dwiczo41jeg1y` (H200, **$4.39/h**) — **STOPPED and verified** (`runpodctl pod list` → `[]`).
  Pod `1ppb7l0i5xuna8` is $2.99/h; `start-pod-watcher.sh` tries it first but breaks on whichever
  starts. Prefer the cheaper one for CPU-only work.
- **`/workspace` is a shared network filesystem** (`mfs#us-ca-2.runpod.net:9421`) mounted on *both*
  pods, so data and artifacts are available whichever one starts. This retires the "wrong pod" worry.
- SSH: `runpodctl ssh info <pod-id>` gives the exact command; key at `/Users/o/.runpod/ssh/RunPod-Key-Go`.
- Paths: data `/workspace/SAPC2` · artifact `/workspace/onnx_ship/extract32` (originals backed up as
  `*.shipped_backup`) · offline venv `/workspace/onnx_ship/offvenv32/bin/python` (ort 1.27.0) ·
  banked hypotheses `/workspace/onnx_ship/art32/Dev_diag.fp32.predict.csv` · worktree
  `/workspace/SAPC-step01` (now at `92ee96d`; `evaluate.sh` roots are env-parameterised as an
  **uncommitted local edit** — do not `git checkout` over it).
- **This pod's container `python3` has no torchmetrics.** Put `/workspace/venv/bin` first on `PATH`;
  `steps/eval/evaluate.sh` resolves `python3` from `PATH`. Wrap, never edit the scorer.
- Do **not** use `/workspace/SAPC-template` — divergent lineage (`main` @ c48d945, 3-arg
  `utils/metrics/wer.py`). The worktree is the verified 4-arg chain.
- Decode timing on Dev_diag (425 utts, 1 thread): greedy 3889 s; beam2 1.05×; beam4 1.28×.
  Projected Test wall-clock baseline 4620 s against a 15000 s budget.

## 8. Standing constraints (from `CLAUDE.md`, still in force)

`git add .` forbidden — stage individually · never edit `utils/compute_metrics.py`,
`utils/compute_latency.py`, `evaluate.sh`, `steps/eval/*`, `local_decode.py` semantics — wrap ·
**never submit to test a hypothesis**; upload only after the organizers' exact pipeline reproduces
the claimed metric on Dev · do all feasible local work before starting a paid pod, get explicit
approval for any file upload, stop the pod immediately after copying artifacts · SAP reference and
hypothesis text is licensed — strip it from anything committed to the public fork.
