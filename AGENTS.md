# AGENTS.md — SAPC2 Track 2 (Streaming ASR) Working Guide

> Repo-specific operating guide for coding agents. The **global defensive protocol**
> lives at `/Users/o/Downloads/AGENTS.md` and takes precedence on *how* to work
> (prediction protocol, stop-on-failure, verify-don't-assume). This file captures
> *what* this repo is and the contracts you must never break.

## What this repo is
A clone of the organizers' `xiuwenz2/SAPC-template` starting kit for the **Speech
Accessibility Project Challenge 2 (SAPC2)**. Our focus is **Track 2 — Streaming ASR
on dysarthric speech**, CPU-only, latency-scored.

- Challenge site: https://xiuwenz2.github.io/SAPC2-website/
- Upstream: https://github.com/xiuwenz2/SAPC-template
- Our working docs: `PLAN.md` (roadmap), `RESEARCH_NOTES.md` (lit/SOTA), `EXPERIMENT_LOG.md`
  (results), `experiments/` (per-run artifacts + `summary.csv`).

## The contracts you MUST preserve (do not break these)

### 1. Submission interface — `track2_starting_kit/<submission>/model.py`
Class **must** be named `Model` with exactly these methods:
- `__init__()` — load weights/tokenizer once.
- `set_partial_callback(fn)` — register `fn(text:str)`; called once per pass.
- `reset()` — per-file state reset.
- `accept_chunk(np.ndarray)` — float32, 16 kHz mono, **1600 samples (100 ms)**, returns partial `str`.
- `input_finished()` — returns final `str`.
All methods run on the Decoder thread only → no thread-safety needed.

### 2. Two ingestion passes (see `track2_starting_kit/local_decode.py`)
- **Pass 1 (accuracy)**: batch, multiprocess, no delay, callback is no-op → drives CER/WER.
- **Pass 2 (streaming)**: real-time 100 ms pacing, partials timestamped → drives TTFT/TTLT.
- Outputs: `--out-csv` (`id,raw_hypos`) for accuracy; `--out-partial-json` for latency.

### 3. Accuracy scoring — `evaluate.sh` → `steps/eval/*` → `utils/compute_metrics.py`
- sclite alignment → SGML → **min-over-two-refs** CER/WER.
- ref1 = `norm_text_with_disfluency`, ref2 = `norm_text_without_disfluency`.
- Per-utterance error **clipped at 1.0 (100%)**. **CER is the primary metric.**
- `unk` token handling: substitutions/deletions/insertions involving `unk` are reconciled
  (see `parse_sgml_csdi`). Do not emit literal `unk` in hypotheses expecting it to be free.

### 4. Latency scoring — `utils/compute_latency.py`
- `TTFT = first_non_empty_partial_time − (audio_send_start_time + mfa_speech_start)`
- `TTLT = final_visible_time − audio_end_oracle_time` (only counted if ≥ 0)
- Reported as **P50** (P90/P95 in detail). **Pareto rank uses mean(TTFT, TTLT).**
- `mfa_speech_start` comes from the `*_streaming.csv` manifest column.
- Implication: emitting a non-empty partial *earlier* lowers TTFT; finalizing fast after
  audio end lowers TTLT. Empty partials don't count toward TTFT.

### 5. Manifest schema — `utils/manifest.py`
Columns: `id, speaker, etiology, audio_filepath, duration, text`.
- `speaker` → build **speaker-disjoint** dev splits (anti-overfit).
- `etiology` (e.g. "Cerebral Palsy") → per-disease analysis / conditioning.
- `id` = wav stem; `speaker` = wav prefix before first `_`.

## Environment reality (verified)
- Submission runtime: Docker `xiuwenz2/sapc2-runtime:latest`, PyTorch 2.5.0+cu124,
  torchaudio/vision pre-installed. **Track 2 = CPU-only, 15000 s/submission.**
- `setup.sh` runs before model load; `requirements.txt` auto-installed after.
- Baseline `setup.sh` pins **linux x86_64 / cp311** wheels for `k2` + `kaldifeat`
  (`manylinux2014_x86_64`). **It will not install on macOS/arm natively.**
- This dev box is `darwin` → run the baseline only inside the Docker image (or a linux
  x86_64 CPU host). Do not claim a reproduction you have not actually executed.
- RunPod is expensive. **Do not start a pod to write code.** Before starting any paid pod,
  the local workspace must already contain the patch/script/command plan, plus the exact
  validation commands to run and the stop condition for shutting the pod down.

## Cost gate before starting RunPod
Before `runpodctl pod start ...`, do all feasible local work first:
- Update this guide/`PLAN.md` with the specific next gate and expected pod runtime.
- Write or patch the code locally, including packaging helpers and validation scripts.
- Run local syntax/import checks that do not need SAP data or Linux-only wheels.
- Prepare exact `scp`/`ssh` commands or a single shell script for the pod run.
- If uploading local workspace code/artifacts to RunPod, get explicit user approval for
  the exact file list first; sandbox policy may block external transfer without it.
- Decide the minimum experiment size first (smoke → guardrail → full run), and stop after
  the decision metric is known.
- Confirm how artifacts will be copied back, then stop the pod immediately after copying.

Current Nemotron no-submission gate:
- Prior Codabench log showed `Number of workers: 20`.
- Offline reproduction in `experiments/exp_nemotron_speed_002/` found 20-worker batch
  decoding is fastest and safest with **1 compute thread per worker**.
- Do **not** submit or revalidate the old `SAPC2_THREADS=4` default package.
- Patch artifacts already exist:
  `experiments/exp_nemotron_speed_002/model_worker1_runtimefix.diff` and
  `experiments/exp_nemotron_speed_002/model_worker1_runtimefix.py`.
- Packaging helper exists: `scripts/package_nemotron_runtimefix.py`. Run its `--dry-run`
  locally before starting a pod; if disk is tight, package on the pod with the same script
  rather than starting the pod to write packaging code.
- Runtime-fix candidate is now built and guardrailed:
  `/workspace/finetune/eval/nemotron_runtimefix_codex/artifacts/nemo_submission_worker1_runtimefix.zip`
  SHA-256 `6fb803e08ee88385bcd7ca4348d6475c95d27f4900ac375916e76b1edd5a69f4`.
- Guardrails passed in `experiments/exp_nemotron_runtimefix_003/`: old-vs-new 120-row hash parity has
  0 diffs; runtime-fix 20-worker 500-row run has 500/500 OK, aggregate RTF 0.0172, throughput 58.09x.
- Next action is Codabench submission of the runtime-fix zip, not another pod experiment, unless the
  submission path itself requires moving the zip.

## Local dev loop (once data + a linux/Docker host exist)
```bash
# 1) Decode (produces accuracy CSV + latency JSON)
cd track2_starting_kit
python3 local_decode.py \
  --submission-dir ./streaming_zipformer \
  --manifest-csv  $DATA_ROOT/manifest/Dev.csv \
  --streaming-manifest-csv $DATA_ROOT/manifest/Dev_streaming.csv \
  --data-root $DATA_ROOT \
  --out-csv ./Dev.predict.csv \
  --out-partial-json ./Dev.partial_results.json

# 2) Accuracy (stages: 0 install sclite, 1 build refs, 2 score)
./evaluate.sh --split Dev --hyp-csv track2_starting_kit/Dev.predict.csv --start_stage 0 --stop_stage 2
# 3) Latency
./evaluate.sh --start_stage 3 --stop_stage 3 \
  --partial-json track2_starting_kit/Dev.partial_results.json \
  --manifest-csv $DATA_ROOT/manifest/Dev_streaming.csv
```
Edit `DATA_ROOT`/`PROJ_ROOT`/`SCTK_DIR` placeholders in `evaluate.sh` first.

## House rules specific to this repo
- **Never edit** `utils/compute_metrics.py`, `utils/compute_latency.py`, `evaluate.sh`,
  `steps/eval/*`, or `local_decode.py` semantics — they mirror the organizers' scoring.
  Wrap, don't modify. If a wrapper needs them, call them as-is.
- Each new model = a **new sibling dir** under `track2_starting_kit/` (e.g.
  `streaming_zipformer_ft/`) with its own `model.py`/`config.yaml`/`setup.sh`. Keep the
  pristine `streaming_zipformer/` as the always-working reference baseline.
- Every experiment gets a row in `experiments/summary.csv` and a dir `experiments/exp_XXX/`
  (config snapshot + git hash + metrics json). Update `EXPERIMENT_LOG.md` and `PLAN.md`.
- `git add .` is forbidden (global rule). Stage files individually.

## Current status
See `PLAN.md` "Current State" section. As of session 1: docs + tracking scaffold created;
baseline NOT yet executed (no data, platform mismatch). Next gate: secure data + a linux/Docker
CPU host, then reproduce baseline Dev CER/WER + latency as the reference point.
