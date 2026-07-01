# exp_nemotron_speed_002 - Codabench Topology Thread Sweep

Date: 2026-06-23  
Git: 16901c6  
Pod: RunPod `3dwiczo41jeg1y`, constrained with `taskset -c 0-19`  
Model: `/workspace/finetune/nemo_submission`, staged to `/dev/shm/nemo_submission_codex`  
Data: `/workspace/finetune/eval/dev_eval_2k.csv` with audio under `/workspace/SAPC2`

## Executive Summary

- The prior Codabench ingestion log reported `Number of workers: 20`.
- Reproducing that shape offline changes the optimal thread setting completely.
- With 20 worker processes on a 20-logical-CPU cpuset, `SAPC2_THREADS=1` is the clear accuracy-pass setting.
- The submitted/default `SAPC2_THREADS=4` is about 4.2x slower wall-clock than `threads=1` on the 120-row sweep.
- A 40-row hash audit found identical hypotheses between `threads=1` and `threads=4`, so lowering worker threads is a systems fix, not an accuracy gamble.
- The model's startup CPU diagnostic is also too heavy for 20 simultaneous workers and should be removed or gated before another submission.

## Results

All runs used 20 worker processes and callback disabled, matching the batch accuracy pass more closely than single-stream profiling.

| Rows | Threads / worker | OK | Wall s | Aggregate RTF | Audio throughput | Decode RTF p50 | Decode RTF p90 | Load p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 120 | 1 | 120 | 32.67 | 0.029 | 34.21x | 0.285 | 0.316 | 9.47 s |
| 120 | 2 | 120 | 60.93 | 0.055 | 18.35x | 0.858 | 0.995 | 10.06 s |
| 120 | 4 | 120 | 137.09 | 0.123 | 8.15x | 2.220 | 2.561 | 10.33 s |

Short hash audit:

| Rows | Threads / worker | OK | Wall s | Aggregate RTF | Decode RTF p50 | Decode RTF p90 |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 1 | 40 | 19.31 | 0.061 | 0.291 | 0.320 |
| 40 | 4 | 40 | 49.97 | 0.159 | 2.438 | 2.640 |

Hash comparison:

```text
threads=1 vs threads=4: common=40, missing=0, extra=0, sha_len_ok_diffs=0
```

## Interpretation

The single-stream pod result from `exp_nemotron_speed_001` favored 4-8 threads, but the organizer-like batch pass has 20 model processes. In that topology, per-process parallelism oversubscribes CPU and makes every utterance slower. This explains why a model can look viable in single-stream Dev profiling and still collapse on Codabench batch scoring.

The diagnostic logging is also part of the risk. Each worker currently performs CPU diagnostics and a 1024x1024 matmul benchmark during model initialization. Under 20 simultaneous workers, that startup work contends badly and delays model readiness. Keep environment logging lightweight in a submitted model.

## Decision

Do not spend another Codabench submission on the current package. First make a new submission variant with:

1. `SAPC2_THREADS=1` default for non-main/worker processes.
2. Optional override via environment variable for local experiments.
3. Heavy CPU diagnostics disabled by default.
4. No warmup by default unless a later latency-only experiment proves it pays for itself.

Then verify locally:

1. Exact hypothesis hashes match the current package on a fixed Dev subset.
2. 20-worker throughput remains close to the `threads=1` result above.
3. Single streaming pass remains comfortably faster than real time.

Patch status:

- Patched extracted package: `/tmp/nemo_submission_codex_profile/model.py`.
- Original extracted file backup: `/tmp/nemo_submission_codex_profile/model.py.before_worker1_runtimefix`.
- Reviewable patch artifact: `model_worker1_runtimefix.diff`.
- Patched `model.py` artifact: `model_worker1_runtimefix.py`.
- Local harness-like model-load smoke passed: cheap `[RUNTIME_INFO]` emitted, ONNX sessions and preprocessor loaded, no heavy `[CPU_DIAGNOSTIC]`.
- Upload zip was intentionally not rebuilt in this run because the local filesystem had only about 1.3 GiB free and the original zip is about 820 MiB.

When disk is available, package from the patched extraction with an explicit include list:

```bash
python3 scripts/package_nemotron_runtimefix.py \
  --source-dir /tmp/nemo_submission_codex_profile \
  --patched-model experiments/exp_nemotron_speed_002/model_worker1_runtimefix.py \
  --output-zip /path/to/nemo_submission_worker1_runtimefix.zip \
  --dry-run

python3 scripts/package_nemotron_runtimefix.py \
  --source-dir /tmp/nemo_submission_codex_profile \
  --patched-model experiments/exp_nemotron_speed_002/model_worker1_runtimefix.py \
  --output-zip /path/to/nemo_submission_worker1_runtimefix.zip
```

Do not blindly zip the whole extraction: it also contains local test caches and the `.before_worker1_runtimefix` backup file.

Current dry-run on this Mac refuses safely:

```text
ERROR: Not enough free space in /private/tmp: free=1.29 GiB, required_with_margin=1.64 GiB.
```

## Artifacts

- `results_multiproc/smoke_w20_t1_40.log`
- `results_multiproc/w20_t1_120.log`
- `results_multiproc/w20_t2_120.log`
- `results_multiproc/w20_t4_120.log`
- `results_multiproc/hash_w20_t1_40.log`
- `results_multiproc/hash_w20_t4_40.log`
- Matching `.jsonl` files for each run
- `model_worker1_runtimefix.diff`
- `model_worker1_runtimefix.py`
