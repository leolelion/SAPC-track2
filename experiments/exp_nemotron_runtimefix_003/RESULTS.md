# exp_nemotron_runtimefix_003 - Packaged Runtime-Fix Guardrails

Date: 2026-06-23  
Pod: RunPod `3dwiczo41jeg1y`; stopped after artifact copy  
Candidate zip: `/workspace/finetune/eval/nemotron_runtimefix_codex/artifacts/nemo_submission_worker1_runtimefix.zip`  
Local copy: `/Users/o/Downloads/nemo_submission_worker1_runtimefix.zip`  
Zip SHA-256: `6fb803e08ee88385bcd7ca4348d6475c95d27f4900ac375916e76b1edd5a69f4`  
Zip size: 803.3 MiB  

## Executive Summary

- Built the patched Nemotron runtime-fix submission zip on the pod.
- Zip manifest is clean: 15 files, no `__pycache__`, no backup file.
- Runtime-fix package passes exact old-vs-new transcript hash parity on 120 rows.
- Runtime-fix package passes the 20-worker throughput guardrail on 500 rows with zero failures.
- Heavy startup diagnostics are gone by default; worker load p50 improved from 9.62 s to 4.43 s in the 120-row parity run.

## Packaging

Built with `scripts/package_nemotron_runtimefix.py` from source package `/workspace/finetune/nemo_submission` and patched model artifact `model_worker1_runtimefix.py`.

The zip remains on persistent RunPod workspace storage and was also copied locally after freeing enough space.

Remote path:

```text
/workspace/finetune/eval/nemotron_runtimefix_codex/artifacts/nemo_submission_worker1_runtimefix.zip
/Users/o/Downloads/nemo_submission_worker1_runtimefix.zip
```

## Parity Guardrail

20 workers, threads=1, CPUs 0-19, 120 Dev rows:

| Package | OK | Failed | Aggregate RTF | Throughput | Decode RTF p50 | Decode RTF p90 | Worker Load p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old | 120 | 0 | 0.0303 | 33.02x | 0.290 | 0.339 | 9.62 s |
| Runtime-fix | 120 | 0 | 0.0259 | 38.55x | 0.287 | 0.319 | 4.43 s |

Exact hash comparison:

```text
old_n=120 runtimefix_n=120 common=120 sha_len_ok_diffs=0
```

## Throughput Guardrail

20 workers, threads=1, CPUs 0-19, 500 Dev rows:

| OK | Failed | Empty Text | Total Audio | Wall Time | Aggregate RTF | Throughput | Decode RTF p50 | Decode RTF p90 | Worker Load p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0 | 86 | 4436.30 s | 76.37 s | 0.0172 | 58.09x | 0.281 | 0.318 | 4.48 s |

## Decision

The runtime-fix candidate is ready for the next Codabench submission from a systems standpoint.

It preserves hypotheses on the parity slice, removes the diagnostic startup storm, and keeps 20-worker batch decoding comfortably fast. The remaining risk is leaderboard/test distribution, not a known local packaging or throughput failure.

## Artifacts

- `artifacts/RUN_SUMMARY.json`
- `artifacts/nemo_submission_worker1_runtimefix.zip.sha256`
- `results_multiproc/parity_old_w20_t1_120.log`
- `results_multiproc/parity_old_w20_t1_120.jsonl`
- `results_multiproc/parity_runtimefix_w20_t1_120.log`
- `results_multiproc/parity_runtimefix_w20_t1_120.jsonl`
- `results_multiproc/runtimefix_w20_t1_500.log`
- `results_multiproc/runtimefix_w20_t1_500.jsonl`
