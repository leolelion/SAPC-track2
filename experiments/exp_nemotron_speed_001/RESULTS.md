# exp_nemotron_speed_001 - Nemotron Runtime Profiling

Date: 2026-06-22  
Git: 16901c6  
Pod: RunPod `3dwiczo41jeg1y`, 192 logical CPUs, dual Intel Xeon Platinum 8568Y+, 2.0 TiB RAM  
Model: `/workspace/finetune/nemo_submission`, staged to `/dev/shm/nemo_submission_codex`  
Data: `/workspace/finetune/eval/dev_eval_2k.csv` with audio under `/workspace/SAPC2`

## Executive Summary

- The exact Nemotron submission runs much faster than real time on the 192-vCPU pod, but this does not prove it will run on Codabench.
- Linux profiling agrees with local profiling: encoder ONNX dominates, not the Python decoder loop.
- `SAPC2_THREADS=8` was best on the pod sample; `4` was close, `2` and `1` were materially slower.
- Cold/no-warmup at 4 threads was similar to one warmup utterance, so `__init__()` warmup is not a primary fix.
- Feature recompute is real and grows with duration: long utterances process about 26-32x redundant audio samples.

## E1 - Baseline Real-Audio Profile

Command shape:

```bash
python3 scripts/bench_nemotron_manifest.py \
  --submission-dir /dev/shm/nemo_submission_codex \
  --manifest-csv /workspace/finetune/eval/dev_eval_2k.csv \
  --data-root /workspace/SAPC2 \
  --out-jsonl results/e1_baseline_threads4_15.jsonl \
  --limit-per-bucket 5 \
  --threads 4 \
  --warmup-utterances 1
```

Results on 15 utterances, 5 short / 5 medium / 5 long:

| Metric | p50 | p90 |
|---|---:|---:|
| Compute RTF | 0.130 | 0.152 |
| First partial proxy, batch mode | 0.370 s | 0.592 s |
| Encoder time | 1.194 s | 2.903 s |
| Decoder time | 0.070 s | 0.205 s |
| Feature time | 0.030 s | 0.178 s |
| Feature redundant factor | 9.10x | 26.10x |

Long-only RTF p50 was 0.119. Short utterances had higher RTF because fixed chunk/cache overhead dominates.

## E2 - Thread Sweep

Sample: 9 utterances, 3 short / 3 medium / 3 long.

| Threads | RTF p50 | RTF p90 | First partial p50 | Encoder p50 | Decoder p50 | Feature p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.366 | 0.502 | 1.256 s | 3.189 s | 0.272 s | 0.048 s |
| 2 | 0.173 | 0.213 | 0.560 s | 1.454 s | 0.112 s | 0.026 s |
| 4 | 0.130 | 0.156 | 0.445 s | 1.093 s | 0.070 s | 0.029 s |
| 8 | 0.113 | 0.137 | 0.391 s | 0.961 s | 0.048 s | 0.024 s |

On this pod, 8 threads is best, but the gain from 4 to 8 is modest. The Codabench worker may have far fewer cores; this sweep is a pod-local result, not a worker setting.

## E3 - Cold/No-Warmup Check

At 4 threads, 9 utterances, no warmup utterance:

| Condition | RTF p50 | RTF p90 | First partial p50 |
|---|---:|---:|---:|
| 4 threads, warmed by 1 utterance | 0.130 | 0.156 | 0.445 s |
| 4 threads, cold/no warmup | 0.134 | 0.158 | 0.481 s |

Cold start is visible but small in this setup. Also, the organizer-style Track 2 local harness runs the batch accuracy pass before the streaming latency pass, so the streaming pass is likely already warm.

## E4 - ONNX Runtime Profile

One 28.5 s long ALS utterance, 8 threads:

- Compute RTF: 0.128.
- Encoder node total: 2906.6 ms.
- Decoder node total: 147.1 ms.

Encoder top ops:

| Op | Total ms |
|---|---:|
| Transpose | 603.1 |
| DynamicQuantizeMatMul | 515.5 |
| Conv | 328.2 |
| MatMulIntegerToFloat | 327.8 |
| LayerNormalization | 154.5 |
| Concat | 133.5 |
| Slice | 108.6 |
| MatMul | 103.4 |

Decoder top ops:

| Op | Total ms |
|---|---:|
| LSTM | 84.9 |
| MatMul | 19.0 |
| FusedMatMul | 10.8 |
| Add | 7.6 |
| Concat | 4.1 |
| Split | 3.9 |

## Artifacts

- `e1_baseline_threads4_15.log`
- `e1_baseline_threads4_15.jsonl`
- `e2_threads1_9.log`
- `e2_threads2_9.log`
- `e2_threads4_9.log`
- `e2_threads8_9.log`
- `e3_cold_threads4_9.log`
- `e4_onnx_profile_long_threads8.json`

## Decision

Keep measuring toward the worker target. On pod/Linux, the best immediate knobs are thread count and encoder/runtime optimization. Incremental mel is still worth implementing for long-tail efficiency, but pod evidence says it is secondary to encoder kernels. Warmup should not be prioritized.
