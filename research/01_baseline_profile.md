# 01 - Baseline Profile

Executive summary:
1. Local profiling is possible with the exact submission zip, but it is not a substitute for a Linux/Codabench profile.
2. On a warm Apple M3 CPU run, encoder ONNX dominates wall time for a synthetic speech sample; decoder dispatch is visible but not primary.
3. The mel frontend does recompute features quadratically; on the 8.5 s sample it processed about 9x more audio than necessary.
4. ONNX Runtime profiling points to encoder `Conv`, `MatMulIntegerToFloat`, and `DynamicQuantizeMatMul` as the largest local node costs.
5. The next decisive measurement is a single-stream Linux/pod profile on real SAP audio, plus the official worker CPU diagnostic.

## Measurement Status

This is an initial local pilot, not the final Phase-1 profile.

Measured:
- Exact zip extracted from `/Users/o/Downloads/nemo_submission.zip`.
- Exact `model.py`, `localmel.py`, encoder ONNX, decoder ONNX, tokens, filterbank, and window.
- Local CPU execution through ONNX Runtime CPUExecutionProvider.
- Synthetic speech generated with macOS `say`, then resampled to 16 kHz.
- Reusable harness added at `/Users/o/Downloads/SAPC-template/scripts/bench_nemotron_single_stream.py`.

Not yet measured:
- SAP Dev or Dev_streaming data.
- Official scorer after any change.
- Codabench worker CPU.
- Linux x86 ONNX Runtime behavior.
- Concurrency sweep.
- >=100-run distributions. The local pilot used 12 repeated warm runs to avoid spending time on non-target hardware.

## Local Environment

- Machine: MacBook Air, Apple M3, 8 cores (4 performance + 4 efficiency), 8 GB RAM.
- OS: macOS 15.3.2 arm64.
- Python: 3.13.0.
- PyTorch: 2.6.0.
- ONNX Runtime: 1.24.4.
- Providers: `CoreMLExecutionProvider`, `AzureExecutionProvider`, `CPUExecutionProvider`; submission sessions force `CPUExecutionProvider`.
- Threads: `SAPC2_THREADS=4`.

## Static ONNX Inspection

Encoder:
- Inputs: `audio_signal`, `length`, `cache_last_channel`, `cache_last_time`, `cache_last_channel_len`.
- Outputs: `outputs`, `encoded_lengths`, `cache_last_channel_next`, `cache_last_time_next`, `cache_last_channel_next_len`.
- Nodes: 6911.
- Initializers: 1070.
- Top graph ops include `MatMulInteger`, `DynamicQuantizeLinear`, `Conv`, `MatMul`, `LayerNormalization`, `Softmax`.

Decoder:
- Inputs: `encoder_outputs`, `targets`, `target_length`, `input_states_1`, `input_states_2`.
- Outputs: `outputs`, `prednet_lengths`, `output_states_1`, `output_states_2`.
- Nodes: 42.
- Top graph ops: `LSTM`, `MatMul`, `Add`, `Transpose`, `Gather`.
- No duration/TDT output was found.

## Code-Level Hotspots Confirmed

Feature recompute:
- `_ensure_features()` concatenates every stored raw 100 ms chunk into a full utterance buffer.
- It then reruns `LocalMel` over the entire audio prefix.
- For a T-second utterance, the same early samples are transformed repeatedly.

Decoder loop:
- `_encode_and_decode()` runs encoder once per model chunk.
- Then for every encoded frame, it calls decoder ONNX until blank or up to `MAX_SYMBOLS_PER_FRAME=10`.
- This creates many small Python to ORT calls.

## Pilot Runs

### 2.0 s Noise

Purpose: smoke-test exact model load and instrumentation. This underexercises the decoder because output is blank-heavy.

Observed:
- Wall: 0.978 s.
- RTF: 0.489.
- Feature calls: 4, feature time 0.041 s.
- Encoder calls: 5, encoder time 0.910 s.
- Decoder calls: 33, decoder time 0.025 s.
- Final text: empty.

### 8.509 s Synthetic Speech, Single Run

Purpose: exercise real token emission without SAP data.

Observed:
- Wall: 2.188 s.
- RTF: 0.257.
- Feature calls: 16, feature time 0.074 s.
- Feature samples processed: 1,225,749 vs 136,149 actual samples, about 9.0x redundant.
- Encoder calls: 17, encoder time 2.000 s.
- Decoder calls: 168, decoder time 0.109 s.
- Partial updates: 15.
- First partial in batch mode: 0.945 s. This is not SAPC2 TTFT because there is no real-time chunk pacing.
- Final transcript was plausible for the generated speech sample.

### 8.509 s Synthetic Speech, 12 Warm Repeats

Purpose: separate warmup from steady-state on local hardware. Same model instance reused.

Distributions:
- Wall p50 0.989 s, p90 1.057 s, p99 1.586 s.
- RTF p50 0.116, p90 0.124, p99 0.186.
- Encoder time p50 0.890 s, p90 0.948 s.
- Decoder time p50 0.080 s, p90 0.085 s.
- Feature time p50 0.017 s, p90 0.048 s.

Per-run invariants:
- Encoder calls: 17.
- Decoder calls: 168.
- Feature calls: 16.
- Redundant feature factor: about 9.0x.
- Partial updates: 15.

Warmup note:
- Run 0 was much slower than later runs: 1.651 s wall vs p50 0.989 s.
- Cold/warm behavior is therefore large enough to matter for p90/p99 reporting.

### 8.509 s Synthetic Speech, Real-Time Pacing

Purpose: approximate callback visibility under 100 ms chunk arrival pacing.

Observed after one warmup utterance:
- Wall: 8.616 s for 8.509 s audio.
- Compute time: 1.360 s.
- Compute RTF: 0.160.
- First visible partial: 1.180 s after audio start.
- Last visible partial: 8.562 s.

This is still not official TTFT because the sample has no MFA speech-start timestamp.

## ONNX Runtime Node Profile

One profiled synthetic speech decode on local CPU:

Encoder profile:
- Node total: 1412 ms.
- Provider: CPUExecutionProvider for all nodes.
- Top op totals:
  - `Conv`: 466 ms over 1309 node events.
  - `MatMulIntegerToFloat`: 328 ms over 1224 node events.
  - `DynamicQuantizeMatMul`: 272 ms over 2465 node events.
  - `Transpose`: 56 ms.
  - `LayerNormalization`: 35 ms.

Decoder profile:
- Node total: 84.7 ms.
- Provider: CPUExecutionProvider for all nodes.
- Top op totals:
  - `LSTM`: 53.1 ms over 336 node events.
  - `MatMul`: 9.7 ms.
  - `FusedMatMul`: 5.9 ms.
  - `Gather`: 4.4 ms.

## Blocker Ranking From Local Evidence

This ranking is local-ARM-only and must be revalidated on the worker or Linux pod.

1. Encoder compute is the largest measured cost.
   - Evidence: warm speech p50 encoder time 0.890 s vs decoder 0.080 s and feature 0.017 s.
   - ORT profile points to convolution and quantized matmul kernels.

2. Mel recompute is real but not the local top wall-time contributor.
   - Evidence: 9.0x redundant samples on 8.5 s audio.
   - On local warm runs, median feature time was small, but this can grow with utterance duration and slower CPUs.
   - It remains a cheap, low-risk optimization candidate if bit-equivalence can be preserved.

3. Decoder Python-to-ORT dispatch exists but was not dominant on the synthetic sample.
   - Evidence: 168 decoder calls but only about 80 ms median total decoder time.
   - Risk: dysarthric speech may cause more nonblank symbols, corrections, or delayed blanks, so real SAP audio is required before deprioritizing it.

4. Cold-start / first-inference overhead affects tail latency.
   - Evidence: first warm-repeat run was about 67% slower than steady p50.
   - A model-load warmup pass may reduce p90/p99 if the harness permits it without harming budget.

5. Harness and worker contention remain unresolved.
   - Evidence missing: actual Codabench `[CPU_DIAGNOSTIC]`, concurrency, and Linux x86 profile.
   - The historical 24 process x 4 thread diagnostic harness is likely not representative of single-stream intrinsic latency.

## Next Measurements

1. Obtain Test1 ingestion log with `[CPU_DIAGNOSTIC]`.
2. Run `scripts/bench_nemotron_single_stream.py` on the Linux pod with real SAP audio staged on `/dev/shm`.
3. Enable ORT profiling on Linux for one representative short, medium, and long utterance.
4. Sweep `SAPC2_THREADS={1,2,4,8}` on the target-like host.
5. Only then implement one change at a time, starting with the highest measured expected value.

## Pod/Linux Update - 2026-06-22

Artifacts: `/Users/o/Downloads/SAPC-template/experiments/exp_nemotron_speed_001`.

Environment:
- RunPod `3dwiczo41jeg1y`.
- 192 logical CPUs, dual Intel Xeon Platinum 8568Y+, 2.0 TiB RAM.
- Model staged at `/dev/shm/nemo_submission_codex`.
- Real SAP Dev audio under `/workspace/SAPC2`.

E1 baseline, 15 utterances, threads=4:
- RTF p50 0.130, p90 0.152.
- Encoder p50 1.194 s, decoder p50 0.070 s, feature p50 0.030 s.
- Feature recompute factor p50 9.1x, p90 26.1x.

E2 thread sweep, 9 utterances:
- 1 thread: RTF p50 0.366, p90 0.502.
- 2 threads: RTF p50 0.173, p90 0.213.
- 4 threads: RTF p50 0.130, p90 0.156.
- 8 threads: RTF p50 0.113, p90 0.137.

E3 cold/no-warmup, 9 utterances, threads=4:
- RTF p50 0.134, p90 0.158.
- First partial proxy p50 0.481 s vs 0.445 s with one warmup utterance.
- Warmup is not a primary speed lever on this harness.

E4 ONNX profile, one 28.5 s long utterance, threads=8:
- Encoder node total 2906.6 ms.
- Decoder node total 147.1 ms.
- Encoder top ops: `Transpose` 603 ms, `DynamicQuantizeMatMul` 515 ms, `Conv` 328 ms, `MatMulIntegerToFloat` 328 ms.
- Decoder top ops: `LSTM` 84.9 ms, `MatMul` 19.0 ms, `FusedMatMul` 10.8 ms.

Updated blocker ranking from pod evidence:
1. Encoder/runtime kernels dominate.
2. Feature recompute is real and grows badly with duration, but is secondary on the large pod.
3. Decoder loop is not the dominant pod bottleneck, though it can still matter on weaker CPUs.
4. Thread count matters; 8 threads won on pod, but worker settings require worker CPU diagnostic.
