# 03 - Recommendation

Executive summary:
1. Do not burn another Codabench submission with the current Nemotron package.
2. The immediate fix is runtime policy: 20-worker accuracy-pass processes should default to 1 compute thread, not 4.
3. Disable or gate the heavy CPU diagnostic in submitted packages; it becomes a synchronized startup load storm under 20 workers.
4. Warmup is possible only as bounded startup work in `__init__()`, but current evidence says it is not the next lever.
5. After the runtime fix passes exact-hypothesis parity, then revisit incremental mel and encoder/runtime optimization.

## What I Believe Now

Measured locally:
- Encoder ORT dominates the exact current stack on an Apple M3 synthetic speech sample.
- Decoder Python-to-ORT dispatch exists but is not the main local wall-time contributor.
- Mel recompute is objectively wasteful, processing about 9x redundant samples on an 8.5 s sample.
- Cold/warm effects are material.

Measured on RunPod/Linux with real SAP Dev audio:
- At 4 threads, 15 utterances decoded at RTF p50 0.130 / p90 0.152.
- Thread sweep on 9 utterances improved from RTF p50 0.366 at 1 thread to 0.113 at 8 threads.
- Cold/no-warmup at 4 threads was close to warmed 4-thread timing, so startup warmup is not a primary lever.
- ONNX profile on one long utterance showed encoder node total 2906.6 ms vs decoder 147.1 ms; encoder top ops were `Transpose`, `DynamicQuantizeMatMul`, `Conv`, and `MatMulIntegerToFloat`.

Inferred, not yet measured:
- The Codabench collapse could be encoder compute on a weaker CPU, harness/thread oversubscription, long-utterance feature recompute, decoder behavior on dysarthric speech, or a mixture.
- The local M3 result does not explain worker TTFT p90 6205 ms by itself; the worker is likely much slower and/or the official utterance distribution has harder/longer examples.

New Codabench log evidence:
- Q provided a previous ingestion log line showing `Number of workers: 20`.
- This substantially raises the probability that the accuracy pass is oversubscribed: 20 workers times the current default 4 ORT/PyTorch threads means about 80 compute threads and 20 Nemotron model copies.
- Because CER is computed from the batch pass, worker oversubscription is now the leading no-new-submission hypothesis for why Dev was about 25% CER on the pod but Test1 collapsed to about 51% CER.

Measured 20-worker topology on RunPod with CPU affinity constrained to 20 logical CPUs:
- threads=1, 120 rows: aggregate wall RTF 0.029, throughput 34.21x, decode RTF p50 0.285 / p90 0.316.
- threads=2, 120 rows: aggregate wall RTF 0.055, throughput 18.35x, decode RTF p50 0.858 / p90 0.995.
- threads=4, 120 rows: aggregate wall RTF 0.123, throughput 8.15x, decode RTF p50 2.220 / p90 2.561.
- A 40-row SHA audit found zero hypothesis differences between threads=1 and threads=4.
- Therefore lowering worker threads is a runtime-risk reduction with no observed accuracy effect.

## Priority Plan

### 1. Patch Runtime Policy Before Submission

Hypothesis:
- The Test1 worker's 20-process batch pass is oversubscribed by the current `SAPC2_THREADS=4` default.

Actions:
- Make a new submission variant with process-aware thread selection.
- Default non-main/worker processes to `SAPC2_THREADS=1`.
- Keep an explicit environment override for local profiling.
- Disable heavy CPU diagnostics by default; log only cheap facts unless `SAPC2_ENABLE_DIAGNOSTIC=1`.
- Run exact-hypothesis hash parity against the current package on a fixed Dev subset.

Expected gain:
- Large accuracy-pass throughput gain under the organizer topology: 120-row wall time improved from 137.09 s at 4 threads to 32.67 s at 1 thread in the offline reproduction.

Accuracy risk:
- Very low. The 40-row hash audit found no transcript changes across thread settings.

Decision rule:
- No new Codabench submission until the patched package matches current-package hashes on a fixed subset and reproduces the 20-worker `threads=1` throughput class.

### 2. Linux Single-Stream Profile On Real SAP Audio

Hypothesis:
- Real dysarthric audio and Linux x86 ORT will change the local blocker ranking.

Actions:
- Stage model and a representative short/medium/long SAP audio set on `/dev/shm`.
- Run a single-process benchmark with warm sessions.
- Report distributions: TTFT proxy, per-utterance RTF, per-chunk max compute, encoder/decoder/frontend split.
- Enable ORT profiling for representative utterances.

Expected gain:
- No direct gain; it prevents optimizing the wrong thing.

Accuracy risk:
- None.

Decision rule:
- Already satisfied for first-pass runtime ranking. Repeat only after code changes or if a new model/export is introduced.

### 3. Warmup

Hypothesis:
- First-inference cost contributes to tail TTFT and maybe official p90.

Action:
- Add an optional warmup in `__init__`: run one zero or tiny synthetic chunk sequence through encoder/decoder, then call `reset()`.
- Do not warm up in `reset()`, `accept_chunk()`, or `input_finished()`; those are per-file or in-stream methods and can affect latency/backlog.
- Keep warmup synthetic only. It must not inspect hidden/test audio or rely on any network/data fetch.

Expected gain:
- Low to medium for p90/p99; little p50 improvement.
- It may help first-utterance cold-start latency but will not fix sustained RTF > 1.

Accuracy risk:
- None if reset is complete.

Cost:
- Low runtime engineering cost, but it spends total Codabench budget. If official accuracy pass is multiprocess, the cost can repeat once per worker.

Decision rule:
- Accept if p90 first-utterance latency improves, transcript parity holds after `reset()`, and added startup wall time remains negligible relative to the 15000 s Track 2 limit.

Current evidence:
- Lower priority. Pod E3 showed only a small difference between cold/no-warmup and one warmup utterance at 4 threads.
- The organizer-style local harness runs the batch accuracy pass before the streaming pass, so the latency pass is likely already warmed unless Codabench differs.

### 4. Incremental Mel

Hypothesis:
- On longer real utterances and weaker CPUs, full-prefix mel recompute meaningfully contributes to lag, and eliminating it reduces p90 compute without changing transcripts.

Action:
- Implement rolling feature extraction in a new sibling submission directory, not by modifying scoring/harness files.
- Preserve exact `torch.stft(center=True, pad_mode="reflect")` behavior or prove any boundary difference does not affect CER.
- First compare mel tensors frame-by-frame against current `_ensure_features()` over many chunk prefixes.
- Then compare final transcripts on fixed audio.

Expected gain:
- Local M3 median feature time is only ~2% of wall on the synthetic sample, so local p50 gain is small.
- On long utterances, the redundant work grows with duration and can affect tails.

Accuracy risk:
- Low/medium. STFT boundary handling with `center=True` and reflect padding makes exact incremental parity non-trivial.

Cost:
- Medium.

Decision rule:
- Accept only if mel parity is exact or transcript/CER parity is proven on Dev.

### 5. Split Decoder And Joiner

Hypothesis:
- The current combined decoder+joint graph prevents the runtime architecture used by the successful on-device study: independent prediction net, joiner, per-component quantization, and better batching/cache handling.

Action:
- Find or produce ONNX exports with three graphs: encoder, prediction network, joiner.
- Rebuild greedy loop to run prediction net only on emitted nonblank tokens and run joiner over encoder frame/pred state as cheaply as possible.
- Evaluate whether joiner calls can be batched across frames when prediction state is unchanged after blanks.

Expected gain:
- Medium/high if decoder or joiner dispatch dominates on real SAP/worker, and still useful for structural parity with the on-device study.

Accuracy risk:
- Low if exported from the same checkpoint and numerically matched.

Cost:
- High.

Decision rule:
- Accept only after transcript parity and official scorer parity.

### 6. Encoder Runtime/Quantization Work

Hypothesis:
- Local profile's encoder dominance will also hold on worker, so optimizing encoder kernels is required.

Actions:
- Verify whether attention fusion appears in Linux ORT profile.
- Test optimized ORT model serialization for startup and runtime.
- Compare available precision variants from the exporter: fp32, int8-dynamic/current, int8-static, maybe int4 if available through the 2026 study/toolchain.
- Consider OpenVINO only if worker CPU is Intel and packaging can stay offline without replacing NumPy/Torch.

Expected gain:
- Potentially high, but highly hardware/runtime dependent.

Accuracy risk:
- Medium for quantization changes, low for graph optimization.

Cost:
- Medium/high.

Decision rule:
- Every quantized variant must pass the 2k Dev official scorer guardrail.

## What I Would Do If I Had To Pick One Change

If I had to edit one thing before another Codabench submission, I would patch the Nemotron submission runtime policy:

- worker processes use 1 compute thread by default;
- diagnostics are lightweight by default;
- local profiling can still override thread count explicitly;
- exact hypothesis hashes must match the current package on a fixed subset.

Why:
- It targets the measured failure mode produced by the actual ingestion topology.
- It does not touch model weights, decoding policy, scorer, or text normalization.
- The 40-row hash audit found no transcript differences.
- The measured wall-clock effect is much larger than warmup or incremental mel in current evidence.

Incremental mel remains a good engineering cleanup for long-tail latency, but it is no longer the first submission-blocking fix.

## Current No-Go Items

- Do not finetune Nemotron until speed is proven.
- Do not change scorer, `local_decode.py`, or official evaluation semantics.
- Do not assume sherpa-onnx supports this exact export; local API suggests it does not directly.
- Do not lower `MAX_SYMBOLS_PER_FRAME` or change emission policy without official CER/WER checks.
- Do not bundle/install packages that replace the worker NumPy unless packaging is revalidated end-to-end.
- Do not submit another package with `SAPC2_THREADS=4` as the default for 20-worker batch scoring.
