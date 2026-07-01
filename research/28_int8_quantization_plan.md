# 28 - Nemotron finetune int8 quantization plan

Date: 2026-06-27

## Decision

Quantize the finetuned encoder first. Keep the RNNT prediction network and joint/decoder graph FP32
for the first candidate.

This is the lowest-risk path because the encoder is the large part of the package and the dominant
CPU cost, while the decoder/joint graph is much smaller and more sensitive to token-level RNNT
dynamics. The prior danielbodart Nemotron package used an int8-dynamic encoder and FP32 decoder,
which explains its roughly 860 MB zip size versus the current 2.2 GB FP32 finetuned package.

Independent verification verdict: the core plan is sound, but the artifact gates must be stricter
than a custom CER script. This repo already had one false sense of safety from a proxy decode path;
the quantized package must be signed off through the organizers' real extracted-package path.

## Literature basis

Primary sources:

- Banfic et al. 2026, "Pushing the Limits of On-Device Streaming ASR":
  https://arxiv.org/abs/2604.14493
- ONNX Runtime quantization docs:
  https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ONNX Runtime threading docs:
  https://onnxruntime.ai/docs/performance/tune-performance/threading.html
- ONNX Runtime graph optimization docs:
  https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
- Daniel Bodart Nemotron ONNX export card:
  https://huggingface.co/danielbodart/nemotron-speech-600m-onnx
- Kim et al. 2021, "Integer-only Zero-shot Quantization for Efficient Speech Recognition":
  https://arxiv.org/abs/2103.16827
- Rybakov et al. 2023, "2-bit Conformer quantization for automatic speech recognition":
  https://arxiv.org/abs/2305.16619

Key takeaways:

- Banfic et al. identify Nemotron Speech Streaming as the strongest CPU streaming candidate among
  the models they benchmarked, and report post-training quantization reducing the model from
  2.47 GB to as low as 0.67 GB while keeping WER within 1 percent absolute of FP32.
- Their Nemotron results are directly relevant architecturally but not identical to this submission:
  they use a custom optimized ONNX pipeline and 32-core CPU measurements, while SAPC2 uses the
  submitted Python/ONNX wrapper inside the organizers' scoring harness.
- The danielbodart export card lists `int8-dynamic/` as the recommended Intel CPU variant:
  799 MB encoder weights plus 34 MB decoder weights, with dynamic quantization on MatMul weights.
- ONNX Runtime recommends dynamic quantization generally for transformer/RNN-style models and static
  quantization for CNN-style models, but also warns that quantization is not lossless and should be
  debugged by comparing float and quantized weights/activations when accuracy drops.
- ASR-specific quantization papers show int8 can be near-lossless for Conformer-class models, but
  low-bit or full-integer quantization can degrade accuracy if activation ranges and nonlinearities
  are mishandled. Therefore static activation quantization is a second-stage experiment, not the
  first submission candidate.

## Repo evidence that matters

Current FP32 package candidate:

- Zip: `/workspace/finetune/nemo_ft/artifacts/nemotron_encoder_sos_submission.zip`
- SHA-256: `0d20d3501fca443bd9da7ae423c2112c075517539242c87adc327a61dbf96f32`
- Size: 2.2 GB
- Extracted package `Dev_diag`: 24.49% CER, 30 empty, 0 `"Wh "`
- Extracted package representative Dev-500: 10.49% CER, 6 empty, 0 `"Wh "`

Prior int8 baseline package:

- Local zip: `/Users/o/Downloads/nemo_submission.zip`
- `zipinfo`: 859,883,202 compressed bytes
- Large files: `weights/encoder_model.onnx.data` about 752 MB compressed, decoder about 33 MB
- The old size difference is precision/export format, not baseline-vs-finetuned parameter count.

Threading history:

- `experiments/exp_nemotron_speed_001`: single-stream pod profiling favored 4-8 threads per process.
- `experiments/exp_nemotron_speed_002`: organizer-like 20-worker batch topology favored 1 thread per
  worker; 120-row wall RTF was 0.029 at 1 thread versus 0.123 at 4 threads, with exact hypothesis
  hash parity on the audit slice.
- `experiments/exp_nemotron_runtimefix_003`: worker1 runtime-fix package passed local throughput and
  parity guardrails.
- `CLAUDE.md` post-mortem later states that the worker1 runtime-fix was submitted and returned the
  same deterministic Test1 CER as the previous failed zero-shot package. Interpretation: the 20-worker
  fact is real and still matters for runtime, but it did not explain the old 51.52% zero-shot CER.

For the finetuned SOS-fixed package, thread count remains a systems gate. It should not be treated as
a model-quality fix, and it must not replace package-level CER scoring.

## Quantization candidates

### Candidate A: dynamic int8 encoder, FP32 decoder

Use ONNX Runtime `quantize_dynamic()` on `encoder_model.onnx` only. Target MatMul/Gemm-like weights
first, preserve external data, and leave decoder/joint FP32.

Expected:

- Biggest zip-size reduction with the least accuracy risk.
- Some CPU speed gain on MatMul-heavy encoder paths.
- Limited TTFT improvement if TTFT is dominated by algorithmic delay or partial emission cadence.

This is the first candidate.

### Candidate B: static QDQ encoder with SAP calibration

Use representative streaming encoder inputs collected from real SAP audio, including cache tensors
from the actual wrapper, then quantize the encoder with static QDQ.

Expected:

- Potentially faster than dynamic quantization if calibration is good.
- Higher accuracy risk because dysarthric speech, cache state, and long-tail severity can shift
  activation ranges.

This is second-stage only if Candidate A is not fast/small enough.

### Candidate C: decoder/joint quantization

Quantize the combined decoder graph after Candidate A passes. Use a very small gate first because
RNNT blank/nonblank boundaries are sensitive.

Expected:

- Modest size gain only; decoder is about 34 MB in the old int8 package.
- Possible latency gain in the inner token loop, but prior profiles show encoder dominates.

This is optional and should not block a submission if Candidate A works.

### Candidate D: int4 / k-quant

Scientifically promising from Banfic et al., but likely too much integration risk for the immediate
submission because it depends on newer ORT/operator support and a more specialized quantization path.

Keep as a later optimization after int8 package correctness is closed.

## Run plan

Do all feasible local work before starting RunPod:

Implemented local scripts:

- `scripts/quantize_nemotron_encoder.py`: copies the FP32 SOS submission tree, inspects encoder and
  decoder ONNX graphs, quantizes only `weights/encoder_model.onnx`, removes stale encoder external
  data, verifies IO signatures/session load/decoder checksum, and writes a JSON preflight report.
- `scripts/compare_cer_pairs.py`: computes paired per-utterance CER deltas, speaker-block bootstrap
  over deltas, empty-output changes, `"Wh "` changes, and etiology deltas.
- `scripts/run_encoder_sos_int8.sh`: pod-side end-to-end runbook for quantize, package, extracted
  smoke, representative/severe gates, official `evaluate.sh`, thread benchmark, and full latency.
- `scripts/autorun_encoder_sos_int8.sh`: Mac-side RunPod launcher/uploader/artifact puller/stopper.

Local verification passed:

- `bash -n scripts/run_encoder_sos_int8.sh scripts/autorun_encoder_sos_int8.sh`
- `python3 -m py_compile scripts/compare_cer_pairs.py scripts/quantize_nemotron_encoder.py`
- `git diff --check`

First launch attempt was blocked before any pod work started:

```text
command: runpodctl pod start 3dwiczo41jeg1y
error: start pod: There are not enough free GPUs on the host machine to start this pod.
likely cause: H200 host capacity, not a script/runtime failure
```

## Candidate A run results

Run started after the pod became available on 2026-06-28. The package was built from the
SOS-fixed encoder-only FP32 submission and quantized only `weights/encoder_model.onnx`.
Decoder/joint stayed FP32.

Structural preflight passed:

- encoder input/output names matched FP32
- decoder SHA-256 matched FP32 source
- encoder and decoder ONNX Runtime sessions loaded from the candidate layout
- quantized encoder contained `DynamicQuantizeLinear` and `MatMulInteger` nodes

Package:

- path: `/workspace/finetune/nemo_ft/artifacts/nemotron_encoder_sos_int8_submission.zip`
- SHA-256: `c227d85bb9b099a8225aa28fe57ba53c56e49c6a9ffae64b5f43dca484413f5f`
- size: 839M

Accuracy gates completed:

| gate | FP32 CER | int8 CER | paired delta | empty | `"Wh "` | comment |
|---|---:|---:|---:|---:|---:|---|
| 20-row smoke | 0.99% | 0.69% | -0.30% | 0 -> 0 | 0 -> 0 | official smoke passed at 0.66% CER |
| representative Dev-500 | 10.49% | 10.60% | +0.12%, CI -0.23 to +0.50 | 6 -> 6 | 0 -> 0 | no material regression |
| `Dev_diag.csv` severe-enriched | 24.49% | 24.85% | not yet paired | 30 -> 31 | 0 -> 0 | severe accuracy held within about +0.36 CER |

The Dev_diag int8 run printed:

```text
CER=24.85% n=425 speakers=103 empty=31 wh=0 speakerCI95=[16.49,33.44]%
  ALS                  n= 110 CER= 33.76%
  Cerebral Palsy       n= 152 CER= 23.08%
  Down Syndrome        n=  67 CER= 35.06%
  Parkinson's Disease  n=  77 CER=  5.97%
  Stroke               n=  19 CER= 27.87%
```

Interpretation: Candidate A is scientifically sound and likely useful. The size drops from the FP32
2.2G class to 839M, while representative Dev-500 moves by only +0.12 CER and severe Dev_diag by
about +0.36 CER. The SOS failure mode did not return.

Blocking issue before submission: the official `evaluate.sh` Dev-500 gate failed after loading and
matching all 500 predictions:

```text
ERROR: preds from sgml-ref1 and sgml-ref2 are not identical!
len(preds_ref1)=500, len(preds_ref2)=500
```

The first inspected SGML mismatch was `unk`-related, but the scorer still exited. Treat this as a
packaging/evaluation-path blocker, not a model-quality blocker, until isolated. Do not edit scorer
semantics.

Skipped in this run to control cost:

- thread/throughput sweep
- full `Dev_streaming` latency
- paired Dev_diag delta JSON pull/archival

Fresh `ssh`/`scp`/`runpodctl` commands were blocked locally by the Codex approval usage limit at
2026-06-28 01:29 CEST. A follow-up Codex attempt from the local workspace also could not reach the
RunPod API:

```text
runpodctl get pod 3dwiczo41jeg1y -a
Error: Post "https://api.runpod.io/graphql": dial tcp: lookup api.runpod.io: no such host
```

So the final Dev_diag artifacts still need to be pulled and pod `3dwiczo41jeg1y` still needs to be
stopped manually or by the next agent with working network/API access.

Continuation tooling added locally:

- `scripts/debug_sgml_pred_mismatch.py` identifies the exact SGML rows where official ref1/ref2
  reconstructed hypotheses diverge, without changing scorer semantics.
- `scripts/run_encoder_sos_int8.sh` now preserves official failure artifacts
  (`hyp/ref .trn`, SGML, manifest, hyp CSV) under `artifacts/official_debug_<split>/`, tars them,
  and continues the remaining gates before exiting nonzero if any official gate failed.
- `scripts/autorun_encoder_sos_int8.sh` uploads the SGML debugger and pulls `official_debug_*`
  artifacts.
- `scripts/pull_encoder_sos_int8_artifacts.sh` is a no-start recovery helper to tar/pull int8
  artifacts plus `/dev/shm` Dev_diag/Dev-500 CSV/JSON files, then stop the pod by default.

1. Write a pod-side script that:
   - Copies the SOS-fixed FP32 package/deploy tree.
   - Runs ONNX graph inspection on encoder and decoder.
   - Builds a dynamic-int8 encoder candidate.
   - Preserves external initializer files and names expected by `model.py`.
   - Rebuilds `nemotron_encoder_sos_int8_submission.zip`.
2. Add explicit checks:
   - ONNX Runtime version, opset, provider list, graph input/output names, cache tensor dtypes/shapes,
     and external-data filenames are recorded before and after quantization.
   - `model.py` still contains `_last_token=[[BLANK_ID]]`.
   - `CHUNK_NEW=16`.
   - Encoder int8 graph contains quantized MatMul path (`MatMulInteger`, `DynamicQuantizeLinear`,
     `DynamicQuantizeMatMul`, or equivalent ORT-produced nodes).
   - Decoder remains FP32 for Candidate A, with a recorded checksum against the FP32 source.
   - Zip has no nested root and includes required files.
   - The quantized model loads successfully from the exact extracted zip layout before SAP data decode.
3. Run gates in increasing cost:
   - 20-row extracted package smoke.
   - Dev_100 or the same SOS-fix 100-row set: CER, empties, `"Wh "` count, latency.
   - Representative Dev-500: CER, speaker-block CI, empties, `"Wh "` count.
   - `Dev_diag`: CER, speaker-block CI, etiology breakdown, empties.
   - Real `local_decode.py` plus `evaluate.sh` on the extracted package for final accuracy sign-off.
   - 20-worker batch throughput/threading sweep on the package with `SAPC2_THREADS=1` and `4`.
   - Full `Dev_streaming` latency under the real streaming pass if the CER gates pass.
   - Clean no-network/offline package gate before Codabench.

Do not use custom decode or custom CER as final sign-off. Custom scripts are useful for debugging and
fast iteration, but submission eligibility requires the extracted zip to pass the organizers'
`local_decode.py` and `evaluate.sh` path.

Because the FP32 model is larger than 2 GB, avoid making ORT graph-optimization steps part of the
first quantization run unless their output-size behavior is verified. Inspect and load the current
finetuned FP32 encoder first; do not rely on stale evidence from the older already-int8 baseline.

Stop conditions:

- Reject immediately if `"Wh "` returns or required package checks fail.
- Reject for submission if paired per-utterance CER deltas on representative Dev-500 show a material
  regression relative to the FP32 package, unless the FP32 package cannot be uploaded and int8 is the
  only viable path. Track the mean delta, paired bootstrap interval, and changed empty-output count.
- Reject if `Dev_diag` empty outputs increase materially, especially in ALS/Down Syndrome.
- Prefer int8 if CER is within tolerance and zip size is close to the previous 860 MB class.
- Fall back to FP32 if int8 hurts severe speakers or RNNT stability.

## Open assumptions

- The finetuned FP32 encoder export can be quantized with the same broad dynamic-int8 recipe as the
  prior baseline export.
- ONNX Runtime version in the bundled wheel supports the produced graph/operators.
- The pod CPU and Codabench CPU both benefit from the chosen quantized ops. This must be measured;
  quantization can reduce size without improving latency if kernels are not favorable.
- Dev-500 and `Dev_diag` are adequate guardrails for hidden Test1. They are strong local evidence,
  not proof.
- `SAPC2_THREADS=1` is the best default for the known 20-worker batch accuracy topology, not a
  universal latency setting. Measure the batch pass and streaming pass separately.
