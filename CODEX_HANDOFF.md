# Codex Handoff - Nemotron SAPC2 Track 2

> Updated 2026-06-27 after the SOS-fix, severe/representative deploy evals,
> package build, and extracted-package gates.
> This is the compact current-state summary. Older detailed context remains in
> `HANDOFF_NEMOTRON_CODEX.md`, `research/20_fullrun_results.md`, `research/23_sos_fix_results.md`,
> `research/24_encoder_sos_eval_results.md`, `research/26_encoder_sos_package_results.md`,
> and `research/27_encoder_sos_package_eval_results.md`.

## Objective

Ship a CPU-only SAPC2 Track 2 streaming ASR submission using finetuned NVIDIA
`nemotron-speech-streaming-en-0.6b`, ONNX Runtime, local mel features, and no NeMo/network at
submission time. CER is primary; Pareto rank also uses mean(TTFT, TTLT).

## Current Decision

**Deploy candidate: encoder-only finetune + SOS fix.**

The deploy `model.py` must initialize the RNNT prediction net with blank/SOS:

```python
np.array([[BLANK_ID]], dtype=np.int32)
```

not token 0. `BLANK_ID=1024`.

## Key Results

### Full finetune, NeMo transcribe at `[70,1]`

Severe-enriched held-out `Dev_diag`:

| model | CER | empty |
|---|---:|---:|
| zero-shot base | 43.4% | 18% |
| **encoder-only finetune** | **23.6%** | 8% |
| full-unfrozen finetune | 25.2% | 6% |

Interpretation: finetuning works; encoder-only is better overall. Full-unfrozen shows overfit
signature despite slightly fewer empties.

### SOS-token deploy bug

Old streaming wrapper used `_last_token=[[0]]`, causing a leading `"Wh "` artifact for encoder-only.
The discriminating test in `scripts/run_sos_fix.sh` confirmed the root cause:

| model + SOS fix | Dev_100 CER | `"Wh "` prefixes | TTFT p50 | TTLT p50 |
|---|---:|---:|---:|---:|
| **encoder-only** | **1.3%** | **0/100** | 1.15 s | 0.31 s |
| full-unfrozen | 5.9% | 0/100 | 1.20 s | 0.28 s |

Conclusion: the earlier encoder-only streaming failure was a harness bug, not a model failure.

### Larger deploy-path CER

Encoder-only + SOS fix through real `local_decode.py` batch pass. These numbers were reproduced
unchanged from both the deploy directory and the extracted submission zip:

| set | n | speakers | CER | speaker-block 95% CI | empty | `"Wh "` |
|---|---:|---:|---:|---:|---:|---:|
| `Dev_diag.csv` severe-enriched | 425 | 103 | 24.49% | 16.06-33.16% | 30 | 0 |
| representative Dev-500, seed 23 | 500 | 119 | 10.49% | 7.92-13.55% | 6 | 0 |

Interpretation: representative Dev is comfortably below the current Zipformer Test1 reference
of 23.44% CER. The severe diagnostic set remains hard, mainly ALS and Down Syndrome empties/errors.

## Important Files

- `research/20_fullrun_results.md` - full finetune results.
- `research/23_sos_fix_results.md` - SOS bug confirmation and model decision.
- `research/24_encoder_sos_eval_results.md` - severe/representative CER + bootstrap CI.
- `scripts/run_sos_fix.sh` / `scripts/autorun_sosfix.sh` - SOS test.
- `scripts/streaming_cer_bootstrap.py` - local_decode CSV CER + speaker bootstrap.
- `scripts/run_encoder_sos_eval.sh` / `scripts/autorun_encoder_sos_eval.sh` - larger deploy eval.
- `scripts/run_package_encoder_sos.sh` / `scripts/autorun_package_encoder_sos.sh` - package build + smoke.
- `scripts/run_package_encoder_sos_eval.sh` / `scripts/autorun_package_encoder_sos_eval.sh` - extracted-package gates.
- `research/28_int8_quantization_plan.md` - literature-backed int8 plan plus independent-verifier fixes.
- `scripts/quantize_nemotron_encoder.py` - encoder dynamic-int8 quantization + ONNX IO/external-data preflight.
- `scripts/compare_cer_pairs.py` - paired FP32-vs-int8 CER/empty/`"Wh "` delta gate.
- `scripts/run_encoder_sos_int8.sh` / `scripts/autorun_encoder_sos_int8.sh` - int8 build, package, official eval, throughput, latency gates.
- `scripts/package_nemotron_runtimefix.py` - older generic zip helper.

## Pod / Infra

- RunPod pod id: `3dwiczo41jeg1y`, H200, host `38.80.152.249`; SSH port changes per restart.
- `/workspace` is persistent MooseFS; container disk resets.
- NeMo venv: `/workspace/nemoenv`.
- Base submission dir on pod: `/workspace/finetune/nemo_submission`.
- Encoder-only ONNX export: `/workspace/finetune/nemo_ft/export_full70_1`.
- Full-unfrozen ONNX export: `/workspace/finetune/nemo_ft/export_unfrozen70_1`.
- Always stop the pod when done. Autorun scripts use `/tmp/sapc_autorun.lock`.

## Constraints

- Do not modify scorer semantics: `local_decode.py`, `evaluate.sh`, `utils/compute_metrics.py`,
  `utils/compute_latency.py`, `steps/eval/*`.
- Submission must be offline: no NeMo, no HuggingFace, no network, no package that replaces NumPy.
- Use local mel + bundled ONNX Runtime wheel only.
- `git add .` is forbidden; stage files individually if committing.
- Push only to `fork`, never `origin`.

## Package Candidate

Built, smoke-tested, and package-gated:

- `/workspace/finetune/nemo_ft/artifacts/nemotron_encoder_sos_submission.zip`
- SHA-256 `0d20d3501fca443bd9da7ae423c2112c075517539242c87adc327a61dbf96f32`
- size 2.2G
- extracted-package 20-row smoke: 0.99% CER, 0 empty, 0 `"Wh "`
- extracted-package `Dev_diag.csv`: 24.49% CER, 30 empty, 0 `"Wh "`
- extracted-package representative Dev-500 seed 23: 10.49% CER, 6 empty, 0 `"Wh "`

See `research/26_encoder_sos_package_results.md` and
`research/27_encoder_sos_package_eval_results.md`.

Size note: the previous local Nemotron zip `/Users/o/Downloads/nemo_submission.zip` is 820M
on disk / 859,883,202 compressed bytes by `zipinfo`, and `HANDOFF_NEMOTRON_CODEX.md`
identifies it as a dynamic-int8 ONNX export. The new 2.2G artifact is larger because it is
a raw FP32 finetuned ONNX export with many external initializer files, not because finetuning
adds parameters.

## Int8 Candidate Status

Started 2026-06-28 on the user-provided pod endpoint
`root@38.80.152.249:30603`.

Encoder-only dynamic int8 quantization passed structural preflight:

- encoder input names match FP32 export
- encoder output names match FP32 export
- decoder checksum unchanged
- encoder ONNX Runtime session loads
- decoder ONNX Runtime session loads
- quantized encoder contains `DynamicQuantizeLinear` and `MatMulInteger` nodes

Built int8 package:

- `/workspace/finetune/nemo_ft/artifacts/nemotron_encoder_sos_int8_submission.zip`
- SHA-256 `c227d85bb9b099a8225aa28fe57ba53c56e49c6a9ffae64b5f43dca484413f5f`
- size 839M

Verified int8 accuracy so far:

| gate | FP32 CER | int8 CER | paired delta | empty | `"Wh "` | notes |
|---|---:|---:|---:|---:|---:|---|
| 20-row smoke, custom CER | 0.99% | 0.69% | -0.30% | 0 -> 0 | 0 -> 0 | official smoke also passed: 0.66% CER |
| representative Dev-500 | 10.49% | 10.60% | +0.12%, CI -0.23 to +0.50 | 6 -> 6 | 0 -> 0 | speaker CI for int8 CER 7.99-13.71% |
| `Dev_diag.csv` severe-enriched | 24.49% | 24.85% | not yet paired | 30 -> 31 | 0 -> 0 | int8 speaker CI 16.49-33.44% |

Interpretation: int8 is very likely viable on accuracy and solves the package-size concern
(839M, close to the old 860M class). It is not yet submission-ready because one official
`evaluate.sh` Dev-500 gate failed in the scorer path even though the custom CER path succeeded:

```text
ERROR: preds from sgml-ref1 and sgml-ref2 are not identical!
len(preds_ref1)=500, len(preds_ref2)=500
```

First inspected mismatch was an `unk`-related reference difference, but the scorer still exited.
Do not edit scorer semantics; debug the generated SGML/hypothesis normalization path or rerun a
smaller official gate that isolates the offending row.

**RESOLVED 2026-06-28 — see `research/30_official_eval_unk_assert.md`.** The pod-side
`diagnose_official_eval.sh` ran the official path for both packages: **`fp32 rc=1` AND
`int8 rc=1`, identically** => the assert is NOT int8-specific (int8 exonerated; both packages
blocked equally). Root cause: the **model emits a literal `unk` token in the hypothesis**
(e.g. utt000491 `... see new `unk` meet ...`). It aligns as a substitution vs ref1 (kept) but
an insertion vs ref2 (dropped by `process_unk`), so the two reconstructed hyps diverge by that
one `unk` (`only_unk_delta=True`). This is the `CLAUDE.md` contract #3 pitfall. **Fix is in the
submission `model.py` (ours, not the scorer):** strip standalone `unk` from what `accept_chunk`
and `input_finished` return. Provably complete — with no `unk` in the hyp, `pred_ref1==pred_ref2`
always, robust to any grader version. Then re-run the real `evaluate.sh` Dev-500 gate (expect
`rc=0`). Still need the `SGML_PRED_MISMATCH ... mismatches=N only_unk_token_delta=M` header to
confirm all mismatches are unk-only (N==M).

## 2026-06-28 Codex Continuation Note

Local follow-up verified the handoff against the repo and attempted the first next step, but RunPod
API access is blocked from this environment:

```text
runpodctl get pod 3dwiczo41jeg1y -a
Error: Post "https://api.runpod.io/graphql": dial tcp: lookup api.runpod.io: no such host
```

No pod artifacts could be pulled and the pod could not be stopped from this session.

Local tooling added for the next network-enabled run:

- `scripts/debug_sgml_pred_mismatch.py` diagnoses the exact ref1/ref2 SGML rows where reconstructed
  hypotheses differ; it is diagnostic only and does not change scorer semantics.
- `scripts/diagnose_official_eval.sh` is the next pod-side no-stop diagnostic: it reruns official
  Dev-500 scoring for both FP32 and int8 hypotheses, preserves SGML/TRN artifacts, and invokes the
  SGML mismatch debugger. It intentionally does not stop the pod.
- `scripts/run_encoder_sos_int8.sh` now preserves official scorer failure artifacts and tars
  `official_debug_<split>` before continuing remaining gates, then exits nonzero if any official
  gate failed.
- `scripts/autorun_encoder_sos_int8.sh` uploads the SGML debugger and pulls `official_debug_*`.
- `scripts/pull_encoder_sos_int8_artifacts.sh` can pull existing int8 artifacts and `/dev/shm`
  Dev_diag/Dev-500 CSV/JSON from a running pod, then stop the pod by default.

Big-picture scientific audit added in `research/29_bigpicture_audit.md`: the model-quality story is
sound; the immediate risk is eval-integrity/scorer robustness, not the finetune choice.

## Next Tasks

1. Pull the final int8 Dev_diag JSON/CSV artifacts from the pod and stop pod `3dwiczo41jeg1y`.
   The manual Dev_diag command completed successfully, but fresh `ssh`/`scp`/`runpodctl` commands
   were blocked locally by the Codex approval usage limit at 2026-06-28 01:29 CEST.
2. Debug the official `evaluate.sh` Dev-500 failure without changing scorer semantics. The int8
   candidate should not be submitted until the organizers' `local_decode.py` + `evaluate.sh` path
   passes on the extracted package.
3. Accept the int8 package only if the official extracted-package path passes and paired CER/empty
   deltas are acceptable. Custom CER scripts are diagnostics, not final sign-off.
4. Run a stricter offline package gate if a clean/no-network container is available:
   - no network
   - no NeMo import
   - `setup.sh` installs only bundled ORT as intended
5. Run the skipped int8 throughput/thread sweep and full `Dev_streaming` latency gate. Existing
   FP32 latency is TTFT p50 1.15 s / p90 2.36 s, TTLT p50 0.31 s / p90 0.35 s.
6. Investigate TTFT; p50 1.15 s is good but likely not optimal.
7. Submit either the package-gated FP32 fallback or the int8 successor after the same gates pass.
