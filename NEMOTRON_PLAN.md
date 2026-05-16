# Nemotron Streaming ASR — Replication & Fine-tune Plan (SAPC2 Track 2)

## Goal

Adapt NVIDIA Nemotron Speech Streaming (0.6B, cache-aware FastConformer-RNNT)
into a dysarthric-speech streaming ASR submission for SAPC2 Track 2: CPU-only,
RTF < 1.0, 100 ms input chunks, evaluated on CER/WER + TTFT/TTLT latency.

## Why this model

`arXiv:2604.14493` (Banfic et al., Microsoft CoreAI, Apr 2026) shows Nemotron
int4 ONNX runs at **7.2x real-time on a plain AVX2 CPU** (no VNNI/AMX) with
8.20% streaming WER on clean English. The 7x RTF headroom means the Track 2
RTF gate is comfortably clearable — unlike our zero-shot NeMo PyTorch run
(RTF 1.117, FAIL — logged as `2026-05-16-nemotron-streaming-zeroshot`).

## Strategic decisions

1. **Use the latest checkpoint.** `nvidia/nemotron-speech-streaming-en-0.6b`,
   `main` branch (updated 2026-03-12, "trained on larger corpora"). The paper
   used the older `nemotron-speech-streaming-jan2026` branch; we prioritise the
   best available model over bit-exact paper reproduction.
2. **Pipeline is fixed by the fine-tuning requirement.** Clean English 8.20% ->
   dysarthric unadapted is ~30-60%. Fine-tuning happens in the NeMo/PyTorch
   domain, so the path is: NeMo checkpoint -> fine-tune on SAP -> export ONNX ->
   quantize -> stream on CPU.
3. **Microsoft's Foundry Local int4 model is a reference ceiling, not a
   submission.** It is an un-adapted clean-English artifact; useful only as a
   sanity check / upper bound.

### De-scoped (vs the full paper blueprint)

- **k-quant reimplementation** — buys only 0.26% WER over plain int4-RTN;
  dysarthric fine-tuning loss dwarfs it and we have 7x RTF headroom. Use stock
  ONNX Runtime int8-dynamic / int4-RTN instead.
- **Three-graph joiner split** — ~5-10% RTFx; irrelevant at 7x headroom.
- **`lokkju` int4 export** — PolyForm Shield license risk; avoid.

## Artifacts

| Artifact | Use | License |
|---|---|---|
| `nvidia/nemotron-speech-streaming-en-0.6b` (`main`) | Base checkpoint, fine-tuning input | NVIDIA Open Model License |
| `danielbodart/nemotron-speech-600m-onnx` | Reproducible NeMo->ONNX export pipeline + zero-shot int8 baseline | export script published |
| Foundry Local `nemotron-speech-streaming-en-0.6b` | Reference ceiling only | NVIDIA OML / MS SDK |
| `run_onnx_streaming_latency.py` (in repo) | ONNX streaming harness — already written for the danielbodart layout | — |

## Phases

Every evaluation run is logged via `track2_starting_kit/experiments/experiments.py`
and committed (see `track2_starting_kit/experiments/README.md`).

### Phase 0 — Reference baseline (~1 day)
- Stand up the danielbodart **int8-dynamic** ONNX export and run it through
  `run_onnx_streaming_latency.py` on `Dev_streaming.csv`.
- Deliver one clean number: ONNX-int8 zero-shot CER + **RTF measured on the
  eval-class CPU**. Confirms the RTF gate is cleared before any further work.
- Optional: pull the Foundry Local int4 model as an upper-bound reference.

### Phase 1 — Zero-shot accuracy map (~1 day)
- Compare int8 ONNX CER against the logged NeMo zero-shot 17.81%.
- Confirm quantization does not collapse accuracy. Choose int8-dynamic vs
  int4-RTN as the deployment quant.

### Phase 2 — Fine-tune on SAP dysarthric speech (multi-day, GPU)
- Fine-tune the NeMo cache-aware FastConformer on the 742h SAP training set
  using NeMo's cache-aware streaming fine-tune recipe. This is the dominant
  accuracy lever.

### Phase 3 — Export + quantize the fine-tuned model (~2 days)
- Export fine-tuned NeMo -> ONNX via danielbodart's `nemo_export_onnx.py`.
- Quantize the encoder (int8-dynamic and/or int4-RTN via `MatMul4BitsQuantizer`).
- Parity check: quantized vs FP32 within ~0.5 CER on ~20 utterances.

### Phase 4 — Streaming wrapper + Pareto tuning (~2 days)
- Wrap in the SAPC2 5-method `Model` class: buffer 100 ms competition chunks
  into the model chunk, feed encoder caches, RNN-T greedy decode.
- Sweep `att_context_size` (`[70,13/6/1/0]` = 1120/560/160/80 ms chunks) for
  the accuracy/latency Pareto. Log CER, TTFT P50, TTLT P50, RTF per config.

### Phase 5 — Submission (~1 day)
- Pick the Pareto-winning config. Verify the submission zip **size limit**
  (int8 ~876 MB, int4 ~0.7 GB). Package, validate with `local_decode.py`.

## Risks

- **RTF on the actual eval CPU** — verify in Phase 0 before investing further.
- **Submission size limit** — verify before Phase 5; may force int4.
- **Cache-aware streaming is batch-size sensitive** (NeMo #12840, ~+/-1% WER);
  evaluate at batch_size=1 to stay comparable.
- **Mel front-end** must be Slaney-normalised; validate against danielbodart's
  `filterbank.bin`.

## Sources

- Paper: https://arxiv.org/abs/2604.14493
- Base model: https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- ONNX export pipeline: https://github.com/danielbodart/nemotron-speech-600m-onnx
- Streaming WER evaluator: https://github.com/nenad1002/open_asr_leaderboard
- Foundry Local release: https://devblogs.microsoft.com/foundry/foundry-local-v1-1/
