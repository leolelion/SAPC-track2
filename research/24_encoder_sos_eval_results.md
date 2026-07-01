# 24 - Encoder-only SOS deploy eval: severe + representative CER (2026-06-27)

Ran the chosen deploy candidate, **encoder-only Nemotron + SOS fix**, through the real
`local_decode.py` batch path on the RunPod H200. The submission dir was rebuilt from
`/workspace/finetune/nemo_submission`, the encoder-only `[70,1]` ONNX export, `CHUNK_NEW=16`,
and `_last_token=[[BLANK_ID]]`.

The run used a one-row streaming smoke manifest with `--streaming-interval 0` because this gate
measured CER and speaker-block bootstrap CI; latency was already measured in `research/23`.

Logs/artifacts:
- local log: `/Users/o/Downloads/encoder_sos_eval.log`
- local JSON: `/Users/o/Downloads/enc_sos_devdiag_cer_bootstrap.json`
- local JSON: `/Users/o/Downloads/enc_sos_devrep500_cer_bootstrap.json`
- pod log: `/workspace/finetune/nemo_ft/encoder_sos_eval.log`
- pod status after run: stopped / `EXITED`

## Results

| set | n | speakers | mean CER | speaker-block 95% CI | empty | `"Wh "` |
|---|---:|---:|---:|---:|---:|---:|
| `Dev_diag.csv` severe-enriched | 425 | 103 | 24.49% | 16.06-33.16% | 30 | 0 |
| representative Dev sample, seed 23 | 500 | 119 | 10.49% | 7.92-13.55% | 6 | 0 |

## Etiology breakdown

### `Dev_diag.csv`

| etiology | n | CER |
|---|---:|---:|
| ALS | 110 | 33.95% |
| Cerebral Palsy | 152 | 22.49% |
| Down Syndrome | 67 | 33.85% |
| Parkinson's Disease | 77 | 6.20% |
| Stroke | 19 | 26.75% |

### Representative Dev sample

| etiology | n | CER |
|---|---:|---:|
| ALS | 103 | 4.60% |
| Cerebral Palsy | 123 | 19.11% |
| Down Syndrome | 82 | 17.24% |
| Parkinson's Disease | 156 | 3.72% |
| Stroke | 36 | 11.80% |

## Interpretation

The SOS fix holds on larger dysarthric evals: the stream-start `"Wh "` artifact is gone on both
sets (`0` prefixes), so the harness bug is no longer contaminating CER.

The representative sample is the main submission-relevance signal so far: **10.49% CER with a
speaker-block 95% CI of 7.92-13.55%**. That is comfortably below the current Zipformer Test1
reference of 23.44%, though it is still a Dev sample rather than hidden Test1.

The severe-enriched diagnostic set remains hard: **24.49% CER**, with wide speaker-block CI and
30 empty outputs. It is deliberately enriched for worst speakers and should not be treated as the
expected hidden-test mean, but it identifies the residual failure mode: severe ALS and Down
Syndrome remain the largest error buckets.

## Decision

Continue with **encoder-only + SOS fix** as the submission candidate.

## Next steps

1. Patch/package the actual offline submission with the SOS fix.
2. Run a clean offline package gate: no network, no NeMo, bundled ORT/local-mel only.
3. Run the faithful Dev gate on the packaged artifact, not just the pod-side deploy dir.
4. Investigate remaining TTFT gap: encoder-only p50 was 1.15 s in `research/23`.
5. Quantize encoder int8 only, keep decoder/joint FP32, then rerun the representative and
   severe CER gates to measure accuracy delta.
6. For a future v2 finetune, prioritize severe ALS/Down Syndrome empties and train longer with
   clean tail checkpoint averaging plus multi-lookahead training.
