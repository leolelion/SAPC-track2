# 09 — Decoding experiment results (E1: beam search)

Experiment E1 from [[08_decoding_lm_experiment_plan]], run 2026-06-21 on the avg-5 chunk-16 zipformer
(`finetune/onnx/a1`), 2k speaker-balanced Dev ruler (official sclite min-two-refs). Decoding-only change,
no retraining. Results persisted: pod `/workspace/finetune/eval/e1_results/`.

## E1 — Beam search (accuracy, 2k ruler)
| decoding | CER | WER | ΔCER vs greedy |
|---|---|---|---|
| greedy (current submission) | 21.61% | 27.03% | — |
| modified_beam_search, paths=4 | **19.01%** | 24.77% | **−2.60** |
| modified_beam_search, paths=8 | **18.33%** | 24.35% | **−3.28** |

Beam search gives a **large** gain (12–15% rel) — bigger than the typical 0.5–2% — because dysarthric
audio makes the acoustic model uncertain (flat per-step distributions), so greedy's locally-greedy picks
are fragile and a wider search recovers a lot. Did not fully saturate at 8.

## Latency (beam-4, Dev_streaming, real-time-paced, CPU)
| | TTFT p50 | TTLT p50 | mean |
|---|---|---|---|
| greedy | 1234 ms | 100 ms | 643 ms |
| **beam-4** | **1208 ms** | **101 ms** | **654 ms** |

**Beam-4 latency ≈ greedy** (TTFT slightly lower; kept real-time, no lag). TTFT is chunk-accumulation-bound
(~640 ms), not search-width-bound, and beam-4's per-chunk decode stays under the 100 ms budget with 4 threads.

## Verdict
**Beam-4 is a strict win: −2.6 CER pts at identical latency, same size, zero retraining → ship it.**
Upgrades the live #1 submission (greedy → modified_beam_search paths=4 in model.py / config.yaml).

## Next
- Quick: measure **beam-8 latency** (CER 18.33% is better; 8× compute — confirm it still keeps real-time / TTFT).
- Package the beam submission upgrade, offline-validate ([[submission-offline-packaging]]), upload.
- **E2 — RNN-LM shallow fusion** stacks on top of beam (sherpa online `lm`+`lm_scale`, train small LSTM LM on
  SAPC2 text via icefall rnn_lm). Expected further 5–15% rel.
