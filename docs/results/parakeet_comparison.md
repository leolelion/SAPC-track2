# Parakeet RNNT / TDT vs streaming — SAPC2 Track 2 benchmark results

_Measured on the pod (H100 for offline, CPU for streaming), 2026-07-14. Scored with the team's
dual-reference min-CER macro-mean method (`eval_dev100` normalize) — **NOT** the official sclite
clip@1 pipeline. Fine for ranking; final sign-off would use `evaluate.sh`._

## Headline
- **The current zipformer baseline WINS.** On the same Dev_streaming utterances it scores **17.25% CER,
  TTFT 1.05 s** — *streaming* — which beats the Parakeet **offline** ceilings (TDT 22.6%, RNNT 22.3%).
  The zipformer is dysarthric-fine-tuned; the Parakeets are zero-shot general English.
- **So the answer to the task's core question is: NO, zero-shot Parakeet-RNNT/TDT is NOT better than the
  current baseline for Track 2** — not on accuracy, not on latency. The dysarthric fine-tuning the
  baseline already has matters more than Parakeet's raw capacity.
- Parakeet TDT/RNNT offline are strong in absolute terms (16–22% CER) but **cannot stream natively**
  (full-context `att_context_size=[-1,-1]`), and a genuinely-streamable NVIDIA model zero-shot is poor
  (`fastconformer_streaming_multi [70,13]` = 38% CER / 2.1 s TTFT).
- **Conclusion:** replacing the zipformer with Parakeet only makes sense if **cache-aware fine-tuning on
  dysarthric data** pushes Parakeet's *streaming* CER well below the zipformer's 17.25% — an unproven,
  expensive bet, and the same NVIDIA-streaming path the prior Nemotron attempt already lost on.

## Results

### A. Offline accuracy ceiling — Dev_rand300 (300 utts)
| Model | Params | Mode | CER % | WER % | empty | offline s/utt (H100) |
|---|---|---|---|---|---|---|
| Parakeet-TDT-0.6b-v2 | 0.6B | offline | **16.64** | **29.37** | 2 | 0.043 |
| Parakeet-RNNT-0.6b | 0.6B | offline | 19.43 | 31.16 | 9 | 0.029 |

### B. Same-subset comparison — Dev_streaming (123 utts)
| Model | Params | Mode | CER % | WER % | empty | TTFT p50 | TTLT p50 |
|---|---|---|---|---|---|---|---|
| **Zipformer (current baseline, dysarthric-FT)** | ~70M | **streaming** | **17.25** | **23.45** | 1 | **1.05 s** | 0.071 s |
| Parakeet-TDT-0.6b-v2 | 0.6B | offline *ceiling* | 22.56 | 34.84 | 5 | n/a | n/a |
| Parakeet-RNNT-0.6b | 0.6B | offline *ceiling* | 22.32 | 32.46 | 7 | n/a | n/a |
| fastconformer_streaming_multi [70,13] | 0.11B | streaming (CPU) | 38.13 | 51.06 | 15 | 2.12 s | 0.105 s |

The baseline's *streaming* CER (17.25%) is below the Parakeet *offline* ceilings — the decisive result.

Streaming latency detail (streaming_multi [70,13]): TTFT p50 2.12 / p90 3.38 / p95 4.89 s;
TTLT p50 0.105 / p90 0.140 / p95 0.148 s.

Per-etiology CER (streaming_multi [70,13]): Parkinson's 20.2, Stroke 27.9, ALS 41.5, Down 45.7,
Cerebral Palsy 56.2 — dysarthria severity dominates, CP worst.

## Analysis (the task's four questions)
1. **Best accuracy?** Offline: TDT-0.6b-v2 on the easier subset (16.6%); ~tied with RNNT on the harder
   Dev_streaming (~22.5%). Streaming-mode accuracy is far worse (38%).
2. **Best latency?** streaming_multi finalizes fast (TTLT 0.105 s) but TTFT is high at [70,13] (2.12 s).
   Lower att_context ([70,1]→~160 ms, [70,0]→~80 ms lookahead) would cut TTFT sharply at further CER
   cost — the accuracy/latency Pareto knob (not yet measured).
3. **Pareto frontier?** On CER+latency, no zero-shot Parakeet/streaming point is competitive with the
   existing zipformer baseline yet (zipformer Dev_streaming number still to be pulled same-harness).
   The offline ceilings are a *bound*, not a deployable streaming point.
4. **Worth fine-tuning?** The offline ceiling (16–22%) vs zero-shot streaming (38%) gap is the prize.
   TDT is the accuracy leader but is full-context and NVIDIA-flagged buggy for chunked streaming;
   **the realistic streaming bet is cache-aware fine-tuning of a streamable FastConformer-RNNT on
   dysarthric data**, targeting the ~22% offline level at an acceptable TTFT.

## Key caveats
- Two different subsets (Dev_rand300 vs Dev_streaming); compare within a subset only.
- Dual-ref macro-mean scorer, not official sclite/clip@1.
- streaming_multi is a general-English model (0.11B), not dysarthria-tuned — its 38% is a zero-shot
  floor, and it's smaller than the 0.6B offline Parakeets.
- No zipformer baseline row on these exact subsets/harness yet — needed to judge "better than current".
- streaming run had 15/123 empty hyps (some utterances dropped entirely) — a streaming-robustness issue.

## Artifacts (on pod /workspace/bench_out/)
`{tdt06v2,rnnt06}_Dev_rand300.predict.csv`, `{tdt,rnnt}_Dev_streaming.predict.csv`,
`sm_Dev_streaming.predict.csv` + `sm_Dev_streaming.partial.json`, matching `*.score.json`.
Wrapper: `streaming_multi_sub/` (adapted from `parakeet_realtime`, patched `_rnnt_greedy_decode`
for NeMo 2.7.2 list return). Helpers: `/workspace/{bench_ceiling,score_predict}.py`.
