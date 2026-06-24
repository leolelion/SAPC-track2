# 11 — Nemotron Test1 51% — DEFINITIVE diagnosis via the faithful harness (2026-06-24)

Closes the investigation opened in [[10_nemotron_verdict]]. The Test1 51.52% was **not** a packaging,
numpy-ABI, 20-worker-contention, or speed bug. The submission code is correct. Zero-shot Nemotron simply
**fails on severe dysarthric speakers** it was never trained on.

## How we finally got it right
Two failed Nemotron uploads returned **byte-identical** Test1 (CER 51.5168 / WER 58.9851; only latency
drifted) → deterministic → contention falsified (truncation would vary run-to-run). So we ran the
**organizers' REAL `local_decode.py`** (int16/32768 scaling, 1600-sample chunks, the actual `Model`
interface) on Dev — the faithful gate we had never actually run (prior "Dev 24.96%" used a hand-rolled
proxy). See [[validate-against-real-harness]].

## Results (faithful harness + official EnglishTextNormalizer, two-ref char-CER)
| set | meanCER | median | empties | >80% near-fail |
|---|---|---|---|---|
| Dev_100 (easy subset) | 5.47% | 0.0 | 0 | 0 |
| **Dev_rand300 (representative)** | **19.76%** | **7.1%** | **17 (5.7%)** | **22 (7.3%)** |

Nemotron zero-shot is **bimodal**: excellent on mild/clear speech (median 7.1%), catastrophic on a tail.
The empties span **1.66–18.71s** (not a duration/chunk bug) and **cluster on severe-dysarthria speakers**
(one ALS speaker = 7 of 17 empties; rest severe CP/DS/Stroke/PD). The model emits **all-blank** when the
acoustics are far from its clean-English prior.

## Why Test1 = 51% while the zipformer holds
| | Dev | Test1 | Δ |
|---|---|---|---|
| Zipformer A1 (SAPC2-finetuned) | ~21.6% | 23.44% | +1.8 |
| Nemotron (zero-shot) | ~24.96% | 51.52% | +26.5 |

Both see the same Test1. The finetuned model barely degrades; the zero-shot model doubles. Test1 just has
a higher proportion of the severe speech that triggers Nemotron's failure tail.

## Verdict
- Retire the packaging/speed/contention thread — all non-issues. The submission path is validated correct.
- The **only** path to make Nemotron competitive is **SAPC2 finetuning** (the adaptation that took the
  zipformer 36%→21.6%). The median-7.1% says the ceiling is high if adapted — a genuine bet, but the same
  large GPU effort, not a decode tweak. A blank-suppression / min-token decode hack might shave the empties
  marginally; it won't close the domain gap.
- Meanwhile the proven zipformer has bankable wins: beam-4 (−2.6 CER, [[09_decoding_results]]) then LM fusion.

## Process lesson
We burned ~a week and 2 submissions on exotic failure theories for code that was never broken — because we
validated on Dev with a proxy and never (a) ran the real harness on a *representative* set, nor (b) sanity-
checked the new model's Dev→Test1 gap against a known-good model. Both are now house rules
([[validate-against-real-harness]], `CLAUDE.md`). Artifacts: `scripts/nemotron_repro_{run.sh,analyze.py}`,
local `Dev_rand300_repro.csv` + `FINDINGS.txt`, pod `/workspace/finetune/eval/nemotron_faithful_repro/`.
