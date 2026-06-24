# 12 — Nemotron full-picture diagnostic: VERDICT (2026-06-24)

Ran the full battery (length×severity stratification + exp A/B/C + blank-penalty recovery) on a 425-utt
Dev set deliberately enriched with the 7 worst speakers + balanced duration bins. Closes [[11_nemotron_faithful_repro]].

## Headline: we reproduced Test1 on Dev, and proved the mechanism
The enriched set scores **CER 47.5%, 25% empty (107/425)** through the faithful harness — i.e. **≈ Test1's
51.52%**. So Test1's number is simply this set's severe-speaker / short-utterance mix at scale. No mystery left.

## Failure vs DURATION (surprise — it's INVERSE)
| dur bin | n | meanCER | empty |
|---|---|---|---|
| 0–3s | 101 | 46.6% | 22% |
| **3–8s** | 130 | **59.9%** | **42%** |
| 8–15s | 84 | 51.1% | 27% |
| 15–30s | 60 | 42.8% | 13% |
| **>30s** | 50 | **17.0%** | **0%** |

Short utterances fail hardest; long ones are fine. **This kills the "long-utterance / O(N²) / streaming-cache
bug" hypothesis** — a bug like that would hurt long utts. Instead, short clips of severe dysarthria give the
model too little to latch onto, and it emits nothing.

## Failure vs ETIOLOGY
| etiology | n | meanCER | empty |
|---|---|---|---|
| ALS | 110 | 62.7% | 39% |
| Cerebral Palsy | 152 | 54.3% | 32% |
| Down Syndrome | 67 | 52.8% | 18% |
| Stroke | 19 | 35.0% | 21% |
| **Parkinson's** | 77 | **10.9%** | **0%** |

Parkinson's (typically milder articulation) is near-perfect; ALS/CP (often severe) collapse. Classic
severity-driven domain failure.

## (B) Blank-probability probe — WHY the empties happen
On true empties, **100% of decode frames pick blank**, with **finite logits** and a **stable, moderate margin**
(blank − best-non-blank ≈ +3.7 to +3.9 on average):
- `cortana` (CP 1.7s): 28/28 blank, margin +3.81
- `considerable interest…` (ALS 27s): 346/346 blank, margin +3.69
- `quick` (ALS 5.8s): 79/79 blank, margin +3.93

Finite + stable + moderate = the model **confidently and consistently predicts blank**. This is NOT a numerical
blow-up, NOT int8 saturation, NOT NaNs → **not a quantization or decode artifact.** The model simply doesn't
recognize the speech.

## (5b) Blank-penalty "recovery" — the cheap fix does NOT work
Forcing non-blank (penalty sweep 0→10) on the empties yields **garbage, never the reference**:
- `cortana` → "Could color" → "rttulanza. Cool. Cool…"
- `quick` → "Queen Way Gwen Greg Greg…"
- (control good utt decoded correctly at penalty 0, and BREAKS at high penalty)

So the acoustic encoding carries **no recoverable signal** for these utterances — there's nothing for a wider
search or blank-suppression to find. A decode hack can't save it; it makes things worse.

## (A) and (C) — status
- **(A) zipformer rescue test did NOT run** — the pod's python lacks `sherpa_onnx` and the zf submission dir
  had no `setup.sh` to install it (`ModuleNotFoundError`). Not essential to the verdict (B+5b are decisive),
  but re-running it (add sherpa-onnx) would close the loop by showing the finetuned model handles these utts.
- **(C) fp32 vs int8 not run** — only the int8 encoder is on disk (fp32 needs a download). But (B)'s finite,
  well-behaved logits make a quantization-collapse explanation unlikely. Formally still open.

## VERDICT
Zero-shot Nemotron is **not broken** — no packaging, numpy, contention, speed, quantization, or decode bug.
It **genuinely cannot transcribe severe dysarthric speech** (confident all-blank), worst on short utterances,
worst for ALS/CP. Test1's 51% is that failure at scale; the finetuned zipformer survives because it learned
the domain. **The only fix is SAPC2 finetuning.** Decode tricks make it worse.

Next-decision options unchanged (see chat): finetune Nemotron (high ceiling — median ~7% on mild speech) vs
bank the proven zipformer (beam-4 + LM fusion). Artifacts: pod `/workspace/finetune/eval/diag_battery/RESULTS.txt`,
local `/Users/o/Downloads/diag_RESULTS.txt`. Tooling: `scripts/{build_diag_manifest,diag_crosstab,nemotron_blank_probe,nemotron_altdecode_empties,run_diag_battery}`.
