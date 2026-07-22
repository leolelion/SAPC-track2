# Investigation — Why did Nemotron (10× params) not beat the finetuned zipformer?

> Consolidates evidence scattered across `research/10–41` + memory `int8-submission-status`
> into one facts-vs-theories document. Follows the global investigation discipline:
> separate FACTS (measured through the faithful harness) from THEORIES (with live status),
> maintain competing hypotheses, don't chase one.
>
> **Bottom line up front:** "10× params" is the wrong frame. On this task the deciding
> variables are (a) domain adaptation, (b) distribution match to Test1, and — the sharpest,
> most under-investigated one — (c) a **train→streaming-export gap**: a finetuned Nemotron
> is *better than the zipformer OFFLINE* (18.46% transcribe proxy) but collapses under
> cache-aware chunked streaming (28.19% faithful). It doesn't lose on capacity; it loses on
> the regime the challenge actually scores.

---

## The two models being compared

| | Zipformer (A1) | Nemotron (v1, best artifact) |
|---|---|---|
| Params | ~66–70M | ~618M (~10×) |
| Pretraining | LibriSpeech (streaming) | large clean-English |
| SAP adaptation | full finetune (16 ep, sp+SpecAug) | encoder-only FT, joint frozen |
| Decoding | greedy → **beam-4** | transducer, cache-aware streaming |
| **Test1 CER** | **23.44%** (greedy); ~21% expected (beam-4) | **27.97%** (int8 `_t1`) |
| Faithful Dev_diag (severe-enriched) | ~29.1% (beam-4) | 23.58% |

Note the Dev_diag numbers *invert* the Test1 order (Nemotron 23.58 < zipformer 29.1) yet
Nemotron *loses* on Test1 (27.97 > 23.44). That gap between "looks better on our Dev proxy"
and "worse on Test1" is the Dev→Test transfer failure (see FACT-9, H5).

---

## FACTS (measured through the faithful harness / official scorer)

- **F1 — Zero-shot Nemotron genuinely cannot transcribe severe dysarthric speech.** On the
  severe-enriched Dev_diag-425 it scores 47.5% CER / 25% empty, reproducing Test1's 51.5%.
  `research/12`.
- **F2 — Failure is severity-driven, not size-driven.** Parkinson's (mild articulation) is
  near-perfect (10.9% CER, 0% empty); ALS/CP (often severe) collapse (62.7% / 54.3%, ~35% empty).
  `research/12`.
- **F3 — Failure is INVERSE to duration.** Short clips fail hardest (3–8 s: 59.9% CER, 42%
  empty); >30 s clips are fine (17.0%, 0% empty). This **falsifies** any long-context / O(N²) /
  streaming-cache-blowup bug — such a bug would hurt long utterances. `research/12`.
- **F4 — Empties are CONFIDENT blank, not numerical failure.** On true empties, 100% of decode
  frames pick blank with *finite, stable* logits (blank − best-non-blank margin ≈ +3.7–3.9).
  Not NaNs, not int8 saturation, not a decode artifact — the model confidently predicts "nothing
  here." `research/12`.
- **F5 — Not quantization.** int8 ≈ fp32 (official Dev-500 8.76% vs 8.64%). `int8-submission-status`.
- **F6 — Domain adaptation rescues most Nemotron failures.** Of 140 utts where zero-shot Nemotron
  catastrophically fails (>0.8 CER or empty), the *finetuned zipformer* rescues 74 (53%) to <0.5
  CER; ~33 partially; only 33 (24%) fail for both (the genuine severe-tail floor). → the failures
  are predominantly **domain**, and a small *adapted* model fixes them. `research/15` exp A.
- **F7 — Complementarity by regime.** On >30 s clips Nemotron *beats* the zipformer (17.0 vs 24.5)
  — long context suits its clean-English prior; on short severe clips the zipformer wins.
  `research/15`.
- **F8 — THE KEY FACT: a train→streaming-export gap.** The PEFT-finetuned Nemotron scores **18.46%
  via NeMo's in-script `transcribe` (offline)** but **28.19% through faithful cache-aware chunked
  streaming export**. Same weights, same audio — a ~10-point collapse caused purely by the
  streaming regime. v1 shows the same internal-vs-faithful divergence. `research/41`, `research/39`.
- **F9 — Nemotron transfers Dev→Test far worse than the zipformer.** Dev→Test slope: Nemotron
  **+18** (Dev_diag 23.58 → Test1 27.97 is consistent with a severe-heavy Test1) vs zipformer A1
  **+1.8** (Dev 21.6 → Test1 23.44). `research/37`, `research/40`.
- **F10 — Every Nemotron encoder-rescue arm FAILED the faithful gate vs v1.** Scoreboard below.

### Scoreboard — attempts to move the Nemotron severe tail (all vs v1 = 23.58% Dev_diag)
| attempt | faithful Dev_diag | verdict | doc |
|---|---:|---|---|
| Cheap inference (blank-penalty, gain-norm) | — | FAILED (garbage / ~net-neutral) | research/37 |
| Arm A (encoder-only + aug + severity, 50k) | 27.35% | FAILED (worse; empties unmoved) | research/39 |
| **PEFT adapters (LoRA-style, base frozen)** | **28.19%** | **FAILED (worst; empties UP)** | research/41 |
| Arm B (joint-unfreeze) | untested | contraindicated (adds capacity to overfit problem) | research/40 |
| **v1 (full encoder FT, no aug)** | **23.58%** | **best Nemotron artifact — still loses to A1** | research/20/24 |

---

## THEORIES / competing hypotheses (with current status)

| # | Hypothesis | Status | Discriminating evidence |
|---|---|---|---|
| H1 | Capacity/params insufficient | **FALSIFIED** | 8.76% rep-Dev; 18.46% offline — capacity is there (F5, F8) |
| H2 | Quantization (int8) collapse | **FALSIFIED** | finite logits (F4); int8≈fp32 (F5) |
| H3 | Streaming/long-context bug | **FALSIFIED** | short utts fail worst, not long (F3) |
| H4 | Domain shift / severe-tail failure | **CONFIRMED** | severe-enriched Dev ≈ Test1; PD fine, ALS/CP collapse (F1,F2,F9) |
| H5 | Overfits small corpus; PEFT (fewer params) fixes it | **FALSIFIED** | PEFT was the WORST arm (28.19%), not better (F10, research/41) |
| H6 | Frozen-joint blank propensity caps empties | **PARTIAL / disputed** | research/39 supports (encoder-aug didn't move empties); research/40 contradicts sole-cause (gain-norm recovered some with joint frozen); Arm B never tested |
| **H7** | **Train→streaming-export gap is the binding cause** | **TESTED & LARGELY FALSIFIED (2026-07-22, research/45)** | Base A/B/C @ [70,1] Dev_diag: transcribe 43.4 ≈ NeMo-native stream 43.8 ≈ our-export 43.4 → pipeline is FAITHFUL, no export/streaming collapse. The finetuned 18→28 (F8) was weights/training-specific (PEFT not surviving streaming cache-norm) or a confounded compare — NOT a fixable export bug. Finetuned ckpt lost → cannot re-measure |

**Read of the field:** H4 is confirmed but is a *symptom framing*. H1/H2/H3 are cleanly dead.
H5 (the independent reviewer's thesis) is dead — PEFT made it worse. H6 is stuck (needs the
untested, contraindicated Arm B to resolve). **H7 is the one that actually explains the shape of
all the evidence** and is the only lever no arm has targeted: the model can transcribe this speech
offline; it's the *cache-aware chunked-streaming export* that destroys the gains. Encoder
finetuning (Arm A, PEFT) all improved the offline proxy and then failed to survive export.

---

## Answer to "why didn't 10× params win?"

1. **Params only help in-distribution.** Scaling laws are in-distribution statements. Severe
   dysarthric streaming speech is maximally far from clean-English pretraining, so the capacity
   advantage doesn't cash out (H1 falsified, H4 confirmed).
2. **A small *adapted* model beats a large *unadapted* one under domain shift.** The zipformer was
   fully SAP-finetuned; Nemotron's finetune froze the joint and never survived the export path.
   F6 shows adaptation — not size — is the discriminator.
3. **The challenge scores the streaming regime, where Nemotron's advantage evaporates.** Offline,
   a finetuned Nemotron (18.46%) *beats* the zipformer; under 100 ms cache-aware chunks it collapses
   (28.19%). This is F8/H7 — the sharpest and most actionable finding.
4. **Poor Dev→Test transfer amplified the loss.** +18 vs +1.8 slope: its Dev lead was on
   easy/moderate speakers; Test1 is severe-heavy.

---

## Topics to understand (to reason about this, not just pattern-match)

- **OOD generalization & scaling-law limits** — why capacity is an in-distribution promise.
- **RNN-Transducer internals** — encoder vs prediction-net vs joint; the **blank** symbol and why a
  transducer over-emits blank on OOD/low-energy audio. (ICASSP-2021 "RNN-T Fail to Generalize to
  OOD Audio"; FastEmit; multi-blank transducers.)
- **Internal LM (ILM) / density-ratio** — decode-time blank-bias correction; explains why a blunt
  blank-penalty produced garbage (research/12 §5b).
- **PEFT (LoRA/adapters/LHUC)** — and, per F10, why "fewer trainable params" is *not* a guaranteed
  fix; it can fail to survive a streaming export even when it helps offline.
- **Cache-aware / chunked streaming vs offline attention** — the mechanism behind H7. This is where
  the next investigation should focus if Nemotron is pursued further.
- **Feature normalization** (`normalize=NA`, per-utterance RMS/CMVN) — quiet dysarthric audio
  underflowing pretrained frontends into blanks.
- **Evaluation methodology** — representative vs severe-enriched Dev; **Dev→Test slope** as an
  overfit diagnostic; which validation set predicts Test (rep-Dev 8.76% was best-case;
  severe-enriched ~25% was the true Test predictor).

---

## Status of "the two controlled comparisons" (both ALREADY RUN — do not re-run)

1. **PEFT arm** → RUN as armC, `research/41` (2026-06-30). **FAILED**, 28.19% (worst arm). H5 falsified.
2. **Zipformer rescue test** → RUN as `research/15` exp A (2026-06-25). **DONE**, 53% of Nemotron
   catastrophic failures rescued → domain-not-size confirmed (F6).

Re-running either reproduces closed results at GPU cost → not justified.

## The ONE genuinely-unexplored lever (if Nemotron is pursued at all)
Per `research/41`'s own recommendation: any further Nemotron GPU should target **the train→streaming
-export gap itself (H7), or a joint/decoder-side change — NOT more encoder adaptation** — and only as
a fresh, explicitly-approved bet. Concretely: diagnose *why* the offline→streaming export loses ~10
points (cache normalization vs full-sequence forward; att_context regime at export; chunk size),
rather than training another encoder variant that will die on the same export.

**Evidence-backed default (research/39 §b, /40 option 3, /41): STOP the Nemotron encoder-rescue.**
Three independent approaches all failed on the faithful harness. Bank A1 (Test1 23.44%) and ship the
gated **beam-4** upgrade (~21% expected). Doc 14 §10's "zipformer track is higher-EV" call is now
reinforced by Test1 reality.
