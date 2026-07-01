# 40 — Independent review of the big picture; v2 plan PIVOT (2026-06-30)

> An independent senior-ASR reviewer (read-only repo audit + literature verification, no GPU) audited
> our assumptions, big-picture thinking, and the original literature review (research/01,02). Its critique
> holds up against our OWN documents and the literature. This note records the verdicts, the one thing we
> got wrong, the literature we missed, and the corrected plan. Supersedes the Arm-B-first thesis in
> research/38 §"Open decision" and the "joint-bound, confirmed" framing in research/39 / memory v2-finetune-plan.

## TL;DR — what changed
1. **Arm B (joint-unfreeze) is contraindicated, not the next move.** Our headline disease is overfitting /
   poor Dev→Test transfer (our own research/37 §2: the encoder "overfits its representations to the
   train/dev speaker pool"; Dev→Test +18 vs A1 +1.8). The literature fix for overfitting on a small atypical
   corpus is **parameter-EFFICIENT** adaptation (PEFT: adapters / LoRA / LHUC), i.e. FEWER trainable params.
   Arm B ADDS a second trainable module → pushes capacity the wrong way on the exact axis we are losing.
2. **The empties are (at least co-)caused by ENCODER energy-sensitivity, not solely a frozen-joint cap.**
   Our research/37 §5 recovered empties — "8/31, some PERFECTLY" — with the joint STILL FROZEN, using only
   an input gain change. That directly weakens "the frozen joint caps empties regardless of the encoder."
   Downgrade the attribution from "joint-bound, confirmed" to "≥3 live hypotheses": (a) encoder energy
   robustness, (b) decode-time internal-LM (ILM) blank-bias correction, (c) joint adaptation.
3. **"Augmentation hurts atypical speech" is an over-generalization from one confounded run.** Arm A bundled
   aggressive gain (−20/+10 dB) + speed-perturb + 2× severity on a 50k subset and regressed everywhere; the
   literature robustly reports MILD aug HELPS dysarthric ASR. (Note: Arm A's aug WAS verified applied —
   `_pipeline:[[1.0,GainPerturbation],[1.0,SpeedPerturbation]]` — so "empties unchanged" is a real but
   confounded data point, not a no-op. It does NOT prove encoder-side aug can't help; the clean mild arm was
   never run.)

## Per-claim verdicts (reviewer, with our cross-check)
| # | Claim | Verdict | Basis |
|---|---|---|---|
| 1 | encoder-only FT is the right adaptation | **CONTRADICTED** | our own research/01 §1 `[V-lit]` "Full FT > freezing decoder/joint" + research/14 §5 (Takahashi Parakeet: freezing joint HURT 6.31 vs 6.26; SAP winner fully unfrozen). v1 shipped the config our lit review warned against. But full-unfreeze overfits → answer is **PEFT**, not full unfreeze. |
| 2 | empties are frozen-joint-bound, encoder can't fix | **PARTIAL / attribution CONTRADICTED** | phenomenon real (RNN-T blank/deletion over-emission on OOD/low-energy audio is documented); but §5 gain-norm perfect-recovery with frozen joint + blank-suppress→garbage both point ENCODER-side. "Confirmed" rests on one confounded run. |
| 3 | Arm B vs stop | **see Plan below** | Arm B as written = negative-to-flat EV (wrong module + adds overfit capacity). |
| 4 | aug hurting is surprising | **YES; over-generalized** | lit: speed-perturb ~9.3% rel (UASpeech), two-stage aug +16% abs, SegAug −45% rel RNN-T deletions. Our result = bad config artifact. |
| 5 | big model overfits small atypical corpus, transfers worse | **VERIFIED (most important, under-exploited)** | CUHK: "direct fine-tuning of a large number of parameters on limited impaired speech rapidly leads to overfitting and poor generalisation"; fix = PEFT. Our +18 vs +1.8 is the textbook symptom; full-encoder FT (~0.5B trainable) is the textbook cause. |

## The one thing we were blind to
We were about to spend GPU **adding** trainable capacity (joint-unfreeze) to fix a problem whose root cause
is (a) encoder energy-sensitivity and (b) overfitting/poor transfer — both of which point the OPPOSITE way.
"Frozen-joint blank propensity" became a load-bearing story across docs 36→37→38→39 and each new null was
read as vindicating it, while the contradicting evidence (perfect gain-norm recovery; blank-suppress→garbage;
Arm A confound) sat in the same docs unintegrated. Classic confirmation bias.

## Literature we missed (2021–2025) — to fold into research/01,02
- **PEFT for dysarthric/atypical ASR (the central gap):** Adapter Fusion + Householder (arXiv 2306.07090);
  Structured Speaker-Deficiency Adaptation of foundation models (2412.18832, 2024); SSVD structured-SVD PEFT
  under domain shift (2509.02830, 2025); LHUC/f-LHUC (CUHK 2302.14564, 2407.13782). Note: research/14 §9 ALREADY
  flagged a SAP winner used AdaLoRA — and v1 then used none of it.
- **RNN-T emission / blank control:** ICASSP-2021 "RNN-T Fail to Generalize to OOD Audio: Causes and Solutions";
  Interspeech-2022 sparse self-attention (2108.10752); **FastEmit** (2010.11148) as a blank-reduction LOSS (we
  cited it only for latency); **Multi-blank transducers** (2211.03541); **ILM estimation/subtraction (ILME /
  density-ratio)** as a decode-time blank-bias fix we never tried (we only tried a crude blank penalty).
- **Augmentation that helps (counter to our Arm A reading):** SegAug (2502.14685, targets RNN-T deletions,
  −45% rel); two-stage aug (Bhat, Interspeech 2022 / S0010482525003051); TTS-dysarthric synthesis (2406.08568).

## CORRECTED plan (ranked by EV — replaces research/38's Arm-B-first ordering)
1. **FREE / no-GPU-to-write: principled Pass-1 energy normalization.** Pass 1 (accuracy) is BATCH — whole
   waveform available — so per-utterance waveform RMS-normalization to the training energy mean is legal there
   (keeps `normalize=NA` features; just returns energy to the trained range; does NOT touch the streaming
   contract). Reviewer's #1. HONEST caveat from our own §5: conditional peak-norm netted only ~+0.2 pt and
   TRADED etiologies (ALS −1.4 / DS +1.7) — so EV is modest, not a clear win; a uniform RMS-to-mean (vs our
   conditional boost) may behave differently. Also try **ILM-subtraction at decode** instead of the blunt
   blank penalty. Cheap to write locally; validation still needs one pod session.
2. **If GPU approved — run the clean arm we skipped, as PEFT not full/joint:** encoder **LoRA/adapter**
   (± small joint adapter), **mild** gain aug (±6–10 dB, NOT −20/+10), **full data**, **NO** severity
   oversampling, **NO** speed-perturb in the same run. This respects the overfit diagnosis (PEFT regularizes),
   tests energy-robustness cleanly, and is the literature-correct adaptation for a small atypical corpus that
   transfers poorly. LR ~1e-4–3e-4 on adapters, base frozen; short epochs (Phase 0: CER flat by ep3). Gate on
   Dev_diag empty-rate AND a clean-speech forgetting probe AND an honest Dev→Test projection vs A1.
3. **Banking A1 + pivoting to the zipformer track is a legitimate stop.** We're behind A1 on Test1 (27.97 vs
   23.44); two GPU arms produced null/negative; Dev→Test is unreliable so even a green Dev gate may not beat
   A1. If items 1–2 don't clear A1 cheaply, STOP. research/14 §10's "zipformer higher-EV near-term" call is
   vindicated by Test1.

Severity oversampling stays a guarded ride-along AT MOST (research/37 §6: the lit gains are closed-speaker
TORGO/UASpeech, the opposite of SAPC2 held-out Test1).

## Confidence / open risks (be skeptical of the reviewer too)
- The encoder-vs-joint attribution rests primarily on TWO in-house single-run results (§5 gain-norm recovery;
  blank-suppress→garbage). Strong but not settled — the clean mild-gain PEFT arm would confirm.
- PEFT-beats-full-FT-for-overfit and aug-helps are MULTI-paper, high confidence.
- Our §5 already shows plain gain-norm is ~net-neutral, so do not oversell item 1's EV.

## Decision owner = Q (GPU spend, cost gate)
The next step that "runs" anything requires a pod (cost gate → explicit Q approval + chosen arm). Local/free
work (doc updates here; optionally writing the Pass-1 RMS-norm model.py variant + a LoRA finetune script) can
proceed without GPU. Held for Q's pick: (1) cheap RMS-norm + PEFT arm, (2) PEFT arm only, (3) stop & bank A1.
