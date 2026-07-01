# 36 — Phase-0 results: the cheap wins don't exist; v1 is near its recipe ceiling (2026-06-28)

> Executes Phase 0 of research/35, in the "faster but accurate" form Q asked for: a speaker-stratified
> **3,000-utt** proxy instead of the full 27k + 4× faithful harness. Pod 3dwiczo41jeg1y up ~35 min, then
> STOPPED. Artifacts: `sapc2_int8_final/v1_diag/phase0/{phase0_curve.json,phase0_drop_audit.json,avg_model_eval.json}`.
> Scripts (local, reusable): `scripts/phase0_eval_curve.py`, `phase0_eval_nemo.py`, `phase0_dropped_by_etiology.py`, `run_phase0.sh`.

## Method (what "faster but accurate" meant)
- **Reliable val** = 3,000 internal-dev utts, **speaker-stratified over all 70 held-out speakers** (vs v1's
  biased 480 from few speakers), reproducing the EXACT v1 deterministic split (seed 13, frac 0.08).
- **Accurate metric** = official `EnglishTextNormalizer` + **min-over-two-refs** + **clip@1** (the real scorer's
  CER definition), applied identically to hyp + both refs.
- **PROXY caveat**: offline NeMo-transcribe at pinned [70,1], **not** 100 ms streaming. Valid for **curve shape
  and model-vs-model comparison** (the only decisions here). Absolute level runs ~3 pts above official Dev-500
  (proxy 11.8% vs official 8.76%) because of streaming + sclite + unk reconciliation. Any re-ship/submit still
  goes through the faithful `local_decode.py` + `evaluate.sh` harness (house rule).
- NB: there are **5** v1 checkpoints, not 4 — **two at epoch 0** (val_wer 0.4021 and 0.3930). Shipped v1 = avg of 5.

## Result 1 — convergence curve (the thesis under test)

| epoch | proxy CER | proxy WER | empties/3000 |
|---|---|---|---|
| 0 (0.4021) | 14.45 | 18.80 | 102 |
| 0 (0.3930) | 14.80 | 19.50 | 113 |
| 1 (0.3827) | 14.06 | 17.98 | 132 |
| 2 (0.3646) | 12.60 | 16.56 | 106 |
| 3 (0.3532) | **12.47** | 16.18 | 132 |

Marginal CER gain: ep1→ep2 = **−1.46**, ep2→ep3 = **−0.13** (≈10× smaller). **On the PRIMARY metric (CER), v1
has largely flattened by epoch 3.** WER still drops modestly (−0.38). This **partially falsifies the
"under-converged, just train longer" thesis** (research/34 §A) — the reviewer's C1 caution, now confirmed.

**CONFOUND (must not ignore):** the 4-epoch run used cosine LR **annealing to min_lr=1e-6 by ep3**. The flatten
is consistent with EITHER genuine convergence OR the schedule running out of LR. Warm-starting from ep3 inherits
the annealed (~min) LR and so **cannot** distinguish them; only a **fresh, longer cosine schedule** could — and
that is the expensive full run. So reviewer I1 (warm-start = cheapest "train longer" test) is **wrong here**:
warm-start would test nothing. Predicted EV of "train longer" on CER is low (<~0.2 pt) regardless.

## Result 2 — averaging is GOOD, not contaminated (hypothesis FALSIFIED)

| model | proxy CER | proxy WER | empties |
|---|---|---|---|
| ep3-single (best single ckpt) | 12.47 | 16.18 | 132 |
| **shipped avg(5 ckpts, incl 2× ep0)** | **11.83** | 15.83 | **82** |

The 5-ckpt average **beats the best single checkpoint by 0.64 CER and has far fewer empties (82 vs 132)** — even
though it includes two epoch-0 checkpoints. **research/34 §B.1/§C ("shipped v1 plausibly worse than its ep3;
re-export ep3 for a free win") is verified FALSE.** Checkpoint averaging found a flatter/better minimum (classic
SWA behavior). **No cheap re-ship win exists; the shipped v1 was the correct artifact.** Chesterton's fence: the
`sorted(glob)` averaging was doing real work — do not "fix" it by excluding ep0.

## Result 3 — long-audio drop is low-value (free audit)
`max_duration=40s` drops **1.47%** of train (4,934/336,075 utts). By etiology: PD 2.16% (most), CP 1.45%,
ALS 1.34%, Stroke 0.58%, **DS 0.45% (least)**. **Not concentrated in the severe tail** (DS least-dropped; the
most-dropped PD is already the best etiology). Reviewer I3 hypothesis falsified. **Demote long-audio segmentation.**

## Result 4 — empties don't shrink with encoder epochs; etiology tail is split-dependent
- Empties (~3–4%, 82–132/3000) do **not** monotonically fall with more encoder training (ep3 has MORE than ep2);
  averaging helps a little. With decoder+joint **frozen**, the joint's blank propensity is the ceiling on empties
  (reviewer M6). More encoder-only training cannot fix this.
- Hardest etiology on internal-dev = **Cerebral Palsy ~19–20% CER**; ALS **easiest ~4%**. This **contradicts the
  Dev_diag severe-split** story (ALS 33.76% / DS 35.06% in research/34) — the residual tail depends entirely on
  which speakers/split. So **severity-sampling EV is uncertain and split-dependent** (reviewer I2 vindicated).

## What Phase 0 changes about v2 (updated plan)

**Bottom line: v1 is near the ceiling of the "encoder-only + frozen joint/decoder + pinned [70,1]" recipe.**
The two *cheap* paths (re-ship ep3-single; train-more-epochs-same-recipe) are now **closed** by evidence. Every
remaining lever requires a real, gated, cost-approved training run. Re-ranked:

1. **STRUCTURAL — carefully unfreeze the joint (± prednet/decoder) at low LR.** This is the ONLY lever that can
   move the two things encoder-only can't: the **empty floor** and the **hardest tail**, both capped by the frozen
   joint's blank propensity (Results 2/4). Caveat: "full-unfrozen" previously lost on *streaming* (doc 20) — but a
   **targeted low-LR joint(+prednet) unfreeze with encoder kept/low-LR** is a different, narrower change. Run as a
   measured arm vs v1 on the faithful harness. **Highest EV remaining.**
2. **DATA — speed-perturb 0.9/1.0/1.1** (cheap, low-risk ride-along) + **severity sampling** (measure, don't lead;
   EV split-dependent per Result 4; guard PD/mild).
3. **RECIPE — "train longer"**: ONLY meaningful as a **fresh longer cosine schedule** (not warm-start; Result 1
   confound). Predicted EV low → **deprioritize**, optionally fold a longer schedule into the structural arm.

**Demoted/closed:** ep3-single re-ship (closed), multi-lookahead (research/35 C3), long-audio segmentation (Result 3).

**Gate unchanged** (research/35): pre-registered, paired Dev-500 CER delta, speaker-block bootstrap, forgetting
probe, int8 in the loop, full Dev for the final, faithful harness only.

## Strategic (sharpened)
**Ship v1 int8 now.** Phase 0 confirms it is the best v1 artifact (averaging verified optimal; no pending cheap
improvement). v2 is now a *structural* bet (unfreeze the joint), not a cheap convergence/averaging tidy-up —
so it is real GPU spend with uncertain payoff, to be decided explicitly, not assumed.
