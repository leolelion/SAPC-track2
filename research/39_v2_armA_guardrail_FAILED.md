# 39 — v2 Arm A guardrail: FAILED the gate (encoder-only + aug + severity is WORSE than v1) (2026-06-30)

> First real v2 GPU run (research/38 plan, de-bundled Arm A). Trained encoder-only + gain/speed aug on the
> severity-weighted 50k guardrail subset, 4 epochs. Faithful eval (FP32, rebuilt-from-ckpts after a quota bug
> truncated the .nemo save — see §infra). Pre-registered verdict: **STOP**. Pod stopped, ckpts freed.
> Artifacts: `~/Downloads/v2_guardrail_results/`.

## Result: WORSE than v1 across the board

| set | v2 Arm A | v1 baseline | delta |
|---|---:|---:|---:|
| **Dev_diag CER** | **27.35%** | 23.58% | **+3.77 (worse)** |
| Dev_diag empties | 7.76% (33/425) | 7.3% | ~same (NOT fixed) |
| **Dev-500 CER** | **12.27%** | ~9.91% | **+2.4 (worse)** |
| ALS / CP / DS / Stroke (Dev_diag) | 38.6 / 26.0 / 35.8 / 30.3 | 33.5 / 21.6 / 33.4 / 25.1 | all worse |
| PD (Dev-500) | 3.94% | ~4.5% | ok (no forgetting) |

Pre-registered gate: empty-rate<5% ✗, Dev_diag CER<22% ✗, Dev-500≤10.5% ✗, DS-not-worse ✗ → **STOP**.
(The dedicated PD-probe scored n=0 — a script bug in the pd_probe.csv build — but Dev-500 PD 3.94% shows no
forgetting, so that check is moot.)

## What this tells us (the informative part)
1. **The empties did NOT move (7.76% vs 7.3%).** Gain augmentation at the *encoder* did not fix the
   quiet-audio empties → **confirms the empties are gated by the FROZEN joint's blank propensity, not just
   input energy.** Encoder-only cannot fix them. This vindicates the core diagnosis (research/37): the
   joint must be unfrozen. Arm A never touched the joint, so empties were always going to be capped.
2. **Aug + severity actively HURT general acoustic modeling** — Dev-500 (easy/moderate) regressed too
   (12.27% vs 9.91%), not just the severe tail. The aggressive gain range (-20/+10 dBFS) + speed-perturb +
   2x severity oversampling degraded the model. Augmentation is not free here.
3. **Attribution caveat:** Arm A bundled aug + severity (trained on the severity-weighted subset), so we
   cannot separate which hurt more. The de-bundle was incomplete.
4. **NeMo-internal val CER (descended 22.1→20.4%) ≠ faithful Dev_diag (27.35%).** The streaming/export path
   gives the real number; v1 went through the same path (23.58%), so the v2-worse comparison is fair. Do not
   trust NeMo-internal val for the decision (house rule).

## Scoreboard so far (all attempts to move the severe tail)
- Cheap inference fixes (blank-penalty fallback; gain-norm): **failed** (research/37).
- v2 Arm A (encoder-only + aug + severity): **failed, worse than v1** (this doc).
- v2 Arm B (joint-unfreeze): **untested** — the only remaining diagnosed lever; the optimizer bug was fixed
  in `nemo_finetune_v2.py` but not yet verified on GPU.

## Decision for Q (open)
Two reasonable paths, explicitly:
- **(a) Run Arm B (joint-unfreeze)** — the diagnosis says the frozen joint is the binding constraint on the
  empties, and Arm A confirms encoder-only can't touch them. This is the one lever that directly targets the
  mechanism. But: another ~2h GPU, the optimizer wiring needs Gate-0 re-verification, and aug/severity should
  be DROPPED or made gentle (they hurt). Suggested Arm B = joint-unfreeze, NO severity, mild-or-no aug.
- **(b) Stop the Nemotron severe-tail rescue.** Cheap fixes failed; encoder-only+aug failed. A1 (23.44% Test1)
  remains the safe Pareto entry. Doc 14 §10 called the zipformer track higher-EV. Reconsider whether more GPU
  on Nemotron is worth it vs. banking A1 / pivoting.

## Infra note (cost the run a save)
`/workspace` MooseFS hit a **per-volume quota** mid-run → truncated the `.nemo` save (exactly 1 GiB) and
blocked scp. Worked around by writing to `/tmp` + `/dev/shm` and rebuilding from the 4 intact `.ckpt`
(averaged) → direct ONNX export. Future v2 runs: keep checkpoints off `/workspace` or prune aggressively;
the 4×7.3 GB ckpts alone blew the quota.
