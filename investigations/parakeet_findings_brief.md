# Parakeet — Findings Brief (2026-07-28)

One-page distillation of `parakeet_improvement_framework.md`. `[V]` = literature/toolchain-verified,
`[F]` = measured fact on our data, `[T]` = theory, `[U]` = unverified inference.

## The situation
Parakeet FastConformer-RNNT holds our only Pareto-frontier route: a ~371 ms latency corner no other
submission touches. Its problem is CER on the severe tail. We are #4/4 and Pareto-dominated on Test1;
the win is to drive parakeet's severe CER below the banked zipformer's ~24.85% **while keeping the
latency corner.**

## Where the loss lives `[F]` (Dev_diag, n=425)
| | value |
|---|---|
| Total CER | **29.93%** |
| Non-empty CER | 21.00% |
| Empties | 48/425 (11.3% of utts) |
| Empty contribution to CER | **+11.29 pts** |
| CER if empties were merely *average* | **21.00%** → already beats zipformer 24.85% |

**The empty tail IS the CER problem.** Everything else is second-order. Empties are concentrated by
etiology: ALS 24.6% empty, Down 7.5%, CP 10.5%, **Parkinson 0%, Stroke 0%** — a vocal-energy gradient
(low-energy/hypophonic speakers empty out; normal-energy ones don't).

## The decisive finding `[F]`
On the exact 48 utterances parakeet emits **empty**, the fine-tuned **zipformer is non-empty 58.3%**
and nails many at 0.0 CER (`alexa`, `dog`, `hey siri`, `football`). → **the information is in the
signal; the empty is a parakeet-decoder pathology, not an acoustic floor.** This is what makes the
tail *winnable* rather than a dead end.

## Root cause `[T]`, lit-backed `[V]`
Classic **RNN-T "confident-blank"**: the joint over-emits blank on out-of-domain, low-energy, short
inputs (arXiv 2108.10752). We only fine-tuned the **encoder (Arm A)**; the **joint + prediction net
stayed LibriSpeech-frozen**, so the blank/token boundary was never retrained for dysarthric energy.
Ruled out by probe: EOU misfire (0 eou-only empties), causal-gain patch (falsified), TDT
duration-skip (model is plain RNNT, not TDT).

## The levers (ranked)
1. **Arm B — unfreeze joint + prediction net** `[T]`. The network that owns the blank boundary. Graded
   unfreeze + differential LR to dodge the prior Nemotron collapse scar.
2. **FastEmit loss regularization** `[V]` — **the strongest new lever.** NeMo-native
   (`loss.warprnnt_numba_kwargs.fastemit_lambda`, λ≈0.005, safe band 0.004–0.01, regresses ≥0.02).
   Scales token-prob gradient by (1+λ), leaves blank unchanged → direct counter to confident-blank.
   **Folds into the Arm B retrain for free** (training-time, orthogonal to the greedy_batch decode pin)
   and is the **only lever that improves BOTH Pareto axes** — fewer empties (↓CER) *and* earlier
   emission (↓TTFT). Pairs mechanically with Arm B, which relearns exactly the boundary FastEmit shapes.
3. **Augmentation, reframed** `[V]` — the dysarthric-ASR field has **no gain/energy augmentation**
   (only rate + spectral), so the energy angle borrows generic robustness (random gain / MUSAN noise /
   SpecAugment); the loss does the heavy lifting. Plus **oversample short wake-words/commands** (RNN-T
   internal LM over-deletes rare short phrases).
4. **Beam search** `[T]` — a latency-gated CER squeeze applied *after*, never the empty fix.

## Limitation: the evaluation set (Q's flag, honest) `[F]`
All numbers rest on **Dev_diag, n=425** — a **severity-filtered diagnostic slice**, not a random dev
sample. Two distinct concerns:
- **Sample size:** the empty analysis is n=48; the recovery estimate is ~28/48. 95% CI on 58.3% is
  roughly ±14 pts → the *point estimates are noisy* but the *qualitative conclusion (a large chunk is
  recoverable) survives even at the CI floor.* Direction is robust; exact numbers are not.
- **Representativeness (the bigger risk):** Dev_diag is chosen to be *hard*. A larger, mixed dev set
  would *dilute* the empty rate (Parkinson/Stroke add zeros) — but Test1 is severe-heavy, so the severe
  slice is arguably the *right* optimization target. The open question is whether Dev_diag425's
  severity mix *matches Test1's*.
- **Recommendation:** a larger-dev run is worthwhile, but scoped as a **representativeness check, not a
  power fix** — decode parakeet on the full Dev set (and, if possible, a Test1-severity-matched slice),
  confirm the empty rate + etiology gradient hold, and re-anchor the 21% projection. Feasibility gates
  first: (a) do we have full-Dev audio locally, (b) does parakeet inference run on this box or need the
  pod (cost gate). We currently have on disk only the n=425 decode + probes, no full-Dev decode.

## Pre-registered gate (house rule)
Submit **IFF** the organizers' real harness (`local_decode.py` both passes → `evaluate.sh`) shows
**severe Dev CER ≤ 24% AND mean latency ≤ 420 ms.** Never submit to test a hypothesis; proxy numbers
never sign off. Every number above except official-scorer ones is proxy.

## Immediate next steps
- **(a)** Dig the old Nemotron train config: was the collapse a **Noam-warmup** artifact? (cheap, local;
  could rehabilitate the whole fine-tuning line.)
- **(b)** Scope the full-Dev / severity-matched decode as the representativeness check above.
- Then: draft the Arm B pod run-book (graded unfreeze + `fastemit_lambda` sweep + generic energy aug +
  short-command oversampling), with kill-gates.
