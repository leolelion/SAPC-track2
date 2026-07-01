# 35 — v2 finetune plan, FINAL (after independent review) — the executable plan (2026-06-28)

> This supersedes the v2 plan sketch in doc 34 §D. It folds in the independent reviewer's critique
> (verbatim findings archived below). Background/verified-facts: docs 33, 34. Shipped v1: doc 32.
> Core change from the reviewer: **verification-first + de-bundled**, not one big bundled run.

## The reviewer's verdict (accepted)

The v2 sketch was "directionally reasonable but not sound as written" because (a) the under-convergence
claim rested on a noisy ~480-utt, likely few-speaker val curve; (b) it bundled 5 levers into one
unattributable run whose result would likely sit inside the ±2–3 pt Dev-500 noise; (c) multi-lookahead's
only real benefit (latency flexibility) is low-value on this CER-won track while its downside (capacity
dilution → CER regression at the deploy context) directly opposes the #1 lever.

## Executable plan — phased, cheapest-decisive-first

### Phase 0 — VERIFY before any training GPU (mostly eval; one pod session, gate-protected)
The 4 v1 checkpoints (`ft-epoch={0,1,2,3}-val_wer=*.ckpt`) are on the pod. Do all of this with **no training**:
1. **Re-test under-convergence on a RELIABLE val.** Evaluate ep0–ep3 on the **full** 27,135-utt internal-dev
   (not the biased 480) and on **Dev-500 via the faithful harness** (export/quant or at minimum NeMo-transcribe
   for the quick read, faithful harness for the decision). Plot CER vs epoch.
   - Still descending at ep3 on full val ⇒ "train longer" justified → Phase 1.
   - Flattening ⇒ convergence is NOT the lead lever → pivot to data levers (severity / long-audio), skip Phase 1.
2. **Cheap v1.5 test.** Compare **epoch-3 single ckpt (int8)** vs the shipped **avg(ep0..ep3) (int8)** on the
   faithful harness, same utts/seed. If ep3-single wins → re-ship it for ~zero cost. (Averaging *converged* points
   often helps; excluding ep0 may help or hurt — measure, don't assume. Watch `load_state_dict(strict=False)`
   silently dropping dtype-mismatched keys.)
3. **Free data audit.** Count utts dropped by `max_duration=40.0` **by etiology**. If ALS/DS are over-represented
   in the dropped set (severe speakers talk slowly → longer utts), forced-align long-audio segmentation rises in
   priority as a cleaner severe-tail lever than oversampling.

Phase 0 decides whether v2 is a convergence play, a data play, or neither — at near-zero cost.

### Phase 1 — Convergence arm, ISOLATED (gated on Phase 0.1 = "still descending")
- **Warm-start-continue from the epoch-3 `.ckpt`** (it has Adam optimizer state — the cheapest, most direct
  "train longer" test), **pinned [70,1]**, **no other changes**, to convergence (≈8–15 epochs total) with:
  - reliable validation: `limit_val_batches=1.0` (or a fixed representative ~2–4k subset), select on **CER**
    (needs wiring + its own smoke check — NeMo's native metric is WER).
  - clean checkpoint averaging: average the **converged tail only**, ranked by the **actual** val metric (not
    filename glob), **float params only** (exclude integer buffers); fix the `sorted(glob)` bug.
  - tune warmup down (~5–10%); revisit peak lr=1e-4 for the longer schedule (one sentence, not silent inherit).
- Decision: **paired** Dev-500 CER delta vs v1, speaker-block bootstrap; pre-registered (below). This isolates
  the single highest-confidence lever so a win/null is attributable.
- (Also keep a clean fresh-start-from-base run as the clean final baseline if warm-start shows a real gain.)

### Phase 2 — Data levers, SEPARATE arms (gated on Phase 1 win)
- **Severity-aware sampling** — do NOT lead with it. Use a test-independent reweighting (cap oversampling of
  severe etiologies), and **measure PD/mild as a guard** (PD already 4.8% — must not regress). Its payoff
  depends on the unknown Test2 etiology mix; gate accordingly. Avoid the circular "error-propensity" proxy.
- **Forced-align long-audio segmentation** — promote here if Phase 0.3 shows ALS/DS over-dropped at 40 s.
- **3-way speed perturbation 0.9/1.0/1.1** — cheap, low-risk; can ride in Phase 1 or as its own arm.

### Phase 3 — Multi-lookahead, DEMOTED to a separate measured arm (low priority)
- Keep the deployed run **pinned [70,1]** (matches v1's win) unless a separate arm **measures** multi-lookahead
  as CER-neutral-or-better at the chosen deploy context. If trained, pick the deploy context by sweeping all four
  contexts on Dev-500 (CER) × Dev_streaming (latency) and choosing the Pareto point — do not assume free.

## Gate hardening (all phases) — pre-register BEFORE looking
- Decision metric: **paired** Dev-500 (and final: **full Dev**) CER delta, **speaker-block bootstrap**, ONE
  deploy context by a fixed rule. Require the delta comfortably **outside** the paired CI, not a point estimate.
- 500 utts/119 spk is underpowered for sub-1-pt gains → score full Dev for the final gate to tighten CI.
- Single-run vs single-run confounds modeling change with seed/data-order variance → 2-seed check if budget
  allows, else declare it a known confound and demand a margin.
- Add a **clean-English forgetting probe** to the gate table (longer training + severity sampling can induce it).
- Faithful harness only (`local_decode.py` + official `evaluate.sh`); int8 in the comparison; never submit on a proxy.

## Open decisions to make explicitly (not inherit silently)
- **Target text** cased+punct vs normalized — measure its TTFT/TTLT effect (punct tokens cost emission steps on
  a latency track); decoder/joint are frozen so they emit punct regardless — is cased the right encoder target?
- **Empty-output ceiling**: with decoder/joint frozen, encoder-only can only partly fix empties (frozen joint
  blank propensity). Note the tension; it caps how far the severe tail moves.
- Optional: cheap transcript-quality/outlier filtering before the full run.

## Strategic (unchanged)
**Ship v1 int8 now** (validated, frontier-worthy) to lock a Pareto point; v2 is then upside, not the only entry.

## Single highest-value next action
**Phase 0.1** — re-evaluate the 4 existing v1 checkpoints on the full internal-dev + Dev-500 faithful harness.
It re-tests the thesis the whole plan depends on, at near-zero cost, before any training GPU.

---

## Appendix — independent reviewer findings (archived verbatim, 2026-06-28)

CRITICAL: C1 under-convergence claim rests on noisy ~480-utt (few-speaker) curve; ep2→3 delta (0.0114) <
ep1→2 (0.0181) ⇒ could be asymptoting; re-eval 4 ckpts on full val for free first. C2 5 levers bundled →
unattributable; gains masked by ±2.8 Dev-500 CI; de-bundle. C3 multi-lookahead low-value on CER-won track
(A1 #1 with worst latency), risks CER regression by diluting [70,1]; demote.
IMPORTANT: I1 fresh-start rationale partly wrong — epoch-3 .ckpt has optimizer state; warm-start-continue is
the cheapest "train longer" test; fresh-start for final clean run. I2 severity sampling: don't lead; guards
PD/mild; unknown test mix; error-propensity proxy circular; lit is low-data regime. I3 max_duration=40 drops
long utts, likely over-removes slow severe ALS/DS — quantify free. I4 gate needs paired+speaker-block+
pre-registered+full-Dev; 500 underpowered; seed-variance confound; multiple-comparison risk. I5 target-text/
punctuation latency effect untouched.
MINOR: M1 v1.5 averaging test must be on int8, same seed/utts; strict=False can drop keys. M2 CER selection
needs wiring + smoke. M3 peak lr not revisited for longer run. M4 no transcript-quality filtering. M5 add
forgetting probe to the gate table. M6 empties capped by frozen decoder/joint.
VERDICT: directionally reasonable but not sound as written; execute only after (1) verify convergence cheaply,
(2) de-bundle (convergence isolated at [70,1]), (3) demote multi-lookahead to a measured arm, (4) harden gate.
Highest-value change: re-eval the 4 existing v1 checkpoints on full val + Dev-500 before any training GPU.
