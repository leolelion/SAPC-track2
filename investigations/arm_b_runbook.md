# Arm B Run-book — parakeet FastConformer-RNNT (2026-07-28)

Single-pod, kill-gated recipe to drive parakeet's **severe** CER below zipformer's ~24.85% while
holding the ~371 ms latency corner. All numbers until the real harness runs are proxy. Cost gate:
this doc + the code patches must be complete and syntax-checked locally **before** any pod start.

Companion docs: `parakeet_improvement_framework.md` (why), `parakeet_findings_brief.md` (one-pager).

---

## 0. Pre-registered success criterion (write it before we run it)
**Submit IFF** the organizers' real harness (`local_decode.py` both passes → `evaluate.sh` two-ref
scorer) shows **severe Dev CER ≤ 24% AND mean latency ≤ 420 ms.** Never submit to test a hypothesis.
Kill-gate metric at each stage below is stated explicitly; the pod dies the moment a gate fails.

## 1. What already exists (do NOT rebuild — Chesterton's Fence)
- **`scripts/nemo_finetune_v2.py`** already implements Arm B as `--freeze joint_unfreeze`:
  encoder+joint trainable, **prednet (decoder) frozen** (streaming-safe, doc 20), differential LR
  (joint = enc·`--joint-lr-mult` 0.1), **manual AdamW + warmup→cosine** (NOT Noam), gain+speed aug,
  SpecAugment (dysarthria-tuned), `m.wer.use_cer=True` selection, top-5 CER-ckpt averaging,
  EarlyStopping(patience 2), and GATE0 introspection asserts. This is model-agnostic via `--base-nemo`.
- Arm A (`--freeze encoder_only`) produced the banked 29.96% Dev_diag / 13.18% Dev_streaming.
- **Design note (flagged):** the framework said "joint **+ prediction** unfreeze"; the implemented
  conservative arm keeps the **prednet frozen** and unfreezes only the joint. That is defensible — the
  confident-blank decision lives in the **joint**, and freezing the prednet keeps the internal LM
  stable (streaming-safe). Start here; only escalate to prednet-unfreeze if joint-only under-delivers.

## 2. What this run-book ADDS (three surgical changes, all opt-in / default-off)
1. **FastEmit** (`--fastemit-lambda`, default 0.0 = no behavior change). The one verified lever that
   moves both Pareto axes: scales token-grad by (1+λ), leaves blank-grad → counters confident-blank
   AND lowers TTFT. NeMo-native (`warprnnt_numba_kwargs.fastemit_lambda`). Sweep {0.003, 0.005, 0.01};
   regresses ≥0.02. Folds into the same joint_unfreeze run at zero marginal cost.
   → **IMPLEMENTED** as opt-in `--fastemit-lambda` in `scripts/nemo_finetune_v2.py` (arg at line 34,
   loss rebuild + `[GATE0][fastemit]` print at lines 58–70). Default 0.0 = loss untouched.
   VERIFY-ON-POD: `num_classes` wiring (the GATE0 print must show the lambda actually set).
2. **Short-command oversampling** (`--oversample-short-words N --oversample-short-mult K`, default
   **off**). RNN-T over-deletes rare short phrases (internal-LM bias); our empties are
   disproportionately wake-words (22/48 are ≤3 words). → **IMPLEMENTED** as a manifest pre-pass in
   `scripts/prep_nemo_manifest_v2.py` (`expand()`), composing multiplicatively with the existing
   capped etiology weighting — not in the trainer, which stays clean.
   Note the etiology weights already there: `ALS 2.0 · Down 2.0 · CP 1.5 · Stroke 1.5 · PD 1.0`
   (cap 2×) — that is H3 severity weighting, already banked. ALS and Down are our two highest-empty
   etiologies, so part of the energy-targeted weighting is free.
3. **Stage-0 representativeness decode** (§3). Answers Q's dev-set concern before we pay to train.

## 3. Stage 0 — representativeness check (inference only, cheap, GATE-REP)
**Goal:** the 48/425 empty burden is measured on a severity-filtered slice with n=48 empties (recovery
CI ≈ ±14 pts). Establish, on a broader set, (a) does the empty pathology generalize, and (b) is it a
Dev_diag construction artifact — *before* training spend. This is a measurement to refine the
target, NOT a training go/no-go, with ONE genuine falsification branch.
- **VERIFY-ON-POD first:** `ls $DATA/manifest/` — which Dev manifests exist? Known: `Dev_diag.csv`
  (425 severe), `Dev_streaming.csv` (123 easy). Prefer a **full organizer `Dev.csv`**; if absent, the
  **2k/7-spk speaker-disjoint training val** is the largest population-like set — decode that.
- **Decode BOTH on the SAME broad set, SAME scorer (like-for-like — this is the fix to the old bug):**
  (i) banked **parakeet Arm A** ckpt, (ii) banked **zipformer** (we have it, cheap). Real
  `local_decode.py` → **`evaluate.sh` two-ref sclite** (NOT a hand-rolled proxy — house rule). Report:
  parakeet {empty-rate, empty CER-contribution pts, CER, per-etiology}, zipformer {CER} on that set.
- **GATE-REP branches (thresholds are judgment calls — flagged for Q):**
  - **PROCEED** if parakeet empties still cost **≥ ~5 CER pts** on the broad set (mechanism generalizes;
    the tail is worth fixing) — expected, since Test1 is severe-heavy.
  - **STOP & re-decide with Q** if EITHER: parakeet empty contribution **< ~3 CER pts** on the broad set
    (empties were largely a Dev_diag artifact, not a population problem), **OR** parakeet CER already
    **≤ zipformer CER** on the same broad set (we are not actually behind on the representative
    population → the "catch up to zipformer" premise weakens). Surface with the numbers, do not train on.

## 4. Stage 1 — Arm B training (GPU, GATE-TRAIN)
Base = parakeet `.nemo` (from `setup.sh` / `from_pretrained nvidia/parakeet_realtime_eou_120m-v1`;
VERIFY-ON-POD the resolvable id, §config.yaml). Train = SAP Train json; val = 2k/7-spk val json.

**Ladder (stop at the first that clears the gate; do not run all blindly):**
1. `joint_unfreeze`, **no FastEmit** (λ=0) — isolates the unfreeze effect vs Arm A. GATE0 asserts must
   print 2 optimizer groups + both LRs (already in script).
2. `joint_unfreeze` + `--fastemit-lambda 0.005` — the primary bet.
3. If 0.005 helps but empties remain: try 0.01; if insertions rise (over-emission backfire, open-risk
   #2) back off to 0.003. **Watch val insertion-rate, not just CER.**

Each: 4 epochs, bs 16, `--train-ctx "[70,1]"`, differential LR (enc 1e-4, joint 1e-5 via mult 0.1),
top-5 CER-ckpt average.

**Baseline first (one-time, cheap):** decode banked **Arm A** on the **identical val json** → record
`{val CER, val empty-count, val insertion-rate}`. All GATE-TRAIN comparisons are against THIS, on the
SAME val set (the 27.5% figure is only a prior — re-measure so the comparison is apples-to-apples).

**GATE-TRAIN (all three must hold):**
1. val CER **<** Arm A val CER, AND
2. val empty-count **<** Arm A val empty-count, AND
3. val insertion-rate **≤ Arm A insertion-rate × 1.15** — the hard guardrail on FastEmit over-emission
   backfire (open-risk #2). A rung that cuts empties by trading them for insertions FAILS this gate.

**Early tripwires (kill the rung immediately, keep last good `.nemo`, do NOT escalate LR):** loss NaN/inf;
val CER after epoch 1 **worse** than Arm A (collapse signature — the Nemotron scar); val empties *rise*
vs Arm A; or wall-clock exceeds the per-rung budget [set from the Arm A epoch time on the pod].

**PROXY WARNING (Nemotron scar, non-negotiable):** GATE-TRAIN runs on NeMo's *internal* val metric
(`use_cer`, punct-stripped, single-path) — a PROXY. Passing it **NEVER** implies ship. Only GATE-SHIP
(§5, real two-ref sclite) authorizes a ship claim. This is the exact 24.96%-proxy → 51.5%-real trap.

## 5. Stage 2 — faithful eval + latency (GATE-SHIP, the ONLY ship authority)
For the surviving `.nemo`: point `config.yaml weights.nemo_file` at it, then run the **real** loop.
The two gate halves live on **different sets** — do not conflate them:
```
local_decode.py pass1 (accuracy) on Dev_diag   → evaluate.sh stages 0-2 → CER/WER (two-ref sclite)
local_decode.py pass2 (streaming) on Dev_streaming → evaluate.sh stage 3 (compute_latency) → TTFT/TTLT
```
**GATE-SHIP-CER:** severe **CER ≤ 24%** on **Dev_diag** (two-ref sclite — the primary metric).
**GATE-SHIP-LAT:** **mean(TTFT,TTLT) ≤ 420 ms** on **Dev_streaming** (the manifest with `mfa_speech_start`;
Dev_diag has no streaming manifest). Also report **P50/P90 and the Dev_streaming empty-count**, because
recovering empties **shifts the latency population** — an utterance that was empty (uncounted for TTFT
per `compute_latency.py`) becomes counted, so the mean is not a pure regression check. Sanity:
Arm B changes *weights only* (same [70,1] arch, same greedy_batch decode) and FastEmit *lowers* emission
latency, so LAT should hold or improve — a *rise* means a streaming regression to investigate, not tune.

**Both halves must pass** to equal the pre-registered criterion. Beam is a *separate, later*
latency-gated CER squeeze — never bundled into the empty fix.

## 6. Copy-back + stop
Copy `ft_*.nemo`, all `*.log`, the Stage-0/Stage-2 hyp CSVs + partial JSONs, and per-etiology metric
JSONs to `experiments/exp_armB_parakeet/`. Write the `summary.csv` row + `EXPERIMENT_LOG.md` entry.
**Stop the pod immediately after copy-back.** One session, kill-gates, no idle GPU.

## 7. Open pod-time unknowns to resolve first (cheap, before spending)
- Exact resolvable base-ckpt id (eou vs non-eou) and that `att_context_size [70,1]` is supported.
- Which Dev manifests exist ($DATA/manifest) → picks the Stage-0 set.
- FastEmit `num_classes` wiring for `RNNTLoss` on this NeMo build — the `[GATE0][fastemit]` print in
  `scripts/nemo_finetune_v2.py` (lines 58–70) must show the lambda actually set on the rebuilt loss.
- Train/val json paths (the parakeet FT used SAP Train 331k + 2k/7-spk val — confirm paths on disk).

## 8. What we are NOT doing
- Not starting a pod until Q greenlights (cost gate). Not submitting on any proxy number. Not touching
  scorer files. Not unfreezing the prednet in the first pass. Not cranking λ past the safe band.
