# 38 — v2 structural finetune SPEC (executable, gated) — joint-unfreeze + gain-aug (2026-06-29)

> Decision: Q chose "spec the structural v2" after the Test1 diagnosis (research/37). This is the
> executable plan as a DIFF against v1's `nemo_finetune.py` / `run_fullrun.sh`. Pre-registered gates +
> hard budget cap. NO GPU is spent until Q approves this spec and the guardrail go/no-go is green.
> Grounds: research/37 (diagnosis), 36 (Phase 0: convergence flat, frozen-joint caps tail), 34 (v1 facts).

## Thesis (what v2 must fix, from research/37)
1. **ALS empties** = frozen base-Nemotron joint over-emits blank on quiet/severe audio. Cheap inference
   fixes ruled out → must **unfreeze the joint** so it stops over-blanking.
2. **Quiet-audio sensitivity** (`normalize=NA`, empties 2.4× quieter) → **train with gain augmentation**
   so the encoder is robust across the ALS energy range (fix empties at the root, not at inference).
3. **DS wrong-words** (31% non-empty CER) = acoustic-modeling gap → encoder adaptation + severity
   sampling, with a **DS-regression guard** (DS must not get worse).
NOT the lead: more epochs (Phase 0: CER flat by ep3) or severity-weighting alone (lit gains are
closed-speaker; Test1 is held-out → guard, don't lead).

## CORRECTION (2026-06-29, big-picture re-verification) — DE-BUNDLE, lead with augmentation
The original "conservative" arm bundled gain-aug + joint-unfreeze + severity into one run — the same
"unattributable bundle" the reviewer flagged for v1 (research/35 C2), and it mixes the ONE
generalization-correct lever (augmentation) with two capacity-adding/overfit-prone ones. Since the disease
is a generalization gap (Dev→Test +18 vs A1 +1.8), the corrected Gate-1 ordering is:
- **Arm A (LEAD): encoder-only + gain/speed aug** — isolates the augmentation lever, directly attacks the
  quiet-audio empties root cause, improves generalization, uses v1's proven optimizer (no differential-LR
  risk). Most likely single win and the most informative result.
- **Arm B (only if A insufficient): + joint-unfreeze.** Adds capacity only if A leaves empties/DS.
- **Severity oversampling: separate guarded arm**, never bundled; DS/PD-regression guard is the backstop.
Gate 0 still runs the joint_unfreeze path (stress-tests the hardest wiring); Gate 1 leads with Arm A
(`ARM=encoder_only`, the script default).

## CORRECTION 2 (2026-06-30, independent review → research/40) — Arm B contraindicated; pivot to PEFT
Arm A RAN and FAILED (research/39). The remaining Arm B (joint-unfreeze) is now CONTRAINDICATED, not the
fallback: our disease is overfitting / poor Dev→Test transfer, and unfreezing the joint ADDS trainable
capacity in exactly the wrong direction. The literature-correct adaptation for a small atypical corpus that
transfers poorly is **parameter-efficient** finetuning (LoRA / adapters / LHUC), and the empties are
encoder-energy-sensitive (research/37 §5 recovered them with the joint frozen), so the joint is the wrong
module anyway. Replace the Arm-B path with: (1) free Pass-1 RMS-energy normalization + decode-time ILM-bias
correction, then (2) an encoder-LoRA/adapter + MILD gain-aug (±6–10 dB) arm — no severity, no speed-perturb,
full data. The −20/+10 dB gain range in this spec is too aggressive (it hurt Arm A). See research/40.

## The change set (diff vs v1 `nemo_finetune.py`)

### A. New freeze mode: `joint_unfreeze` (the core lever)
Replace the binary `--freeze {full,encoder_only}` with a third mode. Unfreeze **encoder + joint**;
keep **prednet/decoder FROZEN** (full-unfrozen lost on streaming, doc 20 — keep the LM prior fixed,
only let the joint relearn the blank-vs-token decision):
```python
elif a.freeze == "joint_unfreeze":
    for p in m.decoder.parameters():      p.requires_grad = False   # prednet/LM frozen
    for p in m.joint.parameters():        p.requires_grad = True    # <-- the lever
    # encoder stays trainable
    # ablation arm (cheap, later): also unfreeze prednet at very low LR
```

### B. Differential LR (encoder normal, joint LOW) — bypass single-LR setup_optimization
The joint is pretrained; a high LR destabilizes it. Build param groups manually:
```python
enc_p   = [p for p in m.encoder.parameters() if p.requires_grad]
joint_p = [p for p in m.joint.parameters()   if p.requires_grad]
opt = torch.optim.AdamW([
    {"params": enc_p,   "lr": LR},            # 1e-4 (v1)
    {"params": joint_p, "lr": LR * 0.1},      # 1e-5 (joint, gentle)  <-- ablate 0.05/0.1/0.2
], weight_decay=1e-3, betas=(0.9, 0.98))
# wire a CosineAnnealing scheduler over TOTAL steps with WARMUP onto `opt`, then
# m._optimizer = opt / configure via a LightningModule.configure_optimizers override.
```
(Implementation note: NeMo's `setup_optimization` builds one group; v2 needs a `configure_optimizers`
override returning `[opt], [sched]`. SMOKE-TEST this wiring in Gate 0 — silent fallback to one LR is the
top wiring risk. Verify `len(opt.param_groups)==2` and both lrs in the log.)

### C. Gain augmentation in training (fixes empties at root)
v1 had none. Add random gain + speed perturb to the train augmentor (NeMo `audio_augmentor`):
```python
m.cfg.train_ds.augmentor = OmegaConf.create({
    "gain":  {"prob": 0.5, "min_gain_dbfs": -20.0, "max_gain_dbfs": 10.0},  # span the quiet-ALS range
    "speed": {"prob": 0.5, "sr": 16000, "resample_type": "kaiser_best",
              "min_speed_rate": 0.9, "max_speed_rate": 1.1},
})
```
Rationale: empties RMS ~0.025 vs non-empty ~0.059; ±gain teaches energy-invariance. (If NeMo's classic
dataloader ignores `augmentor`, apply gain in a collate/transform — verify in Gate 0 that loss/CER move
with vs without, i.e., the aug is actually applied.)

### D. Severity-aware sampling — guarded ride-along (NOT lead)
Build the train manifest with capped oversampling of severe etiologies, test-independent:
```python
# in prep_nemo_manifest: weight = {ALS:2.0, DS:2.0, CP:1.5, Stroke:1.5, PD:1.0}, cap 2x, by etiology only
# (no model-error proxy -> avoids circularity). Keep a clean-English/PD slice intact for forgetting probe.
```
Guard: PD/mild must not regress (forgetting), DS must not regress (it's the hardest residual).

### E. Validation + checkpointing fixes (v1 defects, research/34 §B)
- `limit_val_batches`: use a fixed representative **~3k** internal-dev subset (v1 used biased 480).
- Select on **CER** (wire a CER val metric; v1 used WER). Smoke-check it in Gate 0.
- Checkpoint-average the **converged tail only** by the actual val metric, **float params only**
  (v1 averaged all incl. ep0 + int buffers — though Phase 0 showed averaging helped; keep it but clean).
- Epochs: 4–6 is enough (Phase 0: CER flat by ep3); the lever is structural, not convergence.

## Pre-registered gates (cheapest-decisive-first) — WRITE THE CRITERION BEFORE LOOKING

**Gate 0 — wiring smoke (~4k utts, 2 epochs, joint_unfreeze).** No accuracy claim. Confirm: 2 LR param
groups in log; joint grads non-zero; gain aug actually applied (CER differs vs aug-off on 200 utts); CER
val metric wired; no NaN/bf16 blowup; export path still works. Fail ⇒ fix wiring, do not proceed.

**Gate 1 — GUARDRAIL (subset ~40–60k utts OR 3 epochs) → export → int8 → faithful `local_decode.py` +
scorer on Dev_diag-425 AND Dev-500.** Pre-registered PROCEED-to-full criterion (all must hold):
- Dev_diag **empty-rate < 5.0%** (v1 = 7.3%) — the direct test that joint-unfreeze fixed blanking.
- Dev_diag **CER < 22.0%** (v1 = 23.58%), delta outside the paired speaker-block CI.
- **PD CER ≤ 5.0%** (v1 = 4.49%; forgetting guard) AND **Dev-500 CER ≤ 10.5%** (v1 = 9.91–10.6%).
- **DS CER not worse than v1** (23.58%-set DS = 33.39%; regression guard).
Any miss ⇒ STOP, do not pay for the full run. (This gate is the whole point: it kills the bet cheaply
if the structural lever doesn't move empties/tail.)

**Gate 2 — FULL run** (all ~309k weighted, 4–6 epochs) only if Gate 1 green. Same eval on **full Dev**.

**Submit criterion (pre-registered):** submit IFF full-run int8 beats v1 on **full Dev_diag** with
**non-overlapping speaker-block CI**, no PD/DS regression, AND an A1-style Dev→Test projection (apply the
measured Dev→Test slope honestly — ours is large) lands the Test1 estimate **below A1's 23.44%**. Else do
NOT submit; A1 stays the Pareto entry. Never submit on a proxy (house rule).

## Budget & logistics
- Pod `3dwiczo41jeg1y` (H200, $4.39/hr). v1 full run ≈ 13 h wall (doc 33); v2 joint-unfreeze similar.
- **Hard cap: 24 GPU-hours (~$105).** Go/no-go at Gate 1 (guardrail ≈ 2–4 h) before the full run.
- Persist checkpoints to /workspace (network volume), not /dev/shm. Stop pod immediately after each gate.
- All eval through the faithful harness + int8 in the loop (research/32/37 pattern).

## Open risks / decisions for Q
1. **Joint LR ratio** (0.05/0.1/0.2 of encoder LR) — start 0.1; Gate-0 cheap ablation.
2. **Prednet:** keep frozen (default, safer) vs unfreeze at very-low LR (ablation arm if Gate 1 is close).
3. **Severity cap** (2× vs none) — default modest 2×, with the DS/PD guards as the backstop.
4. The Dev→Test gap means even a green full-Dev gate may not transfer; the submit criterion bakes in an
   honest projection, but the residual risk is real. A1 remains the safe fallback.

## Single next action
Implement A–E into `scripts/nemo_finetune_v2.py` locally + a `run_v2_guardrail.sh`, then run **Gate 0 +
Gate 1 only** (guardrail, hard-capped), and decide at the pre-registered criterion before any full run.
