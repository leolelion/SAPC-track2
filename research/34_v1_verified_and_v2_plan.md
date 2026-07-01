# 34 — v1 finetune VERIFIED from pod code/logs + refined v2 plan (2026-06-28)

> Step (b) of the v2 workup: a read-only pod diagnostic (`v1_diag.sh`, pod up ~4 min, ~$0.30, then stopped)
> pulled the actual training scripts, checkpoint val metrics, and manifest stats. This replaces doc-derived
> assumptions with verified facts and sharpens the v2 plan. Source artifacts:
> `/Users/o/Downloads/sapc2_int8_final/v1_diag/{nemo_finetune.py,run_fullrun.sh,prep_nemo_manifest.py}`.

## A. Verified v1 facts (from code + logs, not docs)

### Data (exact)
- `train.json`: **308,909 utts / 683.0 h / 805 speakers**.
- `dev_internal.json`: **27,135 utts / 59.6 h / 70 speakers** — speaker-disjoint holdout (`--dev-speaker-frac 0.08`).
- 805 + 70 = 875 Train speakers (matches doc 18). Held-out **official Dev** (Dev_diag-425, Dev-500) was NEVER
  in training or selection → the 8.76% Dev-500 number is clean.
- Target text = **`text` (cased + punctuation)** (`run_fullrun.sh` `--target text`). Scorer normalizes both sides.
- `max_duration=40.0` → utts >40 s are **dropped** (no long-audio segmentation).

### Recipe (exact, `nemo_finetune.py`)
- Base: `nemotron-speech-streaming-en-0.6b.nemo`, `EncDecRNNTBPEModel.restore_from`.
- Encoder-only: `decoder`+`joint` `requires_grad=False`.
- Optim: **AdamW lr=1e-4, wd=1e-3, betas (0.9,0.98), CosineAnnealing, warmup=15% of total steps, min_lr=1e-6**,
  bf16-mixed, bs=16.
- SpecAugment: dysarthria-tuned (freq_masks=1, freq_width=10, time_masks=10). **ON.**
- Context: `set_default_att_context_size([70,1])` — **single pinned context** (train = eval = export).
- Classic dataloader (`use_lhotse=False`). **No speed perturbation** anywhere.

### Convergence (DECISIVE — empirically under-converged)
Encoder-only checkpoint `val_wer` (in-training selection metric) by epoch:

| epoch | val_wer |
|---|---|
| 0 | 0.3930 |
| 1 | 0.3827 |
| 2 | 0.3646 |
| 3 (final) | **0.3532** |

Monotone decrease through the **final** epoch → training stopped on the **4-epoch cap, not early-stop**. The
model was still improving. This is direct evidence (not literature analogy) that more epochs will help.

## B. NEW defects found in the code (beyond what docs 20/21 flagged)

1. **Checkpoint averaging is contaminated.** `nemo_finetune.py:120-134`: `sorted(glob("ft-*.ckpt"))` then
   averages **all** found checkpoints, including the least-converged **epoch-0 (val_wer 0.393)**. With only 4
   epochs and `save_top_k=5`, every epoch is kept and averaged → the deployed v1 is `avg(ep0..ep3)`, diluting
   the best epoch-3 (0.353) with the worst epoch-0 (0.393). Also casts **all** tensors incl. integer buffers via
   `.float()`. **Consequence: shipped v1 is plausibly worse than its own epoch-3 single checkpoint.**
2. **Validation selection on 1.8% of internal-dev.** `limit_val_batches=30` + `validation_ds.shuffle=False` →
   checkpoint ranking saw the **same 480 of 27,135** utts every epoch. Selection signal is tiny, fixed, biased.
   (We *had* 27k internal-dev utts and used 480.)
3. **Selection metric = WER, deployment metric = CER** — minor misalignment.
4. **Warmup = 15% of total steps.** For the 4-epoch run that is ~11.4k steps ≈ 60% of epoch 1 spent warming up →
   compounds under-convergence. Proportionally fine at more epochs, but worth tuning down (~5–10%).

## C. Cheap pre-v2 win to TEST (no retraining)

Re-export **v1 epoch-3 single checkpoint** (`ft-epoch=3-val_wer=0.3532.ckpt`) instead of the contaminated
average → int8 → faithful `local_decode.py` + official `evaluate.sh` on Dev-500/Dev_diag. If epoch-3-alone
beats `avg(ep0..ep3)` on the real harness, that is a **near-zero-cost CER gain on the already-shipped model**.
Caveat: checkpoint averaging of *converged* points often helps generalization; whether excluding ep0 helps is a
hypothesis the faithful harness must decide, not assume. ~30 min pod, gate-protected.

## D. Refined v2 plan (fresh-start from base .nemo, encoder-only)

Do NOT warm-start from v1: the shipped v1 is an **average** (no optimizer state, blurred weights, contaminated
by ep0) and v2 changes the context regime. Fresh-start from the base `.nemo` is clean and cheap (encoder-only
adapts fast). Fold all cheap high-EV levers into ONE gated run:

| # | Lever | v1 | v2 | Evidence |
|---|---|---|---|---|
| 1 | Epochs to convergence | 4 (capped, still descending) | **8–15**, early-stop on a *reliable* val | §A table |
| 2 | Checkpoint averaging | avg(all incl ep0), int buffers | average **converged tail only** (by actual val metric), float params only; sort by metric not filename | §B.1 |
| 3 | Validation | 480/27k, fixed | `limit_val_batches=1.0` (or fixed ~2–4k representative); select on **CER** | §B.2/3 |
| 4 | Context | pinned [70,1] | **multi-lookahead** `att_context_probs` over `[[70,13],[70,6],[70,1],[70,0]]` | docs 19/21 |
| 5 | Severity sampling | uniform | **oversample ALS/DS** (etiology-proxied) | docs 13/14/01 |
| 6 | Speed perturb | absent | **3-way 0.9/1.0/1.1** | §A, docs 06/01 |
| 7 | Warmup | 15% | tune ~5–10% | §B.4 |
| 8 | Arms | enc-only + full-unfrozen | **enc-only only** (full lost) | doc 20 |
| 9 | Long audio | drop >40 s | (defer) forced-align segmentation | doc 14 |

**Gates (house rules):** smoke (re-confirm wiring with multi-lookahead + severity sampling + speed-perturb) →
full run, hard GPU budget cap → export → int8 → faithful `local_decode.py` + official `evaluate.sh` on Dev-500
**and** Dev_diag with speaker-block bootstrap CI → require **rc=0 AND CER improvement with credible CI vs v1**
before any v2 submission. Never submit on a proxy.

**Residual target:** v1 int8 Dev_diag errors concentrate in **ALS 33.76% / DS 35.06%** — lever 5 aims here.

## E. Out of scope but noted on the pod
The pod also has logs for `parakeet_ctc`, `whisper_finetune`, `gec_training` (generative error correction),
`dars_*` (dysarthria augmentation/retrain), `qwen3` — likely parallel Track-1 / other-agent work. Not reviewed
here; flag for the owner in case any (esp. GEC) is meant to stack on Track 2.

## F. Open question for the independent reviewer
Is fresh-start correct, or is there a case for warm-starting from v1's **epoch-3 single** checkpoint (real
training state) to bank the convergence cheaply while still switching to multi-lookahead? And is the EV ranking
(convergence > multi-lookahead > severity-sampling > speed-perturb) right, or should severity-sampling lead
given the ALS/DS tail dominates the residual?
