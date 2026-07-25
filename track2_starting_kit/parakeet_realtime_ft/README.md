# parakeet_realtime_ft — low-latency Pareto-corner candidate (Track 2)

Fine-tuned deploy target for `parakeet_realtime` (120M) cache-aware streaming ASR.
Plan: `research/47_parakeet_realtime_120m_finetune_plan.md`. Rationale: `research/46` §6–§7.

## Status (Stage 0 — local prep, no pod spend)
- **Deploy wrapper (`model.py`) VERIFIED LOCALLY**: 5-method contract, 100 ms→model-chunk
  buffering, callback firing, reset semantics, `<EOU>`/special-token stripping, text extraction.
  Checks: `python3 -m py_compile model.py`, NeMo-absent `import`, and
  `scripts/smoke_parakeet_ft_wrapper.py` (mock-model contract smoke).
- **UNVERIFIED — needs pod** (marked `# >>> VERIFY-ON-POD <<<` in code/config):
  1. exact `from_pretrained` id + model class (RNNT vs TDT) — `config.weights.model_name`.
  2. NeMo streaming-param setup + `conformer_stream_step` return tuple (the Nemotron break point).
  3. decoder priming/SOS on first chunk (NeMo-internal here; confirm first-chunk hyp isn't garbage).
  4. `[70,1]` support + frame→sample chunk conversion for this ckpt.

## Files
- `model.py` — the 5-method streaming `Model` (edit logic here only if a `# VERIFY` block fails on pod).
- `config.yaml` — checkpoint id, `att_context_size=[70,1]`, rnnt head, special-token strip. **Edit values here.**
- `setup.sh` — install NeMo + fetch base `.nemo` into `weights/` (idempotent; won't overwrite an FT ckpt).
- `requirements.txt` — runtime deps (consistent with `setup.sh`).

## Fine-tuning (GPU pod, gated — do NOT run without explicit Q go)
Reuse `scripts/nemo_finetune_v2.py` **unmodified** except the base checkpoint. Arms (research/47 §4):
```bash
# Arm A — encoder-only (safe anchor; less overfit)
python scripts/nemo_finetune_v2.py --mode smoke --freeze encoder_only --epochs 4 \
    --base-nemo weights/parakeet_realtime.nemo --train-ctx "[70,1]" \
    --train-json train.json --val-json val3k.json --out-dir OUT_A
# Arm B — low-LR joint unfreeze (attacks the empty floor; gentler joint LR)
python scripts/nemo_finetune_v2.py --mode smoke --freeze joint_unfreeze --epochs 4 \
    --base-nemo weights/parakeet_realtime.nemo --train-ctx "[70,1]" \
    --train-json train.json --val-json val3k.json --out-dir OUT_B
# augmentation: speed-perturb 0.9-1.1 + gain ON by default; SpecAugment dysarthria-tuned.
# ablate with --no-speed / --no-augment (one variable at a time, paired faithful-gate compare).
```
**GATE0 pod check before the paid run:** `nemo_finetune_v2.py:48` restores via
`EncDecRNNTBPEModel`. parakeet may need the generic `ASRModel.restore_from` (as `model.py` uses)
if it is TDT/a different class. Confirm the class + that `m.joint`/`m.decoder`/`m.encoder.
set_default_att_context_size` all resolve on the loaded parakeet model BEFORE launching. This is a
1-line pod-time fix, not a local edit to the shared Nemotron script.

## Ship gate (house rule — never skip)
Real `local_decode.py` (both passes) → `evaluate.sh` → official sclite on Dev, gated on held-out
severe (Dev_diag) + empty-rate + clean-speech forgetting probe. A Dev win is NOT a submit trigger
(Nemotron Dev→Test +18). See `validate-against-real-harness`.
