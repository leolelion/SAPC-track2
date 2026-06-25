# 19 — Pre-full-run review (independent audit + corrections, 2026-06-25)

Two independent agents + a verification run audited Gate-2 before committing GPU to the multi-hour full
finetune. Net: direction validated, but DON'T fire the full run yet — do the cheap fixes/checks below first.

## Corrections to earlier claims (honesty)
- **Empties↔blank-calibration mechanism = UNVERIFIED just-so story.** The encoder-adaptation/forgetting half is
  literature-backed ([2207.00216](https://arxiv.org/pdf/2207.00216), [2210.03255](https://arxiv.org/pdf/2210.03255));
  the specific empties claim is not. Drop it as "mechanism."
- **Encoder-only "win" is within noise** (1.3 CER pts / 4 empties on 425 utts, no bootstrap CI). Defensible:
  "at least as good, theory-favored," not "wins." Add CIs before claiming.
- **Gate-2 transcribe() CER ≠ deployment CER.** transcribe() is context-faithful (applies att mask) but NOT
  chunk-boundary-faithful (single pass, no 100ms cache carryover). Real number needs ONNX + local_decode.py.
  NVIDIA ships `speech_to_text_cache_aware_streaming_infer.py` for true streaming sim.
- **Clean gain (verified) = 43.4→30.8 CER (−12.6), empties 18→9%** on unseen speakers (overlap=0). Real, large.

## Confirmed
- ONNX/CPU/int8 3-graph deployment path is sound (base already runs it); **export REQUIRES
  `model.set_export_config({'cache_support':'True'})`** or cache I/O is missing (#1 pitfall). int8-vs-fp32 CER
  on dysarthric audio is unmeasured.
- No Train/Dev speaker leakage (SAP is speaker-disjoint by design; our check = overlap 0).

## Fundamentals to fix BEFORE the full run (prioritized)
1. **Target-text ablation [HIGHEST cheap value].** Train smoke on `text` (native cased+punct) vs current
   `norm_text_without_disfluency`; NVIDIA says match native style + let the min-over-two-refs scorer normalize.
   Forcing the model to unlearn punct/caps may waste several CER pts. NEVER RUN — run it.
2. **Export smoke enc-only → ONNX int8 (cache_support=True) → faithful local_decode.py on Dev [DECISIVE].**
   Gives the TRUE streaming deployment CER for a finetuned checkpoint (resolves the transcribe-vs-streaming
   confound) AND proves the finetuned-export path works. Untested. Compare to zero-shot ONNX-harness 47.5%@[70,6].
3. **Context regime decision.** Train multi-lookahead `[[70,13],[70,6],[70,1],[70,0]]` with `att_context_probs`
   (preserve the latency lever; NeMo-intended) OR fix ONE deploy context and benchmark everything at it. Stop
   mixing [70,1]-train with [70,6]-deploy. (Requires adding multi-lookahead support to nemo_finetune.py.)
4. **Forgetting insurance.** Add a general/typical-speech **replay slice** + a held-out clean-English probe
   (current "no forgetting" only tested on mild *dysarthric* PD, not clean English). NVIDIA-recommended.
5. **LR & ckpt.** Lower peak LR to ~1e-4 with ~10–20% warmup (2e-4 is a forgetting accelerant); **wire
   checkpoint averaging** (avg top-5) — currently TODO, not implemented in nemo_finetune.py.
6. SpecAugment: verify config sane for dysarthria (don't inherit blindly); aggressive time-masking can hurt.

## PRE-RUN EXPERIMENT RESULTS (2026-06-25)
- **EXP1a — ONNX export: ✓ WORKS.** Finetuned enc-only exported with `set_export_config({'cache_support':'True'})`
  → `encoder-model.onnx` + `decoder_joint-model.onnx` (cache-aware streaming graphs). Finetuned-deploy path proven.
  (Integration note: NeMo's I/O names differ from danielbodart's; name-matching needed when wiring into
  `local_decode.py` submission model.py at packaging.)
- **EXP2 — target-text ablation: WASH (no lever).** Cased+punct = 30.7% CER / 10% empty vs normalized 30.8% / 9%
  on Dev_diag. Identical within noise → the agent's "normalized costs several pts" did NOT materialize, because
  the official scorer normalizes both sides. **DECISION: keep `norm_text_without_disfluency`** (marginally fewer
  empties). One open question CLOSED.
- **EXP1b — streaming-sim: DEFERRED.** NeMo's `speech_to_text_cache_aware_streaming_infer.py` hung on `--help`
  (heavy import) in our env. The deployment-faithful true-streaming CER (confound C) will be resolved
  authoritatively at packaging via **ONNX → `local_decode.py`** (the real submission gate) rather than the NeMo
  sim script.

## Locked full-run config decisions
- Target text = normalized (wash). Export path = works. Confound C → resolved at ONNX+local_decode packaging.
- Still to set: multi-lookahead training w/ `att_context_probs`; LR ~1e-4 + ~10-20% warmup; replay slice +
  clean-English probe; checkpoint averaging (avg top-5).

## Recommended next (cheap, smoke-scale, before the full run)
- Run #2 (export + faithful harness on the existing enc-only smoke checkpoint) and #1 (target-text ablation)
  together — both reuse the smoke checkpoint / a quick smoke re-run, both decisive, ~1 pod session.
- Then fold #3–#6 into the full-run config and launch.

Sources: NVIDIA Nemotron 3.5 finetune blog (native text style, replay, fixed-step) —
https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr ; NeMo models/streaming —
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html ; sherpa-onnx FastConformer
int8 export #790 — https://github.com/k2-fsa/sherpa-onnx/issues/790 ; SAP challenge speaker-disjoint —
https://arxiv.org/pdf/2507.22047 ; encoder-only adaptation — 2207.00216 / 2210.03255.
