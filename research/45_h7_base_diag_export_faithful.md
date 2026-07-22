# 45 — H7 base-model diagnostic: the streaming export is FAITHFUL (2026-07-22)

> Ran the H7 mechanism experiment on the **base** Nemotron (finetuned v1/PEFT checkpoint was lost —
> gone from pod AND local, confirmed 2026-07-22). Three decode paths, same base weights, same
> `Dev_diag` (425 severe, speaker-disjoint), same `[70,1]`, **same hand-rolled two-ref CER + official
> normalizer** (proxy scorer — diagnostic only, not a ship number). Pod `3dwiczo41jeg1y` (H100),
> stopped immediately after copy-back.

## Result — all three paths agree

| Arm | Path | ALL CER | empties |
|---|---|---:|---:|
| **A** | NeMo `transcribe()` — masked single pass | **43.4%** | 76 (18%) |
| **C** | NeMo-native cache-aware streaming (`speech_to_text_cache_aware_streaming_infer.py`) | **43.8%** | 77 |
| **B** | **our** ONNX export (`[70,1]`, cache_support, CHUNK_NEW=16, SOS=BLANK_ID) → real `local_decode.py` | **43.4%** | 76 |

All 425/425 matched (B keyed by id; C position-aligned, 0 text-mismatches). Per-etiology tracks within
~1–2 pts across arms (PD ~10–11%, ALS ~59–61%, CP ~46%). **Spread = 0.4 CER points.**

## Conclusion — reverses the standing H7 framing
- **There is NO streaming collapse in the pipeline on base weights** — not transcribe, not NeMo-native
  streaming, not our ONNX export. **Our export code is faithful** (43.4% == transcribe, exactly).
- This **falsifies** the leading H7 hypothesis (`investigations/h7_fix_plan.md`, `audio-cpp-fit.md`):
  that the historical finetuned collapse (PEFT: transcribe 18.46% → our streaming 28.19%, research/41)
  was a bug in **our ONNX cache-stepping**. It was not — the same pipeline reproduces transcribe exactly.
- Therefore the finetuned 18→28 gap, **if real and same-set**, was **weights/training-specific**
  (research/41's own hypothesis: PEFT adapters don't survive the streaming cache-normalization), i.e. a
  **cache-aware-finetuning** problem (Arm 3, GPU, re-train), NOT a cheap export fix. The alternative —
  a **confounded comparison** (the 18.46% eval set was never documented; flagged this session) — is now
  *more* plausible precisely because the pipeline is provably faithful.
- Note finetuning **did** survive export and help: base streams at 43.4%, v1 finetuned streamed at
  ~23.58% Dev_diag. The "collapse" was only ever relative to a transcribe *proxy*, never to base.

## Caveats
- **Base weights, not the lost finetuned v1/PEFT.** This proves the PIPELINE is faithful; it does not
  re-measure a finetuned transcribe-vs-streaming gap (checkpoint gone — cannot).
- Proxy scorer (consistent across arms; not sclite/`evaluate.sh`). Fine for A/B/C comparison, not a ship number.
- NeMo `main` migrated the streaming infer example argparse→Hydra (`model_path=/dataset_manifest=/
  att_context_size=[70,1]`); the old `--asr_model` flag is gone. `run_stream_infer.sh` needs updating.

## Decision impact — deflates, does not revive Nemotron
No cheap export fix exists (export is clean). Any Nemotron revival needs GPU cache-aware **re-finetuning**
(checkpoint is gone → from scratch), which is EV-negative with the Dev→Test transfer wall (+18 vs +1.8,
F9) still standing. **Default unchanged: ship the gated beam-4 zipformer (~21%).** H7 is retired with
proof — the export was never the villain.

## Artifacts
- Local: `~/Downloads/h7_base_diag/` (base_ac.log, base_c.log, arm_b.log, transcribe_base.json,
  streaming_out_*.json, arm_b_hyp.csv).
- Scripts: `scripts/base_ac_diag.sh` (A), `scripts/base_c_only.sh` (C, Hydra), `scripts/arm_b_base.sh` (B).
- Supersedes the "export bug" branch in `investigations/h7_fix_plan.md` / `audio-cpp-fit.md`.
</content>
