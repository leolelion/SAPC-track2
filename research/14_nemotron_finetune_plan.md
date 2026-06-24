# 14 — Nemotron (cache-aware FastConformer) SAP finetuning — experiment plan

The path to actually winning Track 2. Grounded in [[13_dysarthria_strategy]] (finetuning is the proven SOTA;
FastConformer 28.4→19.0 on dysarthric; SAP-challenge winners all finetuned). Packaging already solved
([[submission-offline-packaging]]); validation gated by the real harness ([[validate-against-real-harness]]).

## 0. Objective & success gates (define BEFORE training)
- **Primary:** Nemotron Test1 CER from 51.5% → **≤ 20%** (beat zipformer A1 = 23.44%), measured ONLY via the
  faithful pipeline (real `local_decode.py` + official scorer) on Dev first, then a held-out speaker-disjoint set.
- **Streaming gate:** RTF < 1 and TTFT/TTLT within budget on CPU (Track-2 constraint) — re-export must stay
  cache-aware streaming, att_context preserved.
- **Empty-rate gate:** drop the Dev empty rate from ~6% (rand300) toward <1%; specifically improve ALS/CP.
- **No-regression gate:** mild-speech (Parkinson's) CER stays ≤ ~11%.

## 1. Base model
- `nvidia/nemotron-speech-streaming-en-0.6b` **NeMo checkpoint** (.nemo), NOT the ONNX export — finetuning
  needs the NeMo graph. Cache-aware streaming FastConformer encoder + RNN-T/TDT decoder, att_context_size=[70,6].
- Rationale: highest zero-shot ceiling we measured (median ~7% on mild Dev), same family as the published
  Parakeet-TDT dysarthric result, and NVIDIA ships a Nemotron-Speech domain-adaptation recipe.

## 2. Data preparation
- Source: SAP data already on pod — `/workspace/SAPC2/manifest/{Train*,Dev*}.csv` + cuts in
  `/workspace/finetune/data/sapc2_*_cuts.jsonl.gz`. Audio under `$DATA_ROOT/processed/`.
- Convert our manifest schema (`id,speaker,etiology,audio_filepath,duration,text,norm_text_*`) → NeMo JSON
  manifest (`audio_filepath,duration,text`). Use `norm_text_without_disfluency` as the training target
  (matches ref2; disfluency modeling is risky for a first pass).
- **Speaker-disjoint splits** (manifest `speaker` col) — train / internal-dev / held-out. Anti-overfit; mirrors
  how Test1 has unseen speakers. NEVER let a dev/test speaker leak into train.
- Resample/verify 16 kHz mono; drop corrupt/zero-length; cap absurdly long (>40s) or segment them.
- "Use as much data as possible" (lit): include ALL etiologies + any typical-speech we have. Per-etiology
  models lost to all-data models.

## 3. Severity handling (targets the empties)
- We have no explicit severity label, only `etiology`. Proxy severity by (a) etiology prior (ALS/CP severe),
  (b) per-speaker zero-shot CER from our diagnostic (high-CER speakers = severe). Build a `severity_bucket` col.
- **Severity-aware sampling:** upweight severe/empty-prone speakers in the training mix (oversample) so the
  model sees more of the hard tail (lit: severity-specific finetuning ~32% rel).
- Optional Tier-2: condition on severity (prepend a severity token / use a severity embedding) — defer to v2.

## 4. Augmentation (severe-tail boosters)
- **Speed perturbation** 0.9/1.0/1.1 (NeMo built-in) — ~9% rel in UASpeech lit; cheap, do from v1.
- **SpecAugment** (NeMo default) — standard regularization.
- Tier-2: **TTS-synthesized dysarthric speech** for the most severe speakers (multi-talker TTS + severity
  coefficient) — bigger lift for severe/moderate-severe but a separate build; defer to v2.

## 5. Training recipe (NeMo)
- Finetune cache-aware streaming FastConformer-RNNT/TDT, keep att_context_size=[70,6] (must match the export
  for streaming parity). Start from the released weights.
- LR: small (e.g. 1e-4 to 5e-4 with warmup+cosine), AdamW; freeze the first N encoder layers for the first
  epoch then unfreeze (stabilizes adaptation). RNNT/TDT loss as in the base.
- Batch by duration buckets; mixed precision on the H200. ~10–20 epochs over SAP train, early-stop on
  internal-dev CER.
- Checkpoint every epoch; **average top-5 checkpoints** (avg-5 gave our best zipformer; standard NeMo trick).

## 6. Compute & cost
- 1×H200 pod (already provisioned, GPU-scarce — use the autorun retry-start pattern). Budget the run; persist
  checkpoints to MooseFS (`/workspace/finetune/nemo_ft/`), NOT /dev/shm.
- Phase the spend: smoke (1k utts, 1 epoch, confirm loss decreases + decode sane) → subset (10–20% data) →
  full. Stop at each gate if the metric doesn't move.

## 7. Evaluation protocol (rigorous, per house rules)
- **Faithful harness only** ([[validate-against-real-harness]]): export → run real `local_decode.py` + official
  scorer on Dev. Never trust a NeMo-internal number for the submission decision.
- Break down by **etiology × duration × severity** (reuse `scripts/diag_crosstab.py`); track empty-rate.
- Bootstrap 95% CIs; compare against zero-shot Nemotron AND zipformer A1 on the SAME utts.
- Latency on the streaming subset (TTFT/TTLT). A CER win that breaks streaming latency is not a win.

## 8. Export & packaging
- Export finetuned model to cache-aware streaming ONNX (encoder/decoder), int8 quantize, regenerate
  `tokens.txt` — then reuse the validated offline zip recipe ([[submission-offline-packaging]]): local-mel,
  bundled onnxruntime wheel, no NeMo/network. Re-validate offline before any upload.

## 9. Risk register / decision gates
- **R1 streaming parity:** finetune must keep att_context/cache identical or the ONNX streaming export breaks →
  validate a 5-utt streaming decode right after export.
- **R2 overfit to train speakers:** guard with speaker-disjoint dev; watch the train/dev gap.
- **R3 severe-tail ceiling:** lit says low-intelligibility plateaus ~80–90% WER — finetuning will cut the mean
  hard but won't zero the worst speakers; set expectations accordingly.
- **R4 latency regression** from a heavier finetuned decoder — measure, don't assume.
- **Go/No-go after smoke:** internal-dev CER must drop meaningfully vs zero-shot, else stop and re-examine
  data/LR before spending full-run GPU.

## 10. Phased timeline
1. Data prep + NeMo manifest + speaker-disjoint splits + severity buckets (CPU/local-ish).
2. Smoke finetune (1k utts, 1 epoch) — sanity.
3. Subset finetune + faithful-harness eval gate.
4. Full finetune + avg-5 + export + offline-validate + faithful Dev eval.
5. If gates pass → single, validated Codabench submission.

Parallel low-risk track meanwhile: ship zipformer beam-4 (gated) + measure exp A (does the already-finetuned
zipformer rescue the same utts) — see `scripts/run_expA_beam4.sh`.
