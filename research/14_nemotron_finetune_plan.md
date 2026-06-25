# 14 — Nemotron (cache-aware FastConformer) SAP finetuning — experiment plan (v2)

Revised 2026-06-25 after self-critique + an independent skeptical review. Grounded in [[13_dysarthria_strategy]];
packaging solved ([[submission-offline-packaging]]); validation gated by the real harness
([[validate-against-real-harness]]). **Changes from v1 are marked [v2].**

## 0. EVIDENCE STATUS — read before trusting any number in this plan [v3]
**We have ZERO direct evidence for the actual task: finetuning a 0.6B STREAMING FastConformer on dysarthric
speech.** All evidence is transfer-by-analogy:
- The dysarthric-finetuning recipe/numbers come from **Parakeet-TDT 1.1B, OFFLINE** — different size, context
  regime, base data, tokenizer, and possibly a prompt/TDT mechanism (our finetune config is
  `..._streaming_PROMPT.yaml`; the exported decoder is plain RNN-T → train-graph ≠ export-graph).
- exp A (the rescue test) used the **zipformer** — a *completely different* architecture (icefall, not
  FastConformer). It proves *a* finetuned model rescues Nemotron's failures, NOT that finetuning *Nemotron* does.

Split the evidence accordingly:
- **DIRECTIONAL ("finetuning on SAP helps a lot") = robust** — holds across Whisper, Parakeet, wav2vec2, AND our
  zipformer. Not Parakeet-dependent. High confidence.
- **RECIPE + NUMBERS (LR, 20 epochs, "train fully unfrozen", "~6% WER / 15–20% CER") = Parakeet-1.1B-offline
  specific → PRIORS TO ABLATE, not settings.** In particular, "train fully unfrozen" (§5) was imported from the
  Parakeet paper; a smaller streaming model on limited data may instead favor **freezing / LoRA** to avoid
  overfitting + catastrophic forgetting of the mild-speech competence that is our whole reason for Nemotron.

## 0b. GATES (cheap → expensive) [v3 reorders]
- **exp A — DONE** ([[15_expA_beam4_results]]): SAP-finetuned zipformer cut the hard-set CER 47.5→29–31%,
  empties 25→6%, rescued 53% of Nemotron's catastrophic failures. => the data IS adaptable (directional green).
  Does NOT prove the Nemotron-specific path works.
- **NEW true gate — direct Nemotron finetune smoke** (§9 gate 1–3): characterize the model, overfit a batch,
  tiny-subset finetune; confirm *Nemotron's own* failures move AND its mild-speech edge survives. Only this
  gates the full Nemotron run.

## 1. Base model — VERIFIED trainable [v2]
- `nvidia/nemotron-speech-streaming-en-0.6b` ships a real **`.nemo` (2.47 GB)**, not just ONNX (NVIDIA Open
  Model License). Official finetune path exists: `speech_to_text_finetune.py` + config
  `conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml`, load via
  `init_from_nemo_model`. Sibling: `nemotron-3.5-asr-streaming-0.6b`; riva tutorial notebook documents the recipe.
- Rationale unchanged: highest zero-shot ceiling we measured (median ~7% mild Dev), cache-aware streaming, same
  family as the published dysarthric Parakeet result.

## 2. Data preparation
- Source: SAP manifests `/workspace/SAPC2/manifest/{Train*,Dev*}.csv` + cuts `data/sapc2_*_cuts.jsonl.gz`.
- **[v2] VERIFY FIRST: actual train hours/utts on the pod, and whether any TYPICAL (non-dysarthric) speech
  exists.** v1 assumed "use all data incl. typical speech" — do not bank on a typical-speech pool that may not
  be present. Convert to NeMo manifest (`audio_filepath,duration,text`).
- **Speaker-disjoint splits** (manifest `speaker`) — train / internal-dev / held-out. Never leak speakers.
- Resample/verify 16 kHz mono; drop corrupt/zero-length; segment >40 s via forced alignment (matches winner).

## 2b. TARGET-TEXT decision — resolve BEFORE training [v2, was missing]
Likely silent CER killer. Nemotron-Speech natively emits **punctuation + capitalization**; SAP refs are
**normalized lowercase**. Two things to settle and ablate up front:
1. **Case/punct:** train on normalized (lowercased, punctuation-stripped) targets so the model stops emitting
   punct/caps that the streaming hypothesis carries into scoring — OR keep cased targets and rely on the
   scorer's normalizer. Decide by a small ablation; don't leave it implicit.
2. **Disfluency:** scorer is **min-over-two-refs** (with/without disfluency). Default target =
   `norm_text_without_disfluency`, but ablate against `with_disfluency` — the choice is not obvious.

## 3. Severity handling (targets the empties)
- No explicit severity label; proxy by **etiology** primarily. **[v2] Avoid pure circularity** — v1 derived
  severity from zero-shot Nemotron's own CER, which risks training a curriculum around the base model's
  idiosyncratic blind spots. Prefer etiology + (if available) an independent intelligibility/severity metric;
  use the model's own error rate only as a secondary signal.
- Severity-aware **oversampling** of severe/empty-prone speakers (lit: severity-specific finetuning ~32% rel).

## 4. Augmentation
- **Speed perturbation 0.9/1.0/1.1 + SpecAugment** (NeMo built-in; matches the winning dysarthric recipe).
- Tier-2 (defer): TTS-synthesized dysarthric speech for severe speakers.

## 5. Training recipe (NeMo) [v2 corrections]
- **Train in cache-aware / limited-context STREAMING mode** — this is a hard requirement, not a footnote. A
  model finetuned in offline full-context mode degrades under streaming (NeMo: "inconsistency between how the
  model is trained and how streaming inference is done"). Use the streaming config's `att_context_size` LIST +
  `att_context_probs` (multi-lookahead); set the train context to match the deployment chunk (note the riva
  recipe trains `[56,3]`, our export uses `[70,6]` — **pin this deliberately**, don't assume).
- **[v2] Do NOT freeze by default.** The one published dysarthric-Parakeet result found freezing the
  decoder+joint *hurt* (WER 6.31 frozen vs 6.26 unfrozen) and trained **fully unfrozen**. Train everything
  unfrozen; make freezing an *ablation*, not the default (v1 had this backwards).
- Optimizer AdamW, warmup + cosine, **LR ~1e-4–5e-4** (the riva notebook's `lr=0.1` is almost certainly a
  placeholder/Noam-scale — do NOT use 0.1 raw). ~20 epochs (winner used 20), early-stop on internal-dev CER.
- **[v2] Guard the `limit_train_batches` / ~1000-samples-per-epoch trap** (NeMo issue #15782): the streaming
  finetune config can silently cap each epoch to ~1000 samples. Verify the dataloader sees the full set, or the
  "smoke (1k)" run could accidentally BE the full run.
- Checkpoint every epoch; **average top-5 by internal-dev CER**.

## 6. Compute & cost [v2]
- 1×H200 (GPU-scarce → use the autorun retry-start pattern). **Estimate (unverified): order tens of GPU-hours
  ≈ ~1–3 days on a single H200** for ~hundreds of hours of audio × ~20 epochs. Persist checkpoints to MooseFS,
  not /dev/shm. **Set a hard budget cap and a go/no-go after the smoke gate.**

## 7. Evaluation protocol
- **Faithful harness ONLY** ([[validate-against-real-harness]]): export → real `local_decode.py` + official
  scorer. Never trust a NeMo-internal number for the submit decision.
- **[v2] State success gates in CER on a Test1-REPRESENTATIVE speaker-disjoint split** (not the severe-enriched
  diag set, which would understate the model). Target: beat zipformer A1 (23.4% CER) with margin; track empty
  rate, ALS/CP, and no Parkinson's regression. Acknowledge the **severe-tail ceiling** (~80–90% WER even at
  SOTA) may cap how low the mean can go.
- Per etiology × duration × severity (reuse `scripts/diag_crosstab.py`); bootstrap 95% CIs; compare on the SAME
  utts vs zero-shot Nemotron AND zipformer A1. Streaming latency (TTFT/TTLT) — a CER win that breaks latency is no win.

## 8. Export & packaging
- Export finetuned → cache-aware streaming ONNX, int8, regen `tokens.txt`; reuse the validated offline-zip
  recipe (local-mel, bundled ORT wheel, no NeMo/network). **[v2] Validate offline-vs-streaming CER delta**, not
  just "it exports" — a small delta confirms streaming parity survived finetuning.

## 9. GATES (ordered) [v3]
0. **MODEL CHARACTERIZATION (NEW, before any training).** Load the `.nemo` and answer: is the decoder TDT or
   RNN-T? what is the `_streaming_prompt` mechanism (a prompt token? context conditioning?) and does finetuning
   need to feed/preserve it? does the train graph match the exported RNN-T inference path? what `att_context`
   list + `att_context_probs` does the streaming config use? Until these are known, every hyperparameter is a
   guess. (The Parakeet evidence cannot answer any of this.)
1. **Overfit-a-single-batch:** ~20–50 hard utts (ALS/CP), **augmentation OFF**, drive train CER → ~0. If it
   can't, the pipeline (data/target alignment, loss, backprop, optim, prompt handling) is broken — STOP.
2. **DIRECT NEMOTRON SMOKE (the real go/no-go) [v3]:** tiny-subset finetune of the actual `.nemo`; confirm
   (a) Nemotron's OWN empties/failures move on internal-dev, and (b) its **mild-speech CER does NOT regress**
   (catastrophic-forgetting check — the median-7% is the whole reason to use Nemotron). exp A only showed a
   *different* model adapts; this is the first Nemotron-specific evidence. No-go here ⇒ don't spend the full run.
3. **Freeze/LoRA ablation [v3]:** small head-to-head — full-unfrozen vs partial-freeze vs **LoRA/adapter** (a SAP
   winner used AdaLoRA). Do NOT default to "fully unfrozen" (that was a Parakeet-1.1B import); pick by dev CER +
   forgetting. Consider mixing in general/typical speech or KD to limit forgetting.
4. **Target-text ablation** (§2b) resolved before full training.
5. **Subset finetune** + faithful-harness eval gate (dataloader sees full data, not the ~1000-cap).
6. **Full finetune** + avg-5 + export + offline-validate + streaming-parity delta + faithful Dev eval.
7. Gates green → ONE validated Codabench submission.

**Expectation reframe [v3]:** we'd be the FIRST to finetune a 0.6B streaming FastConformer on dysarthric speech.
Directional confidence (it will help) is high; the achievable CER is genuinely unknown and the Parakeet numbers
do not bound it. Treat ≤20% CER as an aspiration to test, not a forecast.

## 10. Open risks / unverified
- Whether the `.nemo` finetune config preserves multi-lookahead by default (verify in the YAML).
- True production LR; exact GPU-hours; the typical-speech data question.
- **Strategic:** every public SAP winner used a **1.1B OFFLINE** model; nobody has shown a **0.6B STREAMING**
  dysarthric result. This is a high-variance first-of-its-kind bet. exp A + the zipformer beam-4/LM track is the
  higher-EV near-term move; the Nemotron finetune is the higher-ceiling gamble. Decide explicitly, gated on exp A.
