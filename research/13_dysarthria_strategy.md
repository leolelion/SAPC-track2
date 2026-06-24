# 13 — Dysarthric ASR: literature-grounded strategy (2026-06-25)

Triggered by the question "will finetuning help the severe/empty failures, or do we need a smarter
architecture/strategy?" Answer from the literature: **finetuning is the proven winning path, our exact
architecture has a published dysarthric result, and severity-aware + augmentation target the severe tail.**

## 1. Finetuning is THE winning strategy (not a guess)
- **Interspeech 2025 Speech Accessibility Project (SAP) Challenge** — same data family as ours. 12/22 teams
  beat the whisper-large-v2 baseline; **top WER 8.11%**. Every proposed system **fine-tuned a foundation model
  (NVIDIA Parakeet or Whisper) on SAP data**. (arXiv 2507.22047)
- **Parakeet/FastConformer == Nemotron's architecture.** The Interspeech 2025 winner paper *"Fine-tuning
  Parakeet-TDT for Dysarthric Speech Recognition"* (Takahashi et al.) used **Parakeet-TDT 1.1B, OFFLINE (not
  streaming)**, fully unfrozen, 20 epochs → **Test1 WER 5.97 / Test2 8.11**. NVIDIA also ships a real trainable
  `.nemo` for our exact model (`nemotron-speech-streaming-en-0.6b.nemo`, 2.47 GB) + an official cache-aware
  streaming finetune script. So our model has a direct, supported adaptation path.
  - CORRECTION (verified by independent review 2026-06-25): an earlier version of this doc cited "WER
    28.4%→19.0% / 36.3%→23.7%" for FastConformer dysarthric finetuning. Those numbers are **NOT** in the
    Takahashi paper and could not be verified — treat as unsourced; the verified figures are above.
- Our own evidence agrees: the SAP-finetuned **zipformer A1 = 23% Test1** vs zero-shot Nemotron 51%.

=> **Will finetuning help? Yes — directionally well-supported, but quantify with care.** Every public datapoint
is **WER on OFFLINE ≥1.1B** models; we'd be running **CER on a STREAMING 0.6B** model — an extrapolation across
three regime changes. So "51% → ~15–20%" is a HOPE, not a forecast. The honest expectation: competitive with /
modestly better than the zipformer (23.4%), not a guaranteed blowout. Nemotron's median ~7% on mild speech is
the reason the ceiling *could* be higher.

## 2. Severity-aware training targets exactly our failure (the empties)
- Incorporating **speech-impairment severity** gives significant WER cuts: up to ~16% relative on E2E
  Conformer / Wav2vec2 (Geng et al., arXiv 2305.10659); **severity-specific finetuning ~32% relative** on
  TORGO/UASpeech (Sapkota et al., Springer 2026).
- Our diagnostic showed failure is severity-driven (ALS/CP 39%/32% empty; Parkinson's 0%). Severity-aware
  finetuning is aimed straight at that tail.

## 3. Data augmentation for the severe tail
- **TTS-synthesized dysarthric speech** (multi-talker TTS + severity coefficient + pause insertion): gains
  "in particular for severe and moderate-severe"; FastSpeech2-augmented −13.17% WER moderate (arXiv 2308.08438,
  2406.08568). **Speed/tempo perturbation**: ~9.3% relative (UASpeech).
- "Train on **as much speech as possible** — typical + all etiologies" beat per-etiology models (NeMo SSL work).

## 4. Post-ASR correction (stacks on any model, architecture-agnostic)
- **LLM-based post-ASR correction: −14.5% WER**; **confidence-guided correction 13.10% → 4.19%** on SAP-shared,
  generalizes across ASR backbones (arXiv 2601.21347, 2509.25048). Could apply to the zipformer too.

## 5. The catch — and why our TRACK constrains the menu
- **Severe intelligibility has a hard ceiling: low-intelligibility speech stays ~80–90% WER even at SOTA.** So
  finetuning will convert many empties → partial transcripts and slash the mean, but the most severe speakers
  won't become perfect. The 25% empty rate is mostly moderate-severe (recoverable), not all bottom-tier.
- **We are Track 2: streaming, CPU-only, latency-scored.** The SAP Challenge winners used **offline** Whisper/
  Parakeet full models — not directly usable. Our constraints rule out:
  - heavy offline Whisper-large,
  - heavy LLM post-correction (latency/CPU budget — though a *light* rule-based or small-LM corrector may fit).
  - => our viable lane = **finetune a STREAMING model**: cache-aware FastConformer/Nemotron (NeMo) or the zipformer.

## Recommended strategy (evidence-ranked, Track-2-feasible)
1. **Finetune the streaming model on ALL SAP data** (Tier 1, proven): NeMo cache-aware FastConformer-RNNT/TDT
   for Nemotron (higher ceiling) and/or continue the zipformer. Single biggest win: 51% → ~15–20%.
2. **Severity-aware + augmentation** (Tier 2, severe-tail): severity conditioning/stratified sampling + speed
   perturbation; optionally TTS-synth for severe speakers.
3. **Light post-ASR correction** (Tier 3, if latency budget allows): small LM / rule-based, not a big LLM.

## Still-open empirical checks (cheap, pod-light)
- **exp A (did NOT run yet):** decode the finetuned **zipformer on the same 425-utt diag set** (install
  sherpa-onnx) → confirms how much a SAP-adapted model already recovers on the exact utterances Nemotron empties
  on. Direct lower-bound on "what finetuning buys."
- Capture the **real Codabench worker CPU spec** (ingestion `[CPU_DIAGNOSTIC]` line) — still unknown.

Sources: SAP Challenge 2025 (arXiv 2507.22047); Fine-tuning Parakeet-TDT for Dysarthric Speech (Takahashi,
Interspeech 2025); NVIDIA/AWS Nemotron Speech ASR finetuning guide; Geng 2305.10659; Sapkota (Springer
s00034-026-03515-4); dysarthric TTS aug 2308.08438 / 2406.08568; post-ASR correction 2601.21347 / 2509.25048;
SSL for dysarthric/elderly 2407.13782.
