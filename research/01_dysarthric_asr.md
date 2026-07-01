# Axis 1 — Dysarthric / Pathological Speech ASR

Tags: `[V-lit]` paper-verified · `[V-web]` web-verified · `[K]` knowledge · `[?]` open.
Lens: which accuracy techniques transfer to a **small streaming CPU** model (not just big offline ones).

## 1. The dominant lever: fine-tune a strong supervised model (confirmed)
- SAPC1: all top teams fine-tuned a foundation model; transducers (Parakeet) > Whisper > SSL on SAP.
  Zero-shot SAP Dev WER: wav2vec2 32.6, HuBERT 25.5, Whisper-v3 21.7, Canary-1B 15.4, Parakeet-TDT 12.8.
  `[V-lit]` (Takahashi'25). → **SSL (wav2vec2/HuBERT) is NOT the path here**; supervised transducer is.
- "All-data" fine-tuning (typical + all etiologies) beats disease-specific FT. `[V-lit]`
- Full FT > freezing decoder/joint. `[V-lit]`

## 2. Synthetic / TTS data augmentation (strong, but mostly for *personalized/low-data* regimes)
- TTS-personalization + speaker-embedding interpolation to synthesize controlled-severity dysarthric
  speech: zero-shot CER 36–51% → **7.3%** after FT on real+synthetic; synthetic adds **~18% relative
  CER reduction** / ~7% relative WER over personalized FT alone. `[V-web]` (arXiv 2508.06391, 2505.12991)
- "Adaptation to fluent speech alone did NOT help" — must model impaired articulation specifically. `[V-web]`
- **Relevance to us**: SAPC2 has **1400 h** of real dysarthric data — we are NOT in the low-data regime
  the TTS papers target. Synthetic data's marginal value shrinks with abundant real data. → **Lower
  priority** unless error analysis shows specific gaps (rare words, severe speakers, specific etiology).
  `[?]` revisit after baseline error analysis.

## 3. Data augmentation (cheap, standard, keep)
- Speed perturbation + time-stretch + SpecAugment: ~7.9% relative WER improvement on dysarthric. `[V-web]`
- "Dysarthric SpecAugment (DSA)" — manipulate healthy features to mimic dysarthria. `[V-web]`
- GAN-based augmentation beats plain speed-perturb by ~0.9–3.0 abs WER, but adds pipeline complexity. `[V-web]`
- Severity-prediction as an **auxiliary multitask** prevents overfitting during FT. `[V-web]`
- **Verdict**: speed-perturb(0.9–1.1)+SpecAugment is the must-have, low-cost baseline aug (also what the
  SAPC1 winner used). GAN/synthetic = optional later. Severity/etiology aux-task = train-time only
  (no test-time labels in the streaming interface) — candidate for a later ablation.

## 4. LLM / generative error correction (GER) — powerful offline, but BAD fit for Track 2
- Whisper-medium + LoRA-Llama-3.1-8B + CycleGAN + N-best rerank → UA-Speech WER 20.61% (−73.9% rel). `[V-web]`
- LLM-agent post-ASR "Judge-Editor" over top-k: −14.51% WER + semantic gains. `[V-web]` (arXiv 2601.21347)
- Team d in SAPC1 used LoRA-Whisper + LLM GER. `[V-lit]`
- **Verdict for Track 2**: an 8B LLM post-pass adds huge CPU cost and latency → **destroys TTLT** and
  likely blows the 15000 s budget. A *tiny* rule-based or n-gram correction might be tolerable, but
  LLM GER is **out** for a CPU-streaming latency track. (Could matter for Track 1, not us.)

## 5. Speaker adaptation / personalization
- Team h SAPC1: speaker-vector personalization. TTS papers personalize per-speaker. `[V-lit/web]`
- **But test speakers are unseen & speaker-disjoint** → per-speaker FT impossible at test time.
  We need speaker-*robust* training, not per-speaker tuning. On-the-fly speaker adaptation (e.g. i-vector/
  x-vector conditioning) is possible but adds latency/complexity on CPU. `[?]` low priority.

## 6. Disfluency / repetition handling (competition-specific, important)
- min-two-refs CER picks the better of {with-disfluency, without-disfluency} ref per utterance. `[V]`
- SAPC1 winner's key failure: internal LM **deletes repetitions** (7 speakers had long stutter runs). `[V-lit]`
- Implication: an over-strong internal LM that "cleans up" disfluency can *hurt* when the with-disfluency
  ref is the scoring ref. A transducer with modest decoder context may transcribe repeats more faithfully.
  → monitor repetition errors explicitly in EDA; consider decoder context size as a knob. `[?]`

## 7. Etiology landscape (SAP) — what we're optimizing for
- 5 etiologies: PD, DS, ALS, CP, stroke. SAP-240430 was **PD-dominated (~74% dur)**; PD = hypokinetic
  dysarthria (low energy, flat pitch, breathy/whispery, articulatory reduction). `[V-lit]`
- Per-etiology (SAPC1): baseline PD 17.09 / ALS 16.88 WER → top5 10.06 / 7.36. ALS improved more
  (lower speaker variability). `[V-lit]`
- → Optimize primarily for PD/hypokinetic, but keep cross-etiology robustness; report per-etiology CER.

## Axis-1 takeaways for the candidate decision
1. Core lever = **full FT of a supervised streaming transducer on all SAP data** + speed-perturb+SpecAugment
   + checkpoint averaging. (High confidence, cheap, proven.)
2. Synthetic/TTS, GAN, severity-aux, speaker-adapt = **second-wave** options, gated by error analysis.
3. LLM GER and big-model tricks = **out** for the CPU-latency track (Track 2), even though they win Track 1.
4. Watch **disfluency/repetition** — it interacts with the min-two-refs metric.

Open: §2 synthetic-data value at 1400 h scale `[?]`; §6 decoder-context vs repetition fidelity `[?]`.
