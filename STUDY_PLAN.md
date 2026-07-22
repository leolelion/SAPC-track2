# STUDY_PLAN.md — Understanding the SAPC2 Track-2 problem from the ground up

> **Goal (yours):** reach the level where you can *read ASR papers and judge whether a
> method applies to us* — and direct GPU spend / brief agents with confidence.
> **Profile:** solid ML basics; comfortable with linear algebra / basic calculus /
> probability (equations OK if explained); ~10+ hrs/week, ~2–3 week sprint.
> **Method per concept:** ① video for intuition → ② written for depth → ③ hands-on last.
> Every module ends **anchored to our own repo** so you learn on the real problem.

## How to use this
- Follow the weeks in order — each builds on the last. **Critical path = M2 (RNN-T + streaming)**;
  that's what directly explains our roadblock (H7, the offline→streaming collapse). If time gets
  tight, protect M2 and compress M0/M4.
- 🎥 = video (intuition), 📄 = written (depth), 💻 = hands-on (do last), 🔗 = connect-to-our-repo,
  �ˑ = **paper checkpoint** (a real paper you should be able to read/judge by this point).
- Adapt freely: this is a living doc. Check off modules; note where you want to go deeper.

---

## WEEK 1 — Foundations + the Transformer
*Outcome: you understand what a neural net is doing, and you can read "Attention Is All You Need."*

### M0 — ML foundations refresher (fast; you know most of this)
- 🎥 3Blue1Brown, *Neural Networks* series (eps 1–4): what a network is, gradient descent, backprop.
- 📄 Skim: loss functions, train/val/test split, **overfitting & generalization**.
- 🔗 You already lived overfitting: re-read `research/40` (CUHK "large param count → overfits small
  atypical corpus") and our **Dev→Test slope +18 (Nemotron) vs +1.8 (zipformer)**. That IS the
  generalization gap, measured on our data.

### M1 — Sequence models: RNNs → Attention → Transformer
- 🎥 StatQuest or 3B1B on **RNNs / LSTMs** (intuition: hidden state, why long sequences are hard).
- 📄 Chris Olah, *Understanding LSTM Networks* (the canonical explainer).
- 📄 Karpathy, *The Unreasonable Effectiveness of RNNs* (why sequence models matter).
- 🎥 3Blue1Brown, *Attention in transformers* + *But what is a GPT?* (best modern intuition for QKV).
- 📄 Jay Alammar, *The Illustrated Transformer* (read slowly; this is the keystone article).
- 📄ˑ **Paper checkpoint:** Vaswani et al. 2017, *Attention Is All You Need*. Goal: understand
  self-attention (Q·Kᵀ → softmax → weighted V), multi-head, positional encoding, encoder vs decoder.
- 💻 (last) Karpathy, *Let's build GPT from scratch* — build a tiny transformer; makes attention concrete.
- 🔗 Why this matters here: our **Zipformer encoder is a transformer variant**, and the "streaming"
  problem (M2) is fundamentally *"how do you run attention when you can't see the whole sequence yet."*

---

## WEEK 2 — Speech & the architectures WE use  ⭐ critical path
*Outcome: you can explain, mechanistically, why Nemotron empties on quiet/severe audio and why it
collapses under streaming.*

### M2a — Audio → features
- 🎥 Valerio Velardo (*The Sound of AI*), mel-spectrogram / MFCC videos (DSP-for-ML intuition).
- 📄 Ketan Doshi, *Audio Deep Learning Made Simple* (waveform → **log-mel spectrogram**, framing).
- 🔗 **Directly explains our biggest bug:** empties are a *quiet-audio* artifact — `localmel.py` uses
  `normalize=NA`, and empty utts are ~2.4× lower RMS → weak log-mel → the model emits blank
  (`research/37 §5`, `investigations/nemotron_vs_zipformer.md` F4). Understand normalization/CMVN here.

### M2b — ASR paradigms: CTC, RNN-Transducer, attention
- 📄 Awni Hannun, *Sequence Modeling with CTC* (Distill.pub) — the clearest alignment explainer.
- 📄 Loren Lugosch, *Sequence-to-sequence learning with Transducers* (best RNN-T blog).
- 📄ˑ **Paper checkpoint:** Graves 2012, *Sequence Transduction with RNNs* (the RNN-T paper).
- 🔗 **This is the core of our whole project.** Both our models are **RNN-Transducers**:
  encoder + **prediction network** + **joint network** + the **blank symbol**. Map these terms onto:
  - our "frozen joint" debate (`research/39` vs `40`, H6 in the investigation doc),
  - "confident all-blank" empties (blank − best-non-blank margin +3.7–3.9, `research/12` F4),
  - why the crude blank-penalty produced garbage (fighting the model's **internal LM** — see M3).

### M2c — The encoders: Conformer & Zipformer
- 📄ˑ Gulati et al. 2020, *Conformer* (conv-augmented transformer for ASR).
- 📄ˑ Yao et al. 2023, *Zipformer: a faster and better encoder for ASR* — **literally our A1 encoder.**
- 🔗 Read `track2_starting_kit/streaming_zipformer/config.yaml` with the paper open: chunk_size,
  left_context, num_active_paths — you'll recognize every knob.

### M2d — Streaming vs offline (cache-aware chunking)  ⭐ the roadblock
- 📄 NeMo docs, *Cache-aware Streaming Conformer*; icefall *streaming zipformer* docs.
- 📄 Skim Emformer / streaming-conformer intro (chunked attention, left-context caching).
- 🔗 **This is H7 — the current binding problem.** A finetuned Nemotron scores **18.46% offline
  (`transcribe`)** but **28.19% through cache-aware chunked streaming** (`research/41` F8). By the end
  of M2d you should be able to form a *mechanistic hypothesis* for why offline gains die under
  streaming (cache normalization vs full-sequence forward; att_context regime at export). That is
  exactly the question we'd hand a fresh agent.

---

## WEEK 3 — Adaptation, decoding, evaluation + paper-judging capstone
*Outcome: you can read the papers behind our failed arms and independently judge the bets.*

### M3a — Transfer learning, freezing, and PEFT
- 📄 What to fine-tune / what to freeze (encoder vs joint); catastrophic forgetting.
- 📄ˑ Hu et al. 2021, *LoRA: Low-Rank Adaptation*; HuggingFace **PEFT** docs; LHUC (Swietojanski)
  for speaker adaptation.
- 🔗 **Judge our own bet:** `research/40` argued PEFT (fewer params) should beat full-FT for our
  overfitting problem; `research/41` ran it → PEFT was the **worst** arm (28.19%). With M3a knowledge,
  decide for yourself: was the PEFT hypothesis reasonable, and why did it fail on *streaming export*
  specifically? (This is the capstone question.)

### M3b — Domain shift / OOD generalization
- 📄ˑ ICASSP-2021, *RNN-T Models Fail to Generalize to Out-of-Domain Audio* (cited in `research/40`).
- 🔗 Ties F1/F2/F4 together: severe dysarthria is OOD for clean-English pretraining → confident blank.

### M3c — Decoding: greedy, beam, LM fusion, internal LM
- 📄 Beam search for transducers ("modified beam search", icefall); RNN-LM **shallow fusion**.
- 📄ˑ Internal LM Estimation (ILME / density-ratio) — Meng et al. / Zeyer et al.
- 🔗 **Our concrete win + our failed hack:** beam-4 gave −2.6 CER at equal latency (`research/09`) →
  understand *why* wider search helps when the acoustic model is uncertain. And ILM explains why
  blank-penalty made empties worse (`research/12 §5b`) — you were fighting the prediction-net's LM prior.
- 📄ˑ Bonus: Yu et al. 2021, *FastEmit* — blank/latency as a **loss**, not a decode hack (`research/40`).

### M4 — Evaluation & the competition mechanics
- 📄 WER/CER, alignment (sclite), the **min-over-two-refs** scoring; latency **TTFT/TTLT**.
- 📄 int8 / ONNX **quantization** intro (why CPU inference needs it; we verified int8 ≈ fp32).
- 🔗 Read `utils/compute_metrics.py` + `utils/compute_latency.py` + `SAPC-template/CLAUDE.md §3–4`.
  Understand *which validation set predicts Test*: rep-Dev 8.76% (best case) vs severe-enriched
  Dev_diag ~25% (the real Test predictor) — the single biggest *process* lesson in this project.

### 🎓 Capstone (the payoff for "read papers & judge methods")
Re-read `investigations/nemotron_vs_zipformer.md` end-to-end, then write your own one-page verdict:
1. Was the H5 "PEFT fixes overfitting" bet reasonable ex-ante? What would have changed your prior?
2. What's the *cheapest* experiment that would confirm/refute **H7** (the streaming-export gap)?
3. Given all evidence, is more Nemotron GPU justified, or is banking A1 + shipping beam-4 correct?

If you can answer those with references to the papers above, you've hit the goal — and you can direct
the next phase (and vet any Fable proposal) without deferring to me.

---

## Resource quick-list (all free)
- **Intuition (🎥):** 3Blue1Brown (NN + transformers), StatQuest, Valerio Velardo (audio), Karpathy Zero-to-Hero.
- **Written (📄):** Illustrated Transformer (Alammar), Olah LSTMs, Hannun CTC (Distill), Lugosch Transducers,
  NeMo/icefall streaming docs, HuggingFace PEFT docs.
- **Papers (📄ˑ):** Attention Is All You Need · Graves RNN-T · Conformer · Zipformer · LoRA ·
  RNN-T-OOD · ILME · FastEmit.
- **Hands-on (💻, last):** Karpathy *build GPT* + *micrograd*; then a small notebook that runs our own
  `streaming_zipformer/model.py` `accept_chunk` loop so streaming stops being abstract.

## Adaptation notes
- Time-boxed? Do M0(skim) → M1 → **M2 (all)** → M3c + M4. Skip M3a/M3b reading if needed; the
  investigation doc summarizes their conclusions.
- Want the *fastest* path to the roadblock only: M1 (attention) → M2b (RNN-T) → M2d (streaming) → H7.
