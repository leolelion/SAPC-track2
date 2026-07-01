# RESEARCH_PLAN.md — PhD-style literature survey for SAPC2 Track 2

> Goal: understand the **whole** landscape before committing model choices, so our candidate
> shortlist is *derived* from evidence, not guessed. Problem = **CPU-only, real-time streaming ASR
> for dysarthric speech, maximizing the CER × latency Pareto** under a 15000 s budget.
>
> Method: start general → read latest literature (2023–2026) → document findings per axis in
> `research/*.md` → synthesize into a candidate decision in `research/05_synthesis_and_candidates.md`.
> Epistemic tags: `[V-lit]` paper-verified, `[V-web]` web-verified, `[K]` model knowledge, `[?]` open.

## Why this plan (the questions that actually decide the competition)
Winning needs a model that is simultaneously: (1) **accurate on dysarthric speech** (low CER,
min-two-refs incl. disfluency), (2) **truly streaming** (low TTFT/TTLT), (3) **real-time on CPU**
(RTF<1, fits 15000 s). Most literature optimizes ≤2 of these; the edge is in the intersection.

## Research axes (each → a `research/NN_*.md` file)
1. **Dysarthric / pathological speech ASR** — what raises accuracy: fine-tuning regimes, augmentation,
   synthetic/TTS data, voice conversion, speaker/severity/etiology adaptation, GER/LLM correction,
   handling disfluency/repetition. Which transfer to a *small streaming* model?
2. **Streaming ASR architectures (2023–2026)** — Zipformer streaming, FastConformer cache-aware,
   Emformer, chunked-attention transducers, TDT/RNN-T/CTC, U2++/WeNet, endpointing. Latency knobs.
3. **CPU-efficient inference** — sherpa-onnx, ONNX Runtime, int8/int4 quant, distillation, pruning,
   ggml; measured CPU RTF for streaming transducers; the on-device compact-ASR papers.
4. **Latency & evaluation strategy** — how TTFT/TTLT/min-two-refs CER actually reward behaviors;
   partial-emission & endpointing policy; chunk-size vs CER vs latency trade curves.
5. **Synthesis & candidate decision** — combine 1–4 into a Pareto-aware shortlist with feasibility
   evidence + an experiment plan. This supersedes the earlier (under-verified) C1/C2/C3 list.

## Progress tracker
- [x] Axis 1 — dysarthric ASR (deep dive beyond SAPC1) → `research/01_dysarthric_asr.md`
- [x] Axis 2 — streaming architectures → `research/02_streaming_architectures.md`
- [x] Axis 3 — CPU inference / quantization → `research/03_cpu_inference_quantization.md`
- [x] Axis 4 — latency & evaluation strategy → `research/04_latency_evaluation_strategy.md`
- [x] Axis 5 — synthesis + revised candidate shortlist → `research/05_synthesis_and_candidates.md`
- [x] Update PLAN.md Phase 3 with the evidence-derived shortlist
- [x] Eval-worker spec resolved (Session 3): ≈24-core EPYC-Milan / 226 GiB (not guaranteed) → see research/03 §1a
- [ ] **Next: get SAP data layout on host**, then start exp_001 (C1 FT). C2 promoted to strong play; C3 = stretch.

## Headline conclusions (see research/05 for the full chain)
1. Lever = full FT of a **streaming-native** supervised transducer on all SAP + standard aug + ckpt-avg.
2. **Streaming-native is non-negotiable**: offline SOTA loses +1.9–77% when chunked; cache-aware loses ~0.2%.
   → do NOT reuse the SAPC1 winner's *offline* Parakeet-TDT for Track 2.
3. CPU feasibility = cores × ONNX/quant, not param count. 66M Zipformer = safe anchor; 114M cache-aware
   FastConformer = higher-ceiling challenger gated by a CPU-RTF benchmark; 600M parked.
4. Latency is policy-driven → a frontier of Pareto points from one fine-tuned model; submit ≥2.

## Sources already in hand (Session 2)
- SAPC1 challenge overview (arXiv 2507.22047) + 1st-place Parakeet-TDT system (Interspeech 2025-1484).
  Key: all top teams fine-tuned transducers/Whisper; recipe in `RESEARCH_NOTES.md §7`.
- sherpa-onnx CPU RTF for zipformer ~0.11–0.19; FastConformer CPU real-time unproven (GPU-only RTF<1).

## Working log
Session 3 (2026-06-19): plan created; beginning Axis 1+2 literature sweep.
