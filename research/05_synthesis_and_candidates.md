# Axis 5 — Synthesis & Candidate Decision (evidence-derived)

This supersedes the earlier under-verified C1/C2/C3 list in RESEARCH_NOTES §7.4. Everything here is
traceable to Axes 1–4. Decision lens: **CER × latency Pareto, real-time on CPU, ≤15000 s.**

## The big picture (what the literature actually says)
1. **Fine-tuning a supervised streaming transducer on all SAP data is THE lever.** SSL (wav2vec2/HuBERT)
   underperforms on SAP; LLM-GER and 1B+ models win Track 1 but are disqualified by Track-2 CPU/latency.
   `[Axis1]`
2. **The model MUST be streaming-native (cache-aware / trained-for-streaming).** Decisive evidence:
   forcing offline SOTA to chunk costs **+1.9 WER abs (Parakeet-TDT) to +77% rel (Qwen)**, while a
   cache-aware streaming model costs **+0.2%**. ⇒ Do NOT repurpose the SAPC1 winner's *offline*
   Parakeet-TDT for Track 2. `[Axis3 §2]`
3. **CPU feasibility = cores × ONNX/quant engineering, not param count.** A 600 M streaming model hits
   6.7× real-time on 32 cores with custom 3-graph ONNX; a 66 M Zipformer hits ~5–9× and is already
   proven in our pipeline. **Eval-container core count is the gating unknown.** `[Axis3 §1]`
4. **Latency is largely policy** (eager first partial, short flush tail, chunk size) → many Pareto points
   from ONE fine-tuned model. The prize rewards the whole frontier → submit multiple points. `[Axis4]`
5. **Deploy int8 (encoder) ONNX** — near-lossless, faster, smaller, for whatever we pick. `[Axis3 §3]`

## Revised candidate shortlist (with evidence + risk)

### C1 — icefall streaming Zipformer-66M, full FT on SAP  → **DO FIRST (anchor)**
- Why: streaming-native (no chunk-degradation), **proven CPU real-time** (5–9× RT) and **already run by
  Q in-pipeline**, int8≈lossless, and icefall has a **documented finetune recipe** (LibriSpeech→custom;
  full FT or adapters; LR ≈ 0.0045 = 1/10 base). `[Axis2/3, V-web]`
- Role: cleanest measurement of *how much in-domain FT buys us*; zero feasibility risk; sets the anchor
  CER and the latency profile we already trust.
- Recipe: full FT on all SAP, speed-perturb 0.9–1.1 + SpecAugment, forced-align long-audio segmentation,
  4-checkpoint averaging (SAPC1-winner recipe, `RESEARCH_NOTES §7`). `[Axis1]`
- Consider a stronger Zipformer **start checkpoint** than the 2023 LibriSpeech one (e.g. Kroko/larger
  multidataset Zipformer) — but check **license** (Kroko community = CC-BY-SA) for competition use. `[?]`

### C2 — NeMo cache-aware streaming FastConformer-Hybrid (medium ~32M / large ~114M), full FT on SAP → **HIGHER-CEILING CHALLENGER, gated**
- Why: **cache-aware → offline≈streaming accuracy** (the property that matters most, +0.2% chunk cost);
  hybrid **Transducer+CTC** decoders; **same FastConformer family that won SAPC1**; multi-latency
  checkpoints (80/480/1040 ms) give Pareto points for free; **fine-tunable in NeMo**; medium size is
  CPU-friendlier than large. `[Axis2/3, V-web]`
- Risks: (a) **CPU RTF** unproven at our core count — medium size mitigates; (b) cache-aware ONNX export
  is **custom/WIP** for sherpa-onnx (GH #790, #2177) → real engineering cost; (c) adds NeMo + ONNX deps.
- **Gate before spending GPU-hours**: benchmark a *pretrained* medium & large streaming FastConformer's
  **CPU RTF + offline CER on SAP Dev** on the eval container. Admit to FT queue only if RTF < ~0.7 and
  zero-shot CER ≲ Zipformer's. `[Axis3]`

### C3 — 600 M Nemotron streaming → **STRETCH (de-parked, but spec-risky)**
- Eval worker observed ≈ **24-core EPYC-Milan / 226 GiB** `[V, Session3]` → 600 M is *plausibly* real-time
  with custom int4 ONNX. But the spec is **one observation, not guaranteed**, and 600 M is marginal on
  ≤8 cores. Pursue only after C1/C2, and only if we confirm the worker stays ≥~16 cores. `[Axis3 §1a]`

> **Spec update (Session 3)**: verified eval worker ≈ 24-core AMD EPYC-Milan, 226 GiB RAM (Codabench log
> 785065; NOT guaranteed). Design posture = **real-time on ≈8 cores int8 ONNX** for safety. This
> **promotes C2 to a strong play** (114M is comfortable even on 8 cores) and de-parks C3 as a stretch.
> Open: total test1 audio duration + streaming-pass parallelism vs the 15000 s cap. `[Axis3 §1a]`

### Explicitly rejected for Track 2
- Offline Parakeet-TDT-1.1B / Qwen / Whisper-large (chunk-degrade and/or not real-time on CPU);
  SSL wav2vec2/HuBERT (weak on SAP); LLM-GER post-pass (kills TTLT/budget). `[Axis1/3]`

## Decision & experiment sequence
1. **EDA + speaker-disjoint internal val** (Phase 2) — never tune to train/dev *text* (eval is "unshared").
2. **exp_001 = C1 full FT** → CER/WER + TTFT/TTLT vs baseline exp_000. *The* go/no-go on the FT lever.
3. **In parallel, cheap gate for C2**: pretrained streaming FastConformer **CPU-RTF + zero-shot SAP CER**.
   If it clears, **exp_002 = C2 full FT**; compare ceiling vs C1.
4. **Latency frontier** (Axis 4): from the winning weights, sweep chunk/left-context/flush-tail/decoding
   → Pareto points. Add **FastEmit** only if TTFT is binding.
5. **Deploy int8 ONNX**; verify 15000 s budget on full test1 size; submit ≥2 Pareto points.

## Top open unknowns that change the plan (need answers)
- **Eval-container CPU core count + RAM** — gates C2/C3 admissibility. (Ask Q / inspect Docker image.)
- **SAP data layout on host** — manifests/columns, total Train/Dev duration → implied RTF budget.
- **Zipformer start-checkpoint license** (Kroko CC-BY-SA vs icefall LibriSpeech) for competition use.
- **Streaming FastConformer zero-shot CER on SAP** — is its ceiling actually above Zipformer's? (Gate measures it.)
