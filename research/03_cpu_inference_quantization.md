# Axis 3 — CPU-Efficient Inference & Quantization

Lens: what makes a streaming transducer real-time on CPU within 15000 s. Tags as before.
Primary source: **"Pushing the Limits of On-Device Streaming ASR"** (arXiv 2604.14493) — the most
Track-2-relevant paper found. Plus sherpa-onnx docs.

## 1. Headline result that updates our priors `[V-lit]` (2604.14493)
A **600 M cache-aware streaming conformer-transducer (NVIDIA Nemotron Speech Streaming)**, via a custom
**3-graph ONNX** pipeline + quantization, runs on CPU (AMD EPYC 7V12, **32 cores**, AVX2):
- ONNX FP32: **6.73× real-time** · int4 k-quant: **7.20×** · int4-mixed: 7.15×.
- int8 k-quant: 1.28 GB, **8.01% WER** vs **8.03% FP32** (≈ lossless). int4: 0.67 GB, +0.17% abs.
→ **600 M is NOT automatically CPU-infeasible.** My earlier "C3 out" was over-stated. Feasibility
hinges on **(cores × ONNX/quant engineering)**, not param count alone.
⚠️ **Caveat**: 6.7× is on **32 cores**. RTFx scales ~linearly with cores.

### 1a. VERIFIED eval-worker spec (Session 3, from Q's Codabench ingestion log 785065) `[V]`
- **~24-core AMD EPYC-Milan, ~226 GiB RAM, no swap.** Local Docker Desktop (8 cores / 3.8 GiB / emulated
  amd64) is **smoke-test only**, not representative.
- ⚠️ **Not a guaranteed spec**: one observation; Codabench workers are organizer/queue-specific and can
  change. Plan for it, don't depend on it.
- **Design posture**: target **real-time on ≈8 cores with int8 ONNX** (safe under any plausible worker),
  exploit 24 cores when present. Under this posture:
  - C1 Zipformer-66M: trivially real-time on 8 cores. ✅
  - C2 FastConformer 32–114M int8 ONNX: comfortable on 8, large margin on 24. ✅
  - C3 600M: marginal on 8, comfortable on ~24 — bets on the spec holding. ⚠️
- **Time-budget caveat (15000 s)**: the *streaming* pass is **real-time-paced** → its wall time is bounded
  below by `total_test_audio / parallelism`, independent of model speed. On 24 cores this fits easily; on
  8 it could be tight for a large test set. **Still need: total test1 audio duration + whether ingestion
  parallelizes the streaming pass.** `[?]`

## 2. The decisive streaming-degradation finding `[V-lit]` (2604.14493)
WER when forced to stream/chunk vs batch:
- **Nemotron cache-aware (7,10,7): 7.28% streaming vs 7.07% batch → +0.21% only.**
- **Parakeet-TDT chunked: 9.22% vs 6.32% → +1.94 abs (+46% rel).**
- Qwen3-ASR-1.7B chunked: 10.45% vs 5.90% → +77% rel.
- Conformer-Transducer-XL chunked: 11.06% WER, RTFx 1.27.
→ **Streaming-native (cache-aware, trained-for-streaming) models keep accuracy; offline SOTA models
collapse when chunked.** ⇒ For Track 2 we must pick/fine-tune a **streaming-trained** model, NOT
repurpose the SAPC1 winner's offline Parakeet-TDT. This is the single biggest architecture lesson.

## 3. Quantization recipe that works `[V-lit][V-web]`
- **Quantize the encoder only** (≈95% of params); keep decoder+joiner FP32. `[V-lit]`
- **3-graph ONNX decomposition** (encoder / decoder / joiner) → per-component quant + graph opt;
  **fuse multi-head attention into one kernel** for big speedups. `[V-lit]`
- int8 ≈ lossless; int4 k-quant ~+0.17%. sherpa-onnx: int8 zipformer ≈ lossless, 2–4× speed, ~75% smaller. `[V-web]`
- Greedy RNN-T decode (no beam) + zero-copy ring-buffer cache + native log-mel = remove overhead. `[V-lit]`

## 4. Our baseline's CPU standing
- icefall streaming Zipformer = **66 M**; sherpa-onnx CPU RTF ~0.11–0.19 (i.e. **5–9× real-time**) `[V-web]`;
  Q already ran it in-pipeline `[V]`. int8 export ≈ lossless. → **Huge CPU headroom**, the safe anchor.

## 5. Engineering-cost reality for the FastConformer/Nemotron path
- The 6.7× result used a **custom ONNX pipeline**, not an off-the-shelf export. sherpa-onnx cache-aware
  FastConformer export is **WIP** (GH #790, #2177). `[V-web]` So C2/C3 carry real integration cost:
  we'd either build/borrow the 3-graph ONNX runtime or run NeMo/PyTorch CPU (slower). Budget for this.
- Pre-installed runtime is **PyTorch 2.5.0** (Docker). ONNX Runtime would be an added dep via setup.sh. `[V]`

## Axis-3 takeaways
1. **int8 (encoder) ONNX is the default deployment** for whatever we pick — near-lossless, faster, smaller.
2. **Eval-container core count is the gating unknown** for admitting bigger models — get it.
3. Zipformer-66M has the most CPU margin; a 600M streaming model is *possible* but only with the custom
   ONNX/quant stack and enough cores — a real engineering bet, not free.
4. Choose **streaming-trained** encoders (Axis 2/3 §2) — offline models lose ~2 WER points when chunked.
