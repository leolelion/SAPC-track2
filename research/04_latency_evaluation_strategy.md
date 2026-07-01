# Axis 4 — Latency & Evaluation Strategy (how to actually win the Pareto)

Lens: the metric is **mean(TTFT_p50, TTLT_p50)** vs **CER** (min-two-refs). Understanding exactly how the
scoring rewards behavior is as important as the model. Repo mechanics = `[V]` (read from code).

## 1. Exactly how Track 2 scores (verified from repo)
- **CER (primary)** = min over {with-disfluency ref, without-disfluency ref}, per-utt, clipped at 1.0,
  sclite-aligned, HF-style normalization. `[V compute_metrics.py]`
- **TTFT** = `first_non_empty_partial_time − (audio_send_start + mfa_speech_start)`. Empty partials don't
  count. Fallback = last event time if all empty. `[V compute_latency.py]`
- **TTLT** = `final_visible_time − audio_end_oracle_time` (counted only if ≥ 0). `[V]`
- Pareto FoM = **mean(TTFT_p50, TTLT_p50)**. `[V Track2.md]`
- Pass 1 (batch, multiprocess) → CER. Pass 2 (real-time 100 ms paced, 2-thread) → latency. `[V]`

## 2. What this rewards (design implications)
- **TTFT**: emit a NON-EMPTY partial as EARLY as possible after true speech start. Because TTFT is
  measured from `mfa_speech_start` (true speech onset), emitting before/at onset is ideal; a model that
  waits a full big chunk before its first token pays. → favors **small first-chunk lookahead** and an
  **eager emission policy** (don't suppress the first token). `[V]`+`[K]`
- **TTLT**: finalize FAST after audio ends. The baseline pads **0.3 s silence** then flushes → that tail
  is pure TTLT cost. Minimize the flush tail subject to CER. Greedy final decode is fastest. `[V model.py]`
- **CER under streaming**: must not collapse vs offline (Axis 3 §2). Cache-aware streaming model keeps CER.
- **CER vs latency are coupled via chunk size**: smaller chunk → lower TTFT but worse CER. We need the
  Pareto curve, not a point. `[V]`+`[K]`

## 3. Emission-latency techniques from the literature
- **FastEmit** (arXiv 2010.11148): sequence-level emission regularization that encourages earlier token
  emission → lower TTFT without extra params. Train-time. `[V-web]`
- **StableEmit** / selection-probability discount → reduce emission latency for monotonic attention. `[V-web]`
- **Alignment-Restricted RNN-T** (2011.03072): constrain token alignment to reduce emission delay. `[V-web]`
- **Partial Emission Interval (PEI)**: instability decreases logarithmically as PEI lengthens — i.e.
  emitting partials *too often* increases flicker/instability; there is a stability/latency trade. `[V-web]`
  (Stability isn't directly scored here, but erratic partials can hurt the first-non-empty timing if early
  partials are empty/garbage.) `[K]`
- **Endpointing**: detect end-of-speech to finalize quickly → directly lowers TTLT. `[V-web]`

## 4. Concrete latency levers we control (no retrain needed first)
1. **Chunk size / left-context** in config.yaml → primary CER↔TTFT trade. Sweep it.
2. **Emit the first non-empty partial ASAP** — ensure `accept_chunk` calls the callback as soon as any
   token exists (don't gate on a full beam). `[V model.py _decode_available]`
3. **Shrink the input_finished flush tail** (0.3 s → as low as CER tolerates) → lower TTLT.
4. **Greedy vs beam**: greedy for streaming pass (fast final), maybe beam only where budget allows.
5. **(Retrain) FastEmit-style regularization** if TTFT is the binding constraint after sweeps.

## 5. Strategy: map the Pareto, then pick operating point(s)
- The prize splits among ALL systems on the test2 Pareto frontier → submitting **multiple points**
  (e.g. a low-latency/decent-CER point AND a higher-latency/best-CER point) can be rational. `[V Track2.md]`
- So we don't need a single model; we need a **frontier**. Same fine-tuned weights at different chunk/
  tail/decoding settings already yields several frontier points cheaply.

## Axis-4 takeaways
1. Latency is heavily **policy-driven** (emission timing, flush tail), not only model size — cheap wins.
2. TTFT favors **eager early partials + small first lookahead**; TTLT favors **fast endpoint + short tail**.
3. Keep CER from collapsing under streaming → cache-aware/streaming-trained model (ties to Axis 3).
4. Aim to produce a **Pareto frontier of configs**, not one submission. FastEmit is the retrain-level lever.
