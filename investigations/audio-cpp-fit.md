# Investigation: Could `0xShug0/audio.cpp` help SAPC2 Track 2?

**Date:** 2026-07-14
**Method:** read-only source inspection of a shallow clone. **No build, no pod, no run.**
Every performance/correctness claim here is UNVERIFIED (static reading only).
**Do not treat "streaming Nemotron ASR: released" as a working result** — that came from
the README; see FACTS below for what the code actually does.

---

## Why we looked
Our Nemotron arm died on the **H7 streaming-export gap**: we could never export the
NeMo cache-aware RNNT into a real chunk-by-chunk streaming path (see
`nemotron-vs-zipformer-roadblock`, `parakeet-offline-not-streaming`). audio.cpp's README
advertises a *streaming Nemotron ASR* in pure C++/ggml. If real, it sidesteps the export
wall entirely. This investigation checks whether it's real and whether it fits our
CPU-only, latency-scored, Python-`model.py` submission contract.

---

## FACTS (verified by reading source)

- **License = Apache 2.0** (`LICENSE`, "Copyright 2026 ShugoAI LLC"). Permissive →
  legally bundlable in a Codabench submission. ✓
- **The Nemotron ASR implementation is real, not a stub.** `src/models/nemotron_asr/`:
  encoder.cpp (71 KB), decoder.cpp (26 KB), session.cpp (21 KB), weights.cpp, frontend.cpp.
- **The cache-aware streaming machinery genuinely exists:**
  - `encoder.h`: `NemotronEncoderStreamState`, `make_stream_state()`,
    `encode_stream_chunk(features, prompt_id, lookahead, state)` — persistent encoder
    state carried across chunks.
  - `weights.h` includes `streaming_conv_modules.h` — cache-aware (streaming) conv, the
    hard part of a streaming conformer.
  - `decoder.cpp`: `decode_streaming(options, chunk_producer, on_text_delta)` — pull-based
    RNNT decode loop that emits per-token text deltas.
  - `session.cpp::run_streaming_audio` (lines 305-358): slices a waveform into
    lookahead-sized chunks, calls `encode_stream_chunk` with ONE persistent `stream_state`,
    feeds them to `decode_streaming` via a `next_chunk` pull lambda + `on_text_delta`
    callback. **This is a real cache-aware streaming RNNT — the exact thing H7 blocked.**
- **BUT the shipped streaming *session* does NOT emit partials during ingestion:**
  - `streaming_policy()` (session.cpp:398): `output = StreamingOutputKind::FinalResult`.
  - `process_audio_chunk()` (session.cpp:424-437): **only appends audio to a buffer
    (`streaming_audio_`) and returns an empty non-final event.** No encode, no partial.
  - `finish_stream()` → `finalize()` (session.cpp:439): runs `run_streaming_audio` over
    the *entire buffered waveform*; the `on_text_delta` partials fire only during this
    final pass, i.e. AFTER all audio is ingested.
  - Same pattern across ALL of audio.cpp's ASR models (higgs_audio_stt, vibevoice_asr all
    use `FinalResult`). Only a TTS (supertonic) uses incremental `PullEvents`.

---

## What this means for OUR metrics (contract §4, latency)

- **TTFT** = first_non_empty_partial − audio_start. As shipped, the first partial appears
  only at `finish_stream`, so TTFT ≈ full audio duration. **Bad — no better than offline.**
- **TTLT** = final_visible − audio_end. Finalize decodes the whole buffer after audio end.
- **Conclusion: as-shipped, this is "streaming input, batch output" — it does NOT solve
  the low-latency streaming problem SAPC2 Track 2 scores.**

The critical nuance: `run_streaming_audio`'s `next_chunk` is a **pull-based lambda** and
`decode_streaming` already supports incremental emission. Making it TRUE real-time is a
"feed `next_chunk` from live `process_audio_chunk` input instead of a buffer" rewire —
**NOT** a re-solve of the export/cache problem. The hard, previously-blocking part
(cache-aware conv + streaming RNNT loop in ggml) is already done here.

---

## THEORIES / UNKNOWNS (all unverified — would need a build+run to settle)

1. **CPU latency is unknown.** ggml/CPU is the whisper.cpp lineage (promising), but README
   optimizes for CUDA and calls CPU a "portability backend, potentially lower performance."
   Must measure on the real 24-core EPYC worker (`eval-worker-cpu-confirmed`).
2. **Weight provenance / accuracy on dysarthric speech unknown.** Which NeMo checkpoint
   maps to their GGUF, and whether ggml numerics match NeMo closely enough, is unverified.
   No dysarthric-speech accuracy evidence.
3. **Integration cost is high.** Our contract is a Python `Model` class inside
   `xiuwenz2/sapc2-runtime` (PyTorch cu124). Using audio.cpp means: compile a C++ binary in
   `setup.sh`, bridge 100 ms `accept_chunk` chunks over subprocess/pipe, parse partials
   back — a new fragile surface, offline-packaged (`submission-offline-packaging`).
4. **The real-time rewire is non-trivial** even though the primitives exist.
5. **Build complexity on the runtime image** (ggml, external deps) unmeasured.

---

## Verdict

- **Not a drop-in.** As shipped it does not emit real-time partials → does not move our
  TTFT/TTLT. Do NOT package it as-is expecting streaming latency wins.
- **But it is the first concrete evidence that cache-aware streaming Nemotron RNNT can run
  outside the NeMo/ONNX export path** — the exact wall (H7) that killed our Nemotron arm.
  The blocking primitive is solved here under Apache 2.0.
- **It is a spike candidate, not a decision.** Value = potentially reviving the higher-CER-
  ceiling Nemotron path (`nemotron-higher-ceiling`) if (a) CPU latency is acceptable and
  (b) the session is rewired to emit incrementally.

## Recommended next gate (only if Q wants to pursue Nemotron again)
Cheap, local/Docker, no pod first:
1. Build audio.cpp CPU-only + run its `tests/nemotron_asr` warm bench → get a GGUF Nemotron
   transcribing *anything* on CPU and measure per-chunk encode latency. Decision metric:
   is per-100ms-chunk encode << real-time on CPU? If no → drop, latency can't work.
2. Only if yes: prototype the real-time rewire (live `process_audio_chunk` → `next_chunk`)
   and measure TTFT on a handful of Dev files via the REAL `local_decode.py` gate
   (`validate-against-real-harness`) — never a proxy.

---

## Can we finetune it? (added 2026-07-14)

**Not inside audio.cpp — it is ggml inference-only (no autograd/optimizer/loss).** But that
doesn't block finetuning; finetuning was never supposed to happen in the runtime. The path:

1. **Finetune upstream in NeMo/PyTorch** on the base checkpoint
   `nemotron-3.5-asr-streaming-0.6b` (NVIDIA Nemotron 3.5 ASR RNNT, 0.6B, natively-streaming).
2. **Export finetuned weights → `model.safetensors`.**
3. **Convert → GGUF:** `audiocpp_gguf --input model.safetensors --output model.gguf --type q8_0`.
   Verified: `tools/gguf_convert.cpp` + `docs/asr.md:140-146`. The converter is a **generic,
   name-based tensor mapper** (`--input [namespace=]<weights>`), NOT hardwired to their weights;
   their own released GGUF was produced from this exact `safetensors → gguf` path.
4. **Stream in audio.cpp.**

**Unverified glue:** (a) obtaining the base checkpoint; (b) that a NeMo→safetensors re-export
preserves the exact tensor NAMES audio.cpp's loader expects (if not, write a name-map shim).
Standard finetuning doesn't change layer names, so re-export should convert as-is.

---

## Does audio.cpp actually solve our Nemotron ROOT problem? (added 2026-07-14)

Cross-checked against `investigations/nemotron_vs_zipformer.md` (F8/H7) and the primary
`research/` provenance (research/41, research/25, research/19, research/18).

### Provenance of the two numbers — and the CORRECTION

Earlier drafts of this file called 18.46% "offline / full context." **That is wrong.** Both
numbers use roughly the SAME limited context (`[70,1]` = 70 left, **1** right). They differ in
*how* that limited context is computed, not in how much context there is.

| | **18.46% "offline"** | **28.19% "streaming"** |
|---|---|---|
| Produced by | NeMo `transcribe()` at `[70,1]` (research/41:41, research/25:22) | ONNX cache-aware export → `local_decode.py` (research/41:41-42) |
| Pass structure | **one forward over the whole utterance** | utterance **chopped into 100 ms chunks**, one forward each |
| Limited context via | **attention mask** on the full sequence | **caches** (conv left-ctx + attn KV) stitched across chunk seams |
| Chunk-boundary effects | none (single pass) | **all of them** (cache init, seam stitching, normalization) |
| Faithful to deployment? | **NO — a proxy** (research/25:55) | **YES — the path the challenge scores** |

Our own audit nails what `transcribe()` is (research/19:12):
> "transcribe() is **context-faithful** (applies att mask) but **NOT chunk-boundary-faithful**
> (single pass, no 100ms cache carryover)."

So the ~10-point gap is **masked-single-pass vs chunked-with-caches**, NOT full-vs-limited
context. This **downgrades** the "fundamental right-context info-loss" hypothesis — the proxy
already runs at right-context = 1. **Plus a confirmed own-goal:** research/19:32 flags we were
"**mixing [70,1]-train with [70,6]-deploy**" — a train/deploy context-regime mismatch on our
side that could explain a chunk of the gap with no NeMo bug and nothing audio.cpp would fix.

### Revised mechanism ranking (what the 10 points actually are)
1. **Train/deploy config mismatch (our own error).** `[70,1]`-train vs `[70,6]`-deploy, and
   possibly finetuning that de-adapted the base model's cache-aware behavior. Fixable in NeMo;
   **not** audio.cpp's to fix.
2. **Chunk/cache computation discrepancy (the real "export bug" bucket).** Conv cache init, seam
   stitching, per-chunk normalization — the masked path and the chunked path diverge numerically.
   **This is the part an independent reimplementation (audio.cpp) could avoid OR expose.**
3. **Fundamental right-context info-loss.** Now WEAK — both numbers are already limited-context.

### Verdict — audio.cpp DIAGNOSES; it does not certify, and it does not train
- It is a **third, independent chunked-cache implementation** with a `lookahead_tokens` knob. It
  cannot give the model back missing context (mechanism 3) and cannot fix a training/config
  mismatch (mechanism 1 — it can't train). Its leverage is purely on mechanism 2.
- **It cannot certify a deployment CER.** An audio.cpp number is ANOTHER proxy, like the 18.46%.
  Only `local_decode.py` + the official scorer count for shipping (`validate-against-real-harness`).
- **Genuine win regardless:** removes the NeMo/ONNX export toolchain and gives a clean **CPU
  streaming runtime** — a deployment/packaging win, not an accuracy win.

---

## ⚠️ Diagnostic ANSWERED 2026-07-22 (research/45) — audio.cpp not needed for this
The question this spec was built to answer (is the collapse intrinsic chunked-cache vs our ONNX export?)
was settled WITHOUT audio.cpp, using NeMo's own streaming infer as the independent impl. Base A/B/C @
[70,1] Dev_diag: transcribe **43.4%** ≈ NeMo-native streaming **43.8%** ≈ our ONNX export **43.4%**.
⇒ the pipeline is FAITHFUL; there is no export-vs-intrinsic collapse to isolate. audio.cpp would add a
third faithful impl and change nothing. The spec below is retained for provenance only.

## Cheap diagnostic experiment spec (added 2026-07-14)

**Question it answers:** is the ~10-point streaming collapse (a) intrinsic to correct chunked-cache
streaming, or (b) specific to OUR ONNX export + `[70,1]/[70,6]` config mismatch? This is the exact
question H7 left open. Runnable BEFORE any finetuning or any pod-scale spend.

**Design — one existing checkpoint, three decode paths, same audio, compare CER.**
Reuse the **v1 finetuned Nemotron weights** (best artifact, research/20/24) and the **same eval set**
all three prior numbers used: severe-enriched **`Dev_diag`** (speaker-disjoint). No new training.

| Arm | Path | Purpose | Prior anchor |
|---|---|---|---|
| A (reference) | NeMo `transcribe()` @ `[70,1]` | reproduce the offline proxy | 23.6% (v1) / 18.46% (PEFT) |
| B (reference) | our ONNX export → `local_decode.py` @ matched ctx | reproduce the faithful collapse | 28.19% |
| C (**new**) | audio.cpp GGUF streaming @ **matched** `lookahead_tokens` | independent chunked-cache impl | — |

**Critical control:** match the context budget across B and C (set audio.cpp `lookahead_tokens`
to the same right-context B deploys at). If B and C are run at different lookahead, the comparison
is void — that is the exact `[70,1]/[70,6]` mistake, repeated.

**Steps (local / Docker first; no paid pod until a build is proven to run):**
1. Convert v1 weights → safetensors → GGUF: `audiocpp_gguf --input v1.safetensors --output
   v1.gguf --type q8_0` (verify tensor-name mapping; write a name-map shim if the loader rejects keys).
2. Build audio.cpp **CPU-only**; sanity-check with `tests/nemotron_asr` warm bench (must transcribe
   *anything* correctly before trusting CER).
3. Run arm C on `Dev_diag` at matched lookahead → CER via our scorer wrapper (NOT audio.cpp's).
4. (Arms A, B are already-closed numbers — only re-run if the checkpoint/env drifted.)

**Decision rule (write success criterion FIRST):**
- **C ≈ 28% (within ~2 pts of B):** the collapse is intrinsic to chunked-cache streaming at this
  context → **audio.cpp does NOT rescue accuracy.** Real fix = cache-aware finetuning in NeMo. Do
  not pursue audio.cpp for CER; at most a deployment-ergonomics option.
- **C ≈ 18–22% (near arm A):** the loss was in OUR ONNX export/config, not the regime → **audio.cpp
  genuinely rescues the Nemotron path.** Escalate to: (i) the real-time session rewire (partials
  during ingestion), then (ii) certify on the REAL `local_decode.py` + official scorer before any
  submission — an audio.cpp proxy is NOT sign-off.
- **C between 22–28%:** partial — decompose (is the residual the [70,1]/[70,6] mismatch? re-run B at
  strictly matched context first).

**Stop conditions / cost gate:**
- If step 1 (GGUF convert) or step 2 (CPU build + warm bench) fails, STOP and report — do not
  proceed to CER. The diagnostic is only meaningful once audio.cpp demonstrably runs the model.
- Per-100ms-chunk CPU encode latency from step 2 is a SECOND decision gate: if it is not well
  below real-time on the 24-core worker (`eval-worker-cpu-confirmed`), the latency case is dead
  regardless of CER, and arm C's accuracy result is moot for shipping.
- Everything here is a **diagnostic**, never a submission. Shipping still requires the real harness.

---

## Files touched
- Created: `investigations/audio-cpp-fit.md` (this file); updated 2026-07-14 with finetune +
  root-cause-fit analysis.
- Clone lives in scratchpad only (not added to repo).
