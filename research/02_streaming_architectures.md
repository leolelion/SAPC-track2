# Axis 2 — Streaming ASR Architectures (2023–2026)

Lens: maximize CER while keeping TTFT/TTLT low AND staying real-time on CPU. The architecture sets the
ceiling on all three. Tags as in Axis 1.

## 1. The streaming families
- **Transducer (RNN-T / Conformer-T / Zipformer-T)** — frame-synchronous, naturally streaming, strong
  with limited context. Dominant for low-latency. Our baseline (Zipformer-T) is here. `[V]`
- **TDT (Token-and-Duration Transducer)** — RNN-T that also predicts token *duration* → skips frames →
  **up to 2.8x faster decoding**, frame-asynchronous; Parakeet-TDT tops HF leaderboard RTFx by ~3x. `[V-web]`
  SAPC1 winner used Parakeet-**TDT**. Frame-skipping is attractive for **CPU latency**. `[V-web]`
- **CTC** — fastest/simplest decode, no autoregressive decoder, but weaker (no internal LM); SAP zero-shot
  evidence says non-autoregressive decoders underperform on dysarthric. `[V-lit]` Possible as a fast
  first-pass / endpointer, not the main model.
- **Cache-aware streaming Conformer/FastConformer** — reuse encoder cache across chunks, configurable
  chunk/lookahead (80/160/320/560/1120 ms); "up to 17x lower latency than naive chunking". `[V-web]`
  FastConformer = 2x faster train/infer than Conformer. `[V-web]` But attention/conv lean on GPU. `[V-web]`
- **Attention-free / SSM / linear-attention streaming** (2025–2026): summary-mixing linear-time conformer
  (ICASSP'25), selective-state-space dual-mode ASR (Interspeech'25), "Do we really need self-attention
  for streaming ASR?" (2601.19960). Motivation = **kill the quadratic attention cost** — directly relevant
  to CPU. `[?]` deep-read pending.

## 2. Latency mechanics (the knobs)
- **Chunk size / lookahead (right context)** = the master TTFT knob: smaller → lower latency, worse CER.
  Our baseline = 16 frames (~320 ms) chunk, 128 left-context frames. `[V config.yaml]`
- **Left context** = past frames attended; more → better CER, modest cost. `[V]`
- **Decoder context size** (transducer predictor history) — small (e.g. 2) is standard; interacts with
  repetition fidelity (Axis 1 §6). `[V config.yaml]`
- **Decoding**: greedy (fast) vs beam/mAES (better CER, more CPU). TDT frame-skip reduces steps. `[V]`
- **Partial-emission policy** — TTFT keys on *first non-empty* partial → emit early. (Axis 4.) `[V]`
- **Endpointing / input_finished tail** — flush latency adds to TTLT; tune tail length. `[V model.py]`

## 3. SOTA reference points (2025–2026)
- **NVIDIA Nemotron Speech Streaming 0.6B** (June 2026) — cache-aware FastConformer-RNNT, configurable
  low-latency chunks, punctuation/cap. 600M → GPU-oriented; CPU real-time questionable. `[V-web]`
- **Parakeet-TDT 1.1B** — RTFx >2000 on GPU; not CPU-real-time at that size. `[V-web]`
- **Compact on-device streaming ASR** (arXiv 2604.14493) — *explicitly* targets low-latency on-device
  (the regime we need). `[?]` deep-read — likely the most relevant single paper for Track 2.
- **Blockwise decoder-only streaming** — 0.47 s latency, 8% rel WER reduction vs enc-dec/transducer on
  LibriSpeech. `[V-web]` (novel, integration cost unknown.)
- **SCAMA / latency-controlled** — 7.39% CER @ 600 ms (Mandarin). `[V-web]`

## 4. Where this points (pre-synthesis)
- The **transducer family is correct** for Track 2 (streaming-native, low-latency, strong with limited
  context, and the SAP evidence favors autoregressive decoders).
- **TDT's frame-skipping** is a genuinely interesting CPU-latency lever — but needs a *streaming + small +
  CPU-exportable* TDT, which may not exist off-the-shelf. `[?]`
- **Zipformer-T (baseline)** is the proven-feasible anchor; **cache-aware FastConformer** is the higher-
  ceiling-but-CPU-risky challenger (Axis 3 must settle its CPU RTF).
- Watch the **attention-free/SSM streaming** line — if a CPU-friendly linear-time streaming encoder exists
  with good dysarthric transfer, that could be the real edge. `[?]`

## Open questions → feed Axis 3/5
- Does a **streaming TDT** exist that exports to ONNX and runs real-time on CPU? `[?]`
- Cache-aware FastConformer measured **CPU** RTF int8 vs Zipformer? `[?]` (Axis 3)
- Streaming-vs-offline CER penalty for each family? `[?]`
- Are attention-free streaming encoders mature enough to fine-tune + export? `[?]`
