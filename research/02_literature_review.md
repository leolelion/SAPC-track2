# 02 - Literature And Runtime Survey

Executive summary:
1. Nemotron's model card and the independent 2026 on-device study both support the core premise: cache-aware FastConformer RNNT is a strong streaming architecture when the runtime is efficient.
2. The most transferable runtime idea is not generic "use ONNX"; it is three-graph decomposition: encoder, prediction network, and joiner optimized separately.
3. TDT frame-skipping is powerful in the literature, but this exact decoder export has no duration head, so it is not available without re-exporting or changing models.
4. ONNX Runtime quantization guidance and our graph inspection both say the current encoder is already dynamic-int8-like; the next quantization question is operator quality, not whether int8 exists.
5. sherpa-onnx is promising for native streaming transducers generally, but the local API expects separate encoder/decoder/joiner graphs and only exposes online NeMo CTC support, not this combined NeMo RNNT export.

## Sources Reviewed

- NVIDIA model card: `nvidia/nemotron-speech-streaming-en-0.6b`  
  https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- Daniel Bodart ONNX export card: `danielbodart/nemotron-speech-600m-onnx`  
  https://huggingface.co/danielbodart/nemotron-speech-600m-onnx
- Banfic et al., "Pushing the Limits of On-Device Streaming ASR: A Compact, High-Accuracy English Model for Low-Latency Inference", arXiv 2604.14493, 2026.  
  https://arxiv.org/abs/2604.14493
- Rekesh et al., "Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition", arXiv 2305.05084, 2023.  
  https://arxiv.org/abs/2305.05084
- Xu et al., "Efficient Sequence Transduction by Jointly Predicting Tokens and Durations", arXiv 2304.06795, 2023.  
  https://arxiv.org/abs/2304.06795
- Yu et al., "FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization", arXiv 2010.11148, 2020.  
  https://arxiv.org/abs/2010.11148
- Kim and Lee, "Accelerating RNN Transducer Inference via One-Step Constrained Beam Search", arXiv 2002.03577, 2020.  
  https://arxiv.org/abs/2002.03577
- ONNX Runtime quantization, threading, graph optimization, and I/O binding docs.  
  https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html  
  https://onnxruntime.ai/docs/performance/tune-performance/threading.html  
  https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html  
  https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html
- sherpa-onnx documentation and local Python API inspection.  
  https://k2-fsa.github.io/sherpa/onnx/index.html

## Cache-Aware FastConformer / Nemotron

NVIDIA describes Nemotron Speech Streaming as a 600M-parameter cache-aware FastConformer RNNT with 24 encoder layers and an RNNT decoder. The model card states that the cache-aware design processes non-overlapping chunks while maintaining self-attention and convolution caches, reducing redundant computation relative to buffered streaming. It also exposes chunk/right-context operating points of 80 ms, 160 ms, 560 ms, and 1120 ms through `att_context_size` variants.

Applicability:
- Directly applicable architecturally. Our extracted model uses the 560 ms `[70,6]` style path: 56 new 10 ms mel frames plus 9 pre-encode cache frames.
- The intended efficient behavior is "process new chunks plus caches"; our encoder path does that, but our feature path does not because it recomputes all prior log-mel frames.

Risk:
- NVIDIA's card discusses model capability and intended runtime behavior, not SAPC2 worker CPU performance.
- Punctuation/capitalization benefits do not matter much here because the official scorer normalizes text.

## Daniel Bodart ONNX Export

The export card matches our local graph and code:
- Pre-emphasis 0.97.
- 16 kHz input.
- FFT 512, window 400, hop 160, 128 mel bands.
- Encoder chunk size 56 mel frames with 9 pre-encode cache frames.
- Encoder cache tensors shaped like our `cache_last_channel` and `cache_last_time`.
- RNNT blank id 1024, vocab size 1025, 2-layer LSTM prediction network with hidden 640.
- Greedy loop is frame-by-frame, up to 10 symbols per frame.

The export card describes `int8-dynamic` as `MatMulInteger`-style and CPU-only, with decoder staying FP32. Our static graph inspection found `MatMulInteger` and `DynamicQuantizeLinear` in the encoder, so the bundled model appears aligned with the dynamic-int8 CPU path rather than QDQ static quantization.

Applicability:
- Very high. It is the exact source family of our ONNX files.

Risk:
- The exporter provides a reference layout, not an optimized Python loop for SAPC2's real-time callback/scoring contract.

## On-Device Nemotron Study

Banfic et al. (2026) is the most directly relevant outside source. They evaluated many ASR configurations and selected Nemotron-0.6B for CPU-only streaming. Their key runtime design is a three-graph decomposition: cache-aware FastConformer encoder, LSTM prediction network, and joiner as independent ONNX sessions. They report that this allows per-component quantization and that multi-head-attention fusion materially speeds up the encoder.

Key reported results:
- Recommended configuration: Nemotron-0.6B int4 k-quant, 0.56 s algorithmic delay.
- Reported average streaming WER: 8.20% across their benchmark suite, close to FP32 ONNX 8.03%.
- Reported size reduction: 2.47 GB to 0.67 GB for int4 k-quant.
- Reported CPU RTFx above 6x for optimized ONNX variants.
- Important negative result: full-integer ConvInteger/MatMulInteger degraded WER, which they attribute to accumulated rounding through the encoder.

Transferability:
- High conceptually: same Nemotron family, CPU-only streaming, ONNX Runtime.
- Medium practically: their hardware includes a 32-core server CPU; SAPC2 worker CPU is unknown and may be weaker.
- The three-graph decomposition is especially relevant because our current decoder ONNX combines prediction net and joiner, which prevents separate caching/quantization/batching strategies.

Implication for SAPC2:
- If we need a structural speed win, re-exporting/sourcing a split decoder and joiner is more promising than micro-optimizing the current combined decoder loop.
- Attention/encoder fusion is worth checking on Linux ORT; our local profile did not show a fused attention op dominating, and the profile still contains many small attention-related ops.

## RNN-T Decoding Efficiency

The standard RNNT greedy loop has an inner while loop over nonblank symbol emissions. The one-step constrained beam-search paper is relevant because it targets that loop shape by constraining/vectorizing expansions. NeMo docs also describe transducer decoding variants such as TSD, ALSD, and mAES; mAES adapts expansion count per timestep and often constrains expansions to a small number.

Applicability:
- For our current greedy path, the main operational knob is `MAX_SYMBOLS_PER_FRAME=10`, but reducing it risks accuracy and must be scorer-gated.
- Native/vectorized approaches would need a different decoder implementation or graph decomposition.
- Beam variants are probably not the first speed fix for Nemotron because the zero-shot collapse is already compute-bound; beam search would likely add compute.

Evidence from our local profile:
- On synthetic speech, decoder calls were numerous but not dominant: 168 decoder calls, about 80 ms median decoder time over 8.5 s audio.
- This may change on dysarthric speech, so real SAP profiling is mandatory before dismissing decoder loop work.

## TDT / Duration Skipping

TDT jointly predicts token and duration, allowing inference to skip input frames. The TDT paper reports up to 2.82x faster ASR inference than conventional transducers in their setting.

Applicability:
- Low for the current export. Our decoder outputs are logits, prednet length, and LSTM states; no duration distribution exists.
- Medium only if we change model/export family to a TDT Nemotron/Parakeet variant, which is a larger modeling decision and may affect SAP accuracy.

## Emission Latency Training

FastEmit adds a sequence-level latency regularizer for transducers and reports substantial emission-latency reductions while preserving or improving WER in their experiments.

Applicability:
- Low for immediate speed gate because it requires training/fine-tuning, while the current blocker is runtime keeping up.
- Potentially relevant after runtime is fixed if TTFT remains poor despite RTF < 1.

## ONNX Runtime Engineering

Quantization:
- ORT supports QOperator and QDQ representations.
- ORT notes dynamic quantization computes activation parameters online, while static quantization uses calibration data.
- ORT recommends dynamic quantization generally for RNNs/transformers and static for CNNs, but real profiles and accuracy checks decide.
- ORT supports int4/uint4 weight-only quantization for constant-weight MatMul and Gather via newer opsets/runtime support.

Threading:
- ORT exposes intra-op and inter-op thread controls.
- ORT docs explicitly recommend sweeping/tuning thread settings for NUMA and parallel sessions rather than assuming defaults.

Graph optimizations:
- ORT optimization levels include basic, extended, and layout optimizations.
- Offline optimized models can reduce startup cost, but must be produced with target-compatible EP/options/hardware assumptions.

I/O binding:
- Useful mainly to avoid copies and bind persistent buffers, especially on GPU or when outputs live on non-CPU devices.
- For our CPU-only small-tensor cache loop, I/O binding may reduce allocations, but the local profile points first to compute kernels rather than copies.

## sherpa-onnx Feasibility

sherpa-onnx supports online streaming transducer models generally, and its docs list many streaming Zipformer/Conformer transducer examples. Local Python API inspection of installed `sherpa_onnx==1.12.29` found:
- `OnlineTransducerModelConfig(encoder, decoder, joiner)` requires three separate files.
- `OnlineNeMoCtcModelConfig(model)` exists for online NeMo CTC.
- No local online NeMo RNNT/FastConformer cache-aware config was exposed.

Applicability:
- Medium for a re-exported three-graph transducer that matches sherpa's expected encoder/decoder/joiner interface.
- Low for directly dropping in the current two-graph `encoder_model.onnx` + combined `decoder_model.onnx`.

Open question:
- A newer sherpa-onnx release may expose more model types than local 1.12.29. This must be checked on the Linux packaging target before committing to a sherpa rewrite.

## Technique Comparison

| Technique | Mechanism | Expected gain | Accuracy risk | Integration risk | Current verdict |
|---|---|---:|---:|---:|---|
| Worker CPU/thread sweep | Set `SAPC2_THREADS`, ORT intra-op/inter-op for actual worker | Medium | None | Low | Do immediately after CPU diagnostic |
| Warmup in `__init__` | Pay first-kernel/session cost before first scored utterance | Low/medium for tail | None | Low | Cheap if budget impact acceptable |
| Incremental mel | Compute only new STFT/mel frames with rolling sample context | Low/medium locally, higher on long/slow cases | Low if bit-equivalent | Medium | Good first code change if profile confirms |
| Split decoder/joiner | Cache prediction net separately, batch/fuse joiner, per-component quant | Medium/high | Low if numerically same export | High | Best structural runtime bet |
| Native sherpa-onnx | C++ streaming loop/features/decoder | Medium/high | Low/medium depending export parity | High | Needs export compatibility check |
| Encoder fusion / optimized ORT model | Improve MHA/MatMul/Conv kernels | Medium/high if fusion missing | Low | Medium/high | Profile-guided Linux task |
| Int4 k-quant | Weight-only compression/speed for MatMul/Gather | Medium/high in paper | Medium | High | Promising but too large before scorer guardrail |
| TDT duration skipping | Predict durations and skip frames | High in TDT models | Model change | High | Not available in current export |
| FastEmit | Train for earlier nonblank emission | TTFT improvement | Training-dependent | High | Post-speed-gate only |

