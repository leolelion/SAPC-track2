# Track 2 (Streaming ASR) — Claude Code Execution Plan

## Context

We are competing in the SAPC2 challenge, Track 2 (Streaming ASR) for dysarthric speech recognition. The critical constraint is that **Track 2 is evaluated exclusively on CPUs** with a 15,000-second time limit for ~10,500 test utterances. Our initial test of Qwen3-ASR (1.7B) on CPU showed RTF of 20–172× (completely unviable). We need to find models that run faster than real-time on CPU while maintaining acceptable accuracy on dysarthric speech.

**Competition interface:** A streaming `Model` class with 5 methods: `__init__()`, `set_partial_callback(fn)`, `reset()`, `accept_chunk(chunk)` (100ms / 1600 samples at 16kHz), and `input_finished()`. Accuracy is measured via two passes (batch for CER/WER, streaming for latency). Latency metrics are TTFT and TTLT at P50. Winners are all teams on the Pareto frontier of accuracy vs. latency.

**Starting kit repo:** https://github.com/xiuwenz2/SAPC-template

**Docker image:** `xiuwenz2/sapc2-runtime:latest` (PyTorch 2.5.0+cu124, torchaudio, torchvision preinstalled)

**Our zero-shot baseline (from Bernard's Track 1 GPU eval on Dev set, 47,929 utterances):**

| Model | CER | WER | Notes |
|-------|-----|-----|-------|
| Qwen3-ASR 1.7B | 14.23% | 23.34% | Best accuracy, but 1.7B too slow for CPU |
| Whisper large-v2 | 20.36% | 26.08% | Previous challenge baseline |
| Whisper large-v3 | 23.01% | 32.23% | Worse than v2 on dysarthric speech |
| Parakeet TDT 0.6B v2 | 44.64% | 51.64% | 17% empty outputs, not viable |

**Official Track 2 baseline (on Test1, CPU):**

| Model | CER | WER | TTFT P50 (ms) | TTLT P50 (ms) |
|-------|-----|-----|---------------|---------------|
| Zipformer (icefall-asr-librispeech-streaming-zipformer-2023-05-17) | 34.59 | 52.77 | 1025.04 | 423.46 |

This is a Librispeech-trained Sherpa/Icefall Zipformer with zero dysarthric adaptation. It confirms the architecture runs on CPU within the time budget. Its accuracy is poor (34.59% CER) — fine-tuning on SAP data should dramatically improve this. The latency numbers (TTFT ~1s, TTLT ~423ms) are beatable.

**SAP data:** Training set has 336k utterances / 742 hours. Dev set has 47,929 utterances / 106.6 hours. Data is on RunPod S3 under `/workspace/SAPC2/processed/`.

---

## Phase 1: CPU Benchmarking (Priority Order)

The goal is to find which models can achieve RTF < 1.0 on CPU while producing non-empty, reasonable transcriptions. Run each model through the same streaming wrapper interface used in competition submission.

**For each model, measure:**
1. Model load time (seconds)
2. `accept_chunk()` latency (ms/chunk average)
3. `input_finished()` latency (ms) — this is where actual inference happens for most models
4. Real-Time Factor (RTF) — wall_time / audio_duration. Must be < 1.0 for viability
5. Peak RAM usage (GB)
6. Transcription output (verify non-empty on speech audio, not just white noise)

**Use real dysarthric speech from the Dev set, not synthetic audio.** Pick ~10 diverse utterances spanning short (2s), medium (5–10s), and long (15s+) durations, and varying severity levels if speaker metadata is available.

### Model 1: Sherpa-ONNX with Zipformer-transducer ⭐ TOP PRIORITY

- **Why first:** The official Track 2 baseline IS a Librispeech-trained Zipformer (34.59% CER, TTFT 1025ms, TTLT 423ms). This proves the architecture runs on CPU within the time budget. The baseline's only weakness is accuracy — it has zero dysarthric adaptation. Fine-tuning on SAP data (742 hours) should dramatically close the gap. The transducer architecture naturally emits tokens incrementally, which is an ideal match for the `accept_chunk` + `set_partial_callback` interface.
- **Install:** `pip install sherpa-onnx`
- **Models:** Start with the exact baseline model: `Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17` (HuggingFace). Also try other English streaming Zipformer variants from https://github.com/k2-fsa/sherpa-onnx for comparison.
- **Streaming support:** Native and specifically designed for chunk-by-chunk processing
- **Expected RTF on CPU:** < 0.3× (the baseline already passed the 15,000s time budget)
- **Fine-tuning path:** Icefall/k2 framework on GPU (RunPod H200) → export to ONNX → deploy on CPU. This is the most direct route to beating the baseline.
- **Action:**
  1. Install sherpa-onnx, download the baseline Zipformer model
  2. Wrap in SAPC2 `Model` class
  3. Reproduce the baseline numbers on Dev set (verify CER ~34% and latency)
  4. Run full Dev set scoring with official pipeline
  5. Begin fine-tuning plan on SAP data (see Phase 4)

### Model 2: Kroko Zipformer2 (Sherpa-ONNX) ⭐ HIGH PRIORITY

- **Why:** Bernard's suggestion. A recent edge-optimised streaming Zipformer variant available in sherpa-onnx, specifically designed for low-resource CPU deployment. May give better accuracy than the standard Zipformer at similar speed. Worth testing alongside Model 1 as a direct comparison within the same architecture family.
- **Install:** `pip install sherpa-onnx` (same as Model 1)
- **Models:** Search sherpa-onnx model zoo for Kroko Zipformer2 variants
- **Streaming support:** Native, same as standard Zipformer — transducer architecture with incremental token emission
- **Expected RTF on CPU:** Similar to or better than standard Zipformer (< 0.3×)
- **Fine-tuning path:** Same Icefall/k2 pipeline as Model 1 → ONNX export → sherpa-onnx deployment
- **Action:**
  1. Locate and download Kroko Zipformer2 model from sherpa-onnx
  2. Wrap in SAPC2 `Model` class (same wrapper as Model 1 with config changes)
  3. Benchmark RTF and zero-shot accuracy on 10 Dev utterances
  4. Compare directly against standard Zipformer (Model 1) on full Dev set
  5. If accuracy is better, prioritise this variant for fine-tuning

### Model 3: Zipformer 20M (Sherpa-ONNX) — Pareto latency anchor

- **Why:** Bernard's suggestion. Sherpa-onnx has a 20M param streaming Zipformer targeting embedded/Cortex-A7 devices. Much smaller than the 70M baseline. If accuracy is sufficient after fine-tuning on SAP data, the latency advantage could dominate the low-latency end of the Pareto frontier.
- **Install:** `pip install sherpa-onnx` (same as Models 1–2)
- **Models:** 20M param streaming Zipformer from sherpa-onnx model zoo
- **Streaming support:** Native transducer, same interface
- **Expected RTF on CPU:** Significantly < 0.3× given the small size
- **Key concern:** Zero-shot accuracy will be worse than the 70M variant. The bet is that fine-tuning on 742h of SAP data closes the gap enough for the latency advantage to win on the Pareto frontier.
- **Action:**
  1. Download 20M Zipformer model
  2. Benchmark RTF and accuracy on 10 Dev utterances
  3. Compare latency/accuracy tradeoff against Models 1–2
  4. If latency is substantially better, include in fine-tuning plan as a second Zipformer variant

### Model 4: Moonshine (Useful Sensors)

- **Why:** Purpose-built for CPU/edge streaming inference. Smallest models in the lineup. Could anchor the low-latency end of the Pareto frontier.
- **Variants:** `moonshine/tiny` (~30M params), `moonshine/base` (~100M params)
- **Install:** `pip install useful-moonshine-onnx` (ONNX variant preferred for CPU) or `pip install useful-moonshine`
- **Streaming support:** Native — has a `MoonshineOnnxModel` with transcribe_stream / transcribe methods
- **Expected RTF on CPU:** < 0.5× (designed for this)
- **Key concern:** Zero-shot accuracy on dysarthric speech is unknown. May need fine-tuning.
- **⚠️ Fine-tuning risk (Bernard's note):** Fine-tuning is undocumented for Moonshine v2; only community toolkits exist for v1. Be aware of this risk before investing significant time — if fine-tuning is needed to reach competitive accuracy, the unclear path here is a liability compared to the well-documented Zipformer/Icefall pipeline.
- **Action:**
  1. Install moonshine (ONNX variant)
  2. Wrap in the SAPC2 `Model` class interface
  3. Run on 10 Dev utterances, record RTF and transcription quality
  4. If RTF < 1.0, run on full Dev set with official scoring pipeline (dual-reference CER/WER)
  5. Assess whether zero-shot accuracy is competitive enough without fine-tuning; if not, deprioritise unless a clear fine-tuning path emerges

### Model 5: GigaAM 100M (Sber)

- **Why:** Bernard's suggestion. SSL-pretrained encoder with chunkwise attention enabling streaming. Reportedly outperforms its 240M teacher model through careful training. MIT license.
- **Params:** ~100M
- **Key concern:** Originally Russian-focused — must verify English speech recognition support before investing benchmarking time. If English is not natively supported or requires significant adaptation, skip.
- **Streaming support:** Chunkwise attention enables streaming inference
- **Action:**
  1. Check English language support — if absent, skip immediately
  2. If English-capable, install and wrap in SAPC2 `Model` class
  3. Benchmark RTF and accuracy on 10 Dev utterances
  4. Assess whether the accuracy/latency tradeoff is competitive with Zipformer variants

### Model 6: Whisper small/base (faster-whisper INT8) — DEPRIORITISED

- **⚠️ Bernard's note:** Encoder-decoder architecture is fundamentally not streaming. You'd buffer chunks and re-encode the whole accumulation on every `accept_chunk()` call, giving terrible TTFT. The Pareto scoring (which averages TTFT and TTLT) would penalise this heavily. Only pursue if transducer-based models (Zipformer, Moonshine) fail to reach acceptable accuracy.
- **Why still listed:** Whisper large-v2 scored 20.36% CER zero-shot — best accuracy lineage in the non-streaming family. If we can't close the accuracy gap with fine-tuned Zipformers, a Whisper fallback with buffered inference might still beat the baseline on accuracy alone (accepting poor latency).
- **Install:** `pip install faster-whisper` (CTranslate2 backend built-in)
- **Variants to test (if reached):** `small` (244M), `small.en`, `base` (74M), `base.en`
- **Expected RTF on CPU:** 0.5–2× for small with INT8
- **Action:** Only benchmark if Models 1–5 are insufficient. Focus on accuracy measurement only — latency will inherently be poor with this architecture.

### Model 7: Distil-Whisper (distil-large-v2) — DEPRIORITISED

- **Same deprioritisation reasoning as Model 6.** ~756M params, encoder-decoder, not truly streaming.
- **Action:** Only pursue as last resort if all other models are too inaccurate.

### Model 8: Voxtral Realtime (4B) — quick feasibility check only

- **Why:** Bernard initially suggested evaluating it. Native 80ms streaming latency on GPU, but 4B params on CPU is almost certainly too slow.
- **Action:** Do a single-utterance smoke test on CPU. If RTF > 10×, discard immediately and document for Bernard.

### Model 9: wav2vec2/HuBERT CTC — fallback only

- **Bernard's note:** Naturally streamable via CTC but typically 10–30% relative WER worse than transducers. Only consider as a fallback if transducer models (Zipformer variants) cannot be fine-tuned successfully.
- **Action:** Do not benchmark unless Zipformer fine-tuning hits a blocker.

### Summary of evaluation priority

| Priority | Model | Params | Rationale |
|----------|-------|--------|-----------|
| ⭐ 1 | Zipformer — standard (Sherpa-ONNX) | ~70M | Official baseline architecture, proven CPU-viable, fine-tuning is clearest path to winning |
| ⭐ 2 | Kroko Zipformer2 (Sherpa-ONNX) | TBD | Edge-optimised variant, may give better accuracy at similar speed |
| 3 | Zipformer 20M (Sherpa-ONNX) | ~20M | Tiny model for Pareto latency anchor, needs fine-tuning to be competitive on accuracy |
| 4 | Moonshine tiny/base | 30–100M | Fast on CPU, but fine-tuning path is risky (v2 undocumented) |
| 5 | GigaAM 100M | ~100M | Strong SSL encoder, but verify English support first |
| 6 | Whisper small/base (faster-whisper INT8) | 74–244M | ⚠️ Deprioritised — encoder-decoder gives terrible TTFT, Pareto penalty |
| 7 | Distil-Whisper | 756M | ⚠️ Deprioritised — same streaming limitation as Whisper |
| 8 | Voxtral Realtime | 4B | Smoke test only, almost certainly too slow on CPU |
| 9 | wav2vec2/HuBERT CTC | varies | Fallback only — CTC is 10–30% worse than transducers |

---

## Phase 2: Accuracy Evaluation on Full Dev Set

For every model from Phase 1 with RTF < 1.5× on CPU, run the full Dev set evaluation:

1. Process all 47,929 Dev utterances through the model
2. Score using the official dual-reference pipeline:
   - Use the scoring scripts from the starting kit: `utils/metrics/cer.py` and `utils/metrics/wer.py`
   - CER and WER computed against both with-disfluency and without-disfluency references, taking the minimum per utterance
   - CER/WER clipped to 100% at utterance level
3. Record: model name, variant, quantization, CER, WER, median RTF, peak RAM, model size on disk

**Build a results table** comparing all viable models against Bernard's baselines.

---

## Phase 3: Streaming Wrapper Implementation

For the top 1–2 models from Phase 2 (best accuracy–latency tradeoff):

1. **Implement the full SAPC2 `Model` class** per the competition interface:
   - `__init__()`: Load model, tokenizer, any preprocessing
   - `set_partial_callback(fn)`: Store callback reference
   - `reset()`: Clear audio buffer and any decoder state
   - `accept_chunk(chunk)`: Buffer the 1600-sample chunk. Optionally run incremental inference and fire partial callback for better TTFT
   - `input_finished()`: Run final inference on full buffered audio, fire callback with final transcript, return final transcript string

2. **Optimize partial hypothesis strategy:**
   - For TTFT: emit a partial result as early as possible (e.g., after 0.5–1s of audio)
   - For TTLT: emit final result as fast as possible after audio ends
   - Test different partial emission intervals (every 0.5s, 1s, 2s of audio) and measure impact on TTFT/TTLT

3. **Test with the local decoding script** from the starting kit:
   - `https://github.com/xiuwenz2/SAPC-template/tree/main/track2_starting_kit#local-dev-test-with-dev-set`
   - Use `Dev_streaming` subset for latency measurement
   - Run `utils/compute_latency.py` to get TTFT P50, TTLT P50

4. **Package for Codabench submission:**
   - Ensure all files fit the expected structure
   - Write `setup.sh` and/or `requirements.txt` for dependency installation
   - Test inside the Docker image: `xiuwenz2/sapc2-runtime:latest`
   - Verify total runtime < 15,000 seconds for test1 (10,521 samples)

---

## Phase 4: Fine-tuning ⭐ CRITICAL PATH

Fine-tuning is not optional — it's the primary strategy. The Zipformer baseline proves the architecture works on CPU but scores 34.59% CER with Librispeech training only. Fine-tuning on 742 hours of SAP dysarthric data should close the gap significantly.

**Expected improvement (reference point):** The SAPC1 winner (Takahashi et al., Interspeech 2025) used Parakeet TDT 1.1B and improved from 36.3% to 23.7% WER on dysarthric speech via fine-tuning. With the Zipformer starting at 34.59% CER baseline and 742h of training data, we should see a similar magnitude improvement.

### Priority 1: Zipformer fine-tuning via Icefall/k2

1. **Set up Icefall** on RunPod H200 (GPU for training)
   - **⚠️ Check k2 + lhotse compatibility with the pod's CUDA version before launching** (Bernard's note)
   - Check GPU availability in US-CA-2 first — there were stock issues previously. If unavailable, check other regions but note the SAP data volume is region-locked to US-CA-2.
2. **Prepare SAP data** in Icefall-compatible format (Lhotse manifests) from `/workspace/SAPC2/processed/Train/`
3. **Fine-tune the streaming Zipformer-transducer** starting from the Librispeech checkpoint (`Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17`)
   - **Two documented approaches in Icefall:**
     - Standard fine-tuning from pre-trained checkpoint (learning rate ~1/10 of original training LR)
     - Adapter-based fine-tuning (lower GPU memory requirement — useful if H200 memory is tight)
   - Fine-tune all promising Zipformer variants (standard ~70M, Kroko Zipformer2, 20M) to compare accuracy/latency tradeoffs after adaptation
4. **Export to ONNX** for CPU deployment via sherpa-onnx
5. **Evaluate** fine-tuned model on Dev set with official scoring
6. **Package** and submit to Codabench

### Other fine-tuning paths (if Zipformer underperforms)

- **Whisper variants:** HuggingFace `transformers` + `peft` (LoRA) on GPU → convert to CTranslate2 for CPU. Note: deprioritised due to poor streaming latency characteristics, but may be worth trying if accuracy is the bottleneck.
- **Moonshine v2:** ⚠️ Fine-tuning is undocumented for v2; only community toolkits exist for v1. High risk of wasted time. Only attempt if Moonshine zero-shot accuracy is promising enough to justify the exploration.
- **Knowledge distillation from Qwen3-ASR:** Consider distilling Qwen3-ASR (14.23% CER) into a smaller streaming model if direct fine-tuning of lightweight models doesn't reach competitive accuracy.
- Fine-tuning always happens on GPU (RunPod H200), then export/quantize for CPU inference.

---

## Key Files and Paths

| Item | Location |
|------|----------|
| SAP training data | `/workspace/SAPC2/processed/Train/` (on RunPod volume) |
| SAP dev data | `/workspace/SAPC2/processed/Dev/` |
| Starting kit | `https://github.com/xiuwenz2/SAPC-template` |
| Scoring scripts | `utils/metrics/cer.py`, `utils/metrics/wer.py` |
| Latency measurement | `utils/compute_latency.py` |
| Local decode script | `track2_starting_kit/` in starting kit repo |
| Docker image | `xiuwenz2/sapc2-runtime:latest` |
| Codabench Track 2 | `https://www.codabench.org/competitions/14177` |

---

## Success Criteria

- Beat the official baseline (34.59% CER) on accuracy while maintaining comparable or better latency
- At least one model with RTF < 1.0 on CPU and CER < 30% on Dev set
- Working SAPC2 `Model` class implementation passing local decode tests
- Packaged submission that runs within 15,000 seconds on Codabench
- Competitive position on the accuracy–latency Pareto frontier (either via accuracy improvement from fine-tuned Zipformer, or via latency improvement from a fast lightweight model like Moonshine)
