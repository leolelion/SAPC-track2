# RESEARCH_NOTES.md — Dysarthric + Streaming + CPU ASR

> **Epistemic tags**: `[K]` = from model knowledge (cutoff Jan 2026), defensible but
> unverified against a live source this session. `[V]` = verified from this repo's code.
> `[?]` = open question to research via web search / experiment before relying on it.
> Treat every `[K]` as a *theory* until a citation or an experiment confirms it.

This is a living log. Re-research before each new architecture/trick and update here.

---

## 0. Problem framing (what actually scores) `[V]`
- Primary objective: **minimize CER** on hidden test1/test2 (dysarthric speech), min-over-two-refs, clipped at 1.0.
- Pareto prize: winners sit on the **accuracy × latency** frontier where latency = `mean(TTFT, TTLT)` P50.
- Hard constraints: **CPU-only**, 15000 s/submission, Docker `sapc2-runtime` (torch 2.5.0).
- Therefore the real target is a **CER–latency–CPU-cost** 3-way frontier. A model that is
  0.5% better CER but 2× slower or that blows the time budget is worthless here.

## 1. Dysarthric / pathological speech ASR `[K]`
Key challenges vs typical ASR:
- High **inter- and intra-speaker variability**; atypical articulation, prosody, speaking rate.
- **Data scarcity** per speaker/disease; long-tail etiologies.
- Disfluencies, false starts, prolonged phonemes → why the challenge uses **two refs
  (with/without disfluency)** and CER over WER (CER tolerates partial-word correctness).

Datasets / prior art `[K]`:
- **TORGO**, **UASpeech** — classic dysarthric corpora (cerebral palsy, ALS). Small, speaker-few.
- **Speech Accessibility Project (SAP)** — the corpus behind this challenge; much larger,
  multi-etiology (Parkinson's, ALS, cerebral palsy, stroke, Down syndrome, etc.). `[?]` confirm
  exact etiology set + per-speaker counts from the manifest once data is in hand.

Strategies that historically help dysarthric ASR `[K]`:
1. **Fine-tune a strong pretrained model** on in-domain dysarthric data — by far the biggest
   single lever. The baseline here is LibriSpeech-only → large headroom. `[K]`
2. **Speaker adaptation / personalization** — per-speaker fine-tuning or adapters yields big
   gains on UASpeech-style data, but test speakers are unseen here, so favor *speaker-robust*
   training over per-speaker overfit. `[K]`
3. **Data augmentation** robust to atypical articulation: speed/tempo perturbation,
   SpecAugment, simulated dysarthria, noise/reverb. Tempo perturbation especially relevant
   given altered speaking rates. `[K]`
4. **Disease/etiology conditioning or multi-task** (predict etiology as aux task, or condition
   encoder/decoder on etiology embedding). Plausible gain; risk of leaking unavailable test-time
   etiology labels — only usable if etiology is provided at inference (it is NOT in the
   chunk-streaming interface). So treat as **train-time-only auxiliary signal**. `[V]` interface
   gives only audio chunks → no test-time etiology.
5. **Self-supervised encoders** (wav2vec2 / HuBERT / WavLM) fine-tuned on dysarthric speech are
   strong, but heavy for CPU streaming. `[K]`

## 2. Streaming ASR architectures `[K]`
- **Streaming Zipformer-Transducer** (icefall) — the provided baseline. Chunked causal
  attention + transducer; designed for low-latency streaming; strong WER/compute trade-off. `[K][V]`
- **FastConformer-Transducer / cache-aware streaming Conformer** (NeMo) — competitive, cache-aware
  multi-lookahead lets one model serve multiple latency settings. `[K]`
- **RNN-T / Transducer** generally preferred for streaming over CTC (better with context) and over
  full attention (no full-utterance dependency). `[K]`
- Whisper-like full-attention encoder-decoder = **not natively streaming**; adapting it to true
  low-TTFT streaming is awkward → better fit for Track 1 (offline). `[K]`

Latency levers within a streaming transducer `[K][V]`:
- **chunk_size / right-context (lookahead)**: smaller → lower TTFT, worse CER. Baseline
  `chunk_size=16` (~320 ms). This is the main TTFT knob. `[V config.yaml]`
- **left_context_frames**: more past context → better CER, modest cost. Baseline 128. `[V]`
- **Decoding**: `greedy_search` (fast, lower CER ceiling) vs `modified_beam_search`
  (`num_active_paths=4`, slower, better CER). Beam width is a CPU-cost knob. `[V]`
- **Partial emission policy**: emit a non-empty partial ASAP (even 1 token) to cut TTFT, since
  TTFT keys on *first non-empty* partial. Stability of partials not directly scored, but jitter
  could hurt perceived quality. `[V compute_latency.py]`
- **input_finished tail**: baseline pads 0.3 s silence to flush; this adds to TTLT. Tune the tail
  length — shorter tail → lower TTLT if CER holds. `[V model.py]`

## 3. CPU-efficient inference `[K]`
- **Quantization**: dynamic int8 (PyTorch `quantize_dynamic` on Linear/LSTM) is the cheapest win
  for CPU transducers; static/QAT int8 better but more work. Expect speedups with small CER cost. `[K]`
- **ONNX Runtime / sherpa-onnx**: icefall zipformers export cleanly to ONNX; sherpa-onnx is a
  mature CPU streaming runtime with int8 support → strong candidate for the final submission
  runtime (predictable CPU cost, low overhead vs PyTorch eager). `[K]`
- **Thread control**: pin `torch.set_num_threads` / OMP threads; CPU eval may give limited cores.
  Over-subscription hurts. Measure RTF (real-time factor) explicitly. `[K]`
- **Feature extraction cost**: baseline already supports `incremental` Fbank mode (O(n) not O(n²))
  — keep it on for streaming. `[V model.py]`
- **Model size vs CER vs latency**: a distilled/smaller encoder may be needed to hit the time
  budget on test1 (10521 utts) within 15000 s. Budget ≈ 1.4 s/utt avg → RTF must stay well under
  real-time on CPU. `[V]` compute the implied RTF once durations are known. `[?]`

## 4. Concrete ideas to test in THIS repo (ranked by expected ROI)
1. **Fine-tune the streaming Zipformer on SAP Train** (biggest lever). Keep architecture/streaming
   config identical so latency profile is unchanged; only weights change. `[K]`
2. **Augmentation during FT**: SpecAugment + speed/tempo perturbation (±10–20%). `[K]`
3. **Decoding sweep** on Dev: greedy vs beam (paths 2/4/8), chunk_size, left_context, tail length →
   map the CER–latency–CPU curve cheaply (no retraining). `[V]` doable purely via config.yaml.
4. **int8 dynamic quantization** of the FT model → measure CER delta + speedup. `[K]`
5. **ONNX / sherpa-onnx export** for the final runtime if PyTorch CPU is too slow. `[K]`
6. **(Stretch)** etiology-aux multitask during FT; LM shallow fusion for rare words. `[K]`

## 5. Open questions to resolve (research / data EDA) `[?]`
- Exact SAP etiology distribution, #speakers, #utts, duration stats per split.
- Implied RTF budget: total Dev/test duration ÷ 15000 s → how much CPU headroom for beam/quant.
- Does the runtime give us N cores? (affects threading + whether multiprocess Pass-1 matters for us).
- Best quantization recipe for streaming zipformer on CPU (dynamic int8 vs sherpa-onnx int8).
- Is there a public *dysarthric-fine-tuned* streaming checkpoint we can legitimately start from?
- Tokenizer: baseline BPE-500 trained on LibriSpeech text — is it adequate for SAP vocab, or
  retrain BPE on SAP transcripts? (CER is char-level so BPE mismatch hurts less, but still check.)

## 6. Verification ledger (fill as we confirm)
| Claim | Status | Evidence |
|---|---|---|
| Baseline is LibriSpeech-only (no dysarthric FT) | confirmed `[V]` | setup.sh HF repo id |
| CER = min over two refs, clipped 1.0 | confirmed `[V]` | compute_metrics.py |
| Pareto latency = mean(TTFT,TTLT) | confirmed `[V]` | Track2.md |
| Fine-tuning gives largest CER gain | confirmed `[V-lit]` | SAPC1: FT cut WER 17.82→8.11; all top teams FT a foundation model |
| Transducer > SSL/Whisper on SAP | confirmed `[V-lit]` | Takahashi'25 Table 1 (Parakeet-TDT 12.79 vs Whisper-v3 21.66 vs HuBERT 25.46) |
| Full FT > freezing decoder/joint | confirmed `[V-lit]` | Takahashi'25 Table 2 |
| int8 dynamic quant is net-positive on CPU | theory `[K]` | needs benchmark |

---

## 7. Session 2 — primary-source findings (SAPC1 winners + SAP data)
Sources: *Interspeech 2025 SAP Challenge* (arXiv 2507.22047) and the **1st-place** system
*Takahashi et al., "Fine-tuning Parakeet-TDT for Dysarthric Speech Recognition"* (Interspeech 2025,
doi 10.21437/Interspeech.2025-1484).

### 7.1 What the winners actually did (SAPC1, WER, GPU/offline — i.e. Track-1-like)
Every top-5 team **fine-tuned a public foundation ASR model** (NeMo Parakeet or OpenAI Whisper). Test2 WER:
- **Team a (1st, 8.11)** — Parakeet-TDT-1.1B (FastConformer enc + Token-Duration-Transducer dec):
  full FT (no freezing), **sentence-level segmentation of long audio via forced alignment**,
  SpecAugment + speed-perturb (0.9–1.1), **4-checkpoint weight averaging**, decoding via **mAES**
  (adaptive expansion search) to afford beam 30–40 within the time limit, decode **temperature ≈1.2–1.3**.
- Team b (10.03) — Whisper-large-v3, 15 s VAD segmentation, semi-supervised self-training.
- Team c (10.51) — Whisper-large-v2 + **AdaLoRA**, WhisperX preproc, rule-based postproc, curriculum.
- Team d (10.90) — Whisper-large-v3 + **LoRA** + **LLM post-ASR generative error correction**.
- Team e (11.62) — Parakeet-RNNT-0.6B; compared (Fast)Conformer / CTC / RNNT / ContextNet.
- Team h — Whisper-large-v3 + AdaLoRA + **personalization** (speaker-vector mapping).

### 7.2 Zero-shot model ranking on SAP Dev (Takahashi Table 1, WER) `[V-lit]`
wav2vec2-large 32.55 · HuBERT-large 25.46 · Whisper-large-v3 21.66 · Canary-1B 15.35 ·
Parakeet-RNNT-1.1B 12.88 · **Parakeet-TDT-1.1B 12.79**. → **Transducers (NeMo) dominate**; SSL is weakest.

### 7.3 SAP data facts `[V-lit]`
- 5 etiologies: PD, DS, ALS, CP, stroke. **PD-dominated (~73% train / ~76% test by duration)** →
  optimize for hypokinetic dysarthria but keep cross-etiology generalization. (SAPC2 = 1400 h successor.)
- Speaker-disjoint 70:10:20; eval on **"unshared"** subset (test text never appears in train) →
  do NOT tune to Train/Dev text; linguistic overfit is penalized. Mirrors our speaker-disjoint plan.
- Two refs (with/without disfluency), min per utterance — matches `compute_metrics.py` `[V]`.
- Normalization: brackets removed, {g:w}→w, unknowns→UNK, NeMo text-norm, UPPERCASE, punctuation
  stripped except in-word apostrophes. Our hypotheses must match this normalization regime.
- Known hard cases: **repetitions/stutters** (7 speakers with long stutter runs). Winner's internal
  LM deletes repeats → hurts the with-disfluency reference. Watch this in error analysis.

### 7.4 Translating the recipe to Track 2 (CPU-only, streaming, CER × latency)
The winning *recipe* transfers; the winning *model size* does NOT. 1.1B Parakeet on CPU cannot stream
in real time. So for Track 2 we want a **streaming transducer at a CPU-feasible size**, fine-tuned with
the winner's recipe. **Candidate feasibility — verified Session 2 (web):**

| Cand | Model | Params | CPU real-time | Fine-tune | Streaming | Status |
|---|---|---|---|---|---|---|
| C1 | icefall streaming Zipformer 2023-05-17 | **66.1 M** `[V-web]` | ✅ sherpa-onnx CPU RTF **0.11–0.19** `[V-web]`; **Q already ran it** `[V]` | icefall | native chunked | **SAFE** |
| C2 | NeMo cache-aware FastConformer-large streaming | **~110 M** `[V-web]` | ⚠️ NVIDIA docs: CPU "struggles at scale"; RTF<1 cited **GPU-only**; needs ONNX 3-part + int8 | NeMo | 80/160/320 ms chunks | **ceiling↑, CPU risk** |
| C3 | Nemotron/Parakeet streaming 0.6B | **600 M** `[V-web]` | ❌ RTF<1 GPU-only; too heavy for CPU RT | NeMo | native | **likely OUT** |

C2 integration risks `[V-web]`: sherpa-onnx cache-aware FastConformer export is an **open/WIP issue**
(GH #790, #2177); FastConformer attention/conv "benefit from GPU parallelism" = the costly-on-CPU ops.

**Decision rule**: C1 first (zero feasibility risk — only weights change vs the model Q ran). For C2,
**benchmark a pretrained checkpoint's RTF on the actual eval CPU BEFORE any fine-tuning** — admit C2 to
the FT queue only if it clears RTF < ~0.7 on CPU. C3 parked unless aggressively quantized.

Likely **out** for Track 2: Whisper-large (not streaming, heavy), SSL wav2vec2/HuBERT (weakest on SAP),
LLM post-ASR correction (adds latency/CPU → kills TTLT). Keep checkpoint-averaging, SpecAugment+speed
perturb, forced-align segmentation, full FT, temperature/beam tuning under the **15000 s** budget.

### 7.5 New open questions `[?]`
- Does NeMo's cache-aware streaming FastConformer fine-tune cleanly on SAP, and what is its CPU RTF int8?
- Streaming-decoding penalty: how much CER do we lose going from offline beam to chunked streaming?
- Can checkpoint-averaging be applied to the zipformer (icefall) the same way (avg `epoch-*.pt`)?
- Tokenizer: keep LibriSpeech BPE-500 or retrain BPE on SAP text? (CER is char-level, lower stakes.)
