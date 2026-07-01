# SAPC2 Track 2 — Finetuning Techniques & Considerations (Team Study Brief)

*A self-contained walkthrough of what we tried with the **Zipformer** and **Nemotron** models, the
techniques behind each step, why we made the calls we did, and the lessons that cost us the most to learn.
Written to be read top-to-bottom and then used as a reference. Source of truth: `research/00`–`research/36`.*

---

## 0. One-paragraph summary

We are building a **streaming, CPU-only, low-latency** ASR system for **dysarthric (disordered) speech**.
We worked two model families in parallel: a small **66M streaming Zipformer** (our reliable, proven anchor)
and a large **600M cache-aware FastConformer-RNNT ("Nemotron")** (higher accuracy ceiling, heavier to deploy).
The Zipformer gave us a trustworthy baseline and a *free* accuracy win from **beam-search decoding**. The
Nemotron, after **encoder-only finetuning + a deployment-correctness fix + int8 quantization**, reached a
**Dev-500 CER of 8.76%** on the official harness — well under the Zipformer's hidden-Test1 reference of 23.44%.
The single most important methodological lesson: **a hand-rolled "proxy" evaluation will lie to you; only the
organizers' exact pipeline counts for any ship decision.**

---

## 1. What makes Track 2 hard (the constraints that shaped every decision)

| Constraint | Consequence for technique |
|---|---|
| **Streaming** — audio arrives in 100 ms (1600-sample) chunks; you must emit partial text as you go | Model must be **streaming-native** (cache-aware / causal). You cannot just run a normal offline model. |
| **CPU-only**, 15000 s total budget per submission | Param count matters less than **ONNX + quantization engineering**. A 600M model is viable only with int8 and careful threading. |
| **Latency-scored** (TTFT, TTLT) *and* **accuracy-scored** (CER) | Two objectives → a **Pareto frontier**, not a single best model. Decoding/flush policy is a lever independent of weights. |
| **CER is the primary metric**, min over two references, per-utterance **clipped at 100%** | Optimize characters, not words. A totally-wrong utterance can't cost more than 100%, so **empty outputs are very expensive** (they're a guaranteed 100%). |
| **Dysarthric speech** — slow, variable rate, imprecise articulation, severity varies by etiology | Acoustic model is **uncertain** (flat per-step probabilities) → favors wider search; augmentation (speed/spec) and severity-aware data matter. |
| Test speakers are **unseen and speaker-disjoint**; transcripts are "unshared" | Validate on **speaker-disjoint** internal splits; **never tune to Dev/Test text**; guard against memorization/overfitting. |

**Two metrics, defined:**
- **TTFT** (time-to-first-token) = time of first non-empty partial − (audio start + speech-onset offset). Lowered by emitting *something* early.
- **TTLT** (time-to-last-token) = final visible text time − audio-end. Lowered by finalizing fast after audio stops.

---

## 2. The two models at a glance

| | **Zipformer** | **Nemotron** |
|---|---|---|
| Architecture | Zipformer encoder + **stateless/pruned RNN-T** (icefall) | 24-layer **cache-aware FastConformer** encoder + 2-layer LSTM pred-net + RNN-T joint |
| Size | ~66M | ~600M |
| Toolkit | k2 / icefall / sherpa-onnx | NVIDIA NeMo → ONNX Runtime |
| Vocab | BPE-500 | BPE-1024 (+ blank id 1024) |
| Role | **Anchor**: proven CPU real-time, low risk, already in our pipeline | **Challenger**: higher ceiling, heavier deploy, more engineering risk |
| Headline result | greedy 21.6% → **beam-4 19.0%** CER (2k ruler); hidden Test1 **23.44%** | finetuned + int8 → **8.76%** CER (official Dev-500) |

**Why two?** The Zipformer is a streaming-native model with *proven* CPU real-time performance (5–9× faster
than real-time) that Q already had running. It de-risks the whole project: even if the big model fails to
deploy, we have a working entry. The Nemotron is the upside bet — cache-aware so its streaming accuracy ≈ its
offline accuracy (the property that matters most for this track), with a much higher accuracy ceiling.

### 2.1 Head-to-head results (and which comparisons are actually same-set)

> **Read this first.** Most of our numbers were measured on *different* dev splits, so they are only loosely
> comparable. We have exactly **two genuinely same-manifest comparisons** — the severe **Dev_diag-425** set
> (accuracy) and the **Dev_streaming-123** set (latency). Those are the controlled ones; everything else is
> directional. All numbers below are through the official/faithful harness unless noted.

**A. Accuracy — same set (Dev_diag-425, severe-enriched, official scorer):** *the controlled accuracy comparison.*

| Model | Dev_diag-425 CER | Empty % |
|---|---|---|
| Nemotron, zero-shot | 47.5% | 25% |
| Zipformer A1 (SAP-finetuned), greedy | 31.3% | 6% |
| Zipformer A1, beam-4 | 29.1% | 5% |
| **v1 Nemotron (encoder-only FT, int8)** | **24.85%** | 7% |

→ On the same hard utterances, **v1 Nemotron beats the Zipformer by ~6.4 CER (vs greedy) / ~4.3 (vs beam-4)**.
(FP32 v1 checkpoint scored 23.6% on this set; the shipped int8 is 24.85% — int8 cost ~+0.36 here.)
*Caveat:* both runs used the same `Dev_diag.csv` construction (7 worst speakers + duration bins) but the two
manifests were **not byte-diffed** — the worst-speaker selection is data-driven, so treat as same-set pending a
manifest id-list diff.

**B. Accuracy — representative set (DIFFERENT manifests — directional only):**

| Model | CER | Set |
|---|---|---|
| Zipformer A1, greedy | 21.61% | Dev-2k ruler |
| Zipformer A1, beam-4 | 19.01% | Dev-2k ruler |
| **v1 Nemotron int8** | **8.76%** (FP32 8.64%) | Dev-500 |
| Zipformer A1 (submitted) | **23.44%** | hidden **Test1** |
| v1 Nemotron | *never submitted* | — |

→ The ~13-pt gap (8.76 vs 21.61) is suggestive but **not controlled** (2k vs 500 utts). The only
*confirmed-on-Test* datapoint is the Zipformer's 23.44%; v1's honest Test1 projection (applying the Zipformer's
~+1.8 Dev→Test gap) is **low-to-mid teens CER** — still a large win, but a projection, not a measurement.

**C. Latency — same set (Dev_streaming-123, pod CPU, real-time 100 ms pacing, 4 threads):** *the controlled latency comparison.*

| Model | TTFT p50 | TTLT p50 | mean(TTFT,TTLT) |
|---|---|---|---|
| Zipformer A1, greedy | 1234 ms | 100 ms | 643 ms |
| Zipformer A1, beam-4 | 1208 ms | 101 ms | 654 ms |
| **v1 Nemotron int8** | **1075 ms** | 199 ms | **637 ms** |

v1 full percentiles: TTFT p90 2.12 s / p95 2.85 s; TTLT p90 0.24 s / p95 0.25 s.
→ **A latency wash on the Pareto axis** (mean 637 vs 643–654 ms): v1 emits its first token *earlier* (better
TTFT) but finalizes ~2× slower (worse TTLT), and the two cancel. *Caveat:* these are **pod** numbers; the
Codabench worker inflates TTFT ~1.5–3× (the Zipformer's pod 1208 ms became **1438 ms** on real Test1). v1 has
**no real-worker latency** — its 1075 ms would likely land ~1.6–3.2 s on Codabench.

**Bottom line:** on both controlled comparisons, **v1 Nemotron wins accuracy decisively and ties on latency** —
the favorable Pareto position. The remaining uncertainty is the unmeasured v1 Test1 / Codabench-worker numbers.

---

## 3. The techniques, one by one

Each subsection: **What it is → Why it matters here → What we actually found.**

### 3.1 Streaming-native (cache-aware) architecture
- **What:** A model trained to consume audio in chunks while carrying forward a *cache* of past
  context (left-context attention + convolution state), instead of seeing the whole utterance at once.
- **Why:** The literature is decisive — forcing an *offline* SOTA model to chunk costs **+1.9 WER (Parakeet)
  up to +77% relative (Qwen)**, whereas a cache-aware model costs **~+0.2%**. So we refused to repurpose an
  offline model for streaming, even a very accurate one.
- **Found:** Both chosen models are streaming-native. The Nemotron uses an attention-context setting written as
  `[left, right]` = `[70, 1]` (≈160 ms lookahead) which we **pinned identically for train, eval, and export** so
  there's no train/deploy mismatch.

### 3.2 Finetuning strategy: full vs. encoder-only
- **What:** "Full" = update all weights. "Encoder-only" = freeze the decoder (pred-net) + joint network, train
  only the acoustic encoder.
- **Why:** Freezing the transducer head **preserves the pretrained language/decoding behavior** while letting the
  encoder adapt its acoustic representations to dysarthric speech. Full finetune risks perturbing/overfitting the
  head, especially with disordered-speech transcripts.
- **Found (decisive):** Two-arm run on all 309k utterances. **Encoder-only won**: Dev_diag **23.6%** CER vs.
  full-unfrozen 25.2%; internal-dev 8.01% vs 11.06% (the gap = full-unfrozen overfitting). The encoder-only
  advantage seen at small scale **held** at full scale. *Nuance:* full-unfrozen had slightly fewer empty outputs
  (6% vs 8%) — a hint that the **joint network governs the empty-output rate**, which mattered later (§3.11).

### 3.3 Data preparation & splits
- **Speaker-disjoint internal validation.** We held out ~8% of speakers (deterministic seed) → train
  308,909 utts / 683 h / 805 speakers; internal-dev 27,135 utts / 70 speakers. Official Dev was kept entirely
  separate. **Why:** test speakers are unseen, so a random utterance split would leak speaker identity and
  over-report accuracy.
- **Target text = cased + punctuation** (the scorer normalizes both sides anyway).
- **Long-audio filter:** utterances > 40 s were dropped (no segmentation). We later audited this (§4) and found
  it only drops 1.47% of data and is *not* concentrated in the severe speakers — so it's a low-value lever.
- **Augmentation:** dysarthria-tuned **SpecAugment** (less frequency masking, more time masking — speaking rate
  varies a lot). **Speed perturbation (0.9/1.0/1.1) was *not* used in v1** — flagged as a high-value missing lever.

### 3.4 Optimization recipe (the actual hyperparameters)
- AdamW, **lr 1e-4**, weight decay 1e-3, betas (0.9, 0.98); **cosine** schedule, **15% warmup**, min_lr 1e-6.
- bf16 mixed precision, batch size 16, **4 epochs** with early-stop on internal val.
- **Consideration we got wrong in v1:** 15% warmup over a *4-epoch* run means ~60% of epoch-1 is spent warming
  up → compounds under-training. For short runs, warmup should be ~5–10%.

### 3.5 Checkpoint averaging (SWA-style)
- **What:** Average the *weights* of several checkpoints before export, instead of picking one.
- **Why:** Averaging tends to land in a **flatter minimum** that generalizes better — a standard trick from the
  SAPC1-winning recipe and the SWA literature.
- **Found (important, and counter-intuitive):** We *suspected* our averaging was "contaminated" because it
  included early (epoch-0) checkpoints, and that a single best checkpoint would be better. We **tested it** on a
  reliable validation set: the **average (11.83% CER) beat the best single checkpoint (12.47%)** *and* produced
  fewer empties. Averaging was doing real work. **Lesson: don't "fix" something you haven't measured** — we
  almost removed a beneficial technique on a plausible-sounding hunch.

### 3.6 Decoding: greedy vs. beam search (+ language model)
- **What:** Greedy picks the top token each step. **Modified beam search** keeps several hypotheses alive.
- **Why here:** Dysarthric audio makes the acoustic model *uncertain* — per-step distributions are flat, so
  greedy's locally-best pick is fragile. A wider search recovers a lot.
- **Found (Zipformer):** greedy 21.61% → **beam-4 19.01%** → beam-8 18.33% CER (official sclite, 2k ruler). That
  **−2.6 to −3.3 pts** is far bigger than the usual 0.5–2% — and critically, **beam-4 latency ≈ greedy** because
  TTFT is dominated by chunk accumulation (~640 ms), not search width, and per-chunk decode stays within the
  100 ms budget. **Beam-4 was a strict win: lower CER, same latency, same model, zero retraining.** Beam helped
  *most* on the hardest etiologies (Cerebral Palsy, Stroke, Down Syndrome) — exactly where the model is most
  uncertain. (RNN-LM shallow fusion was the planned next stack on top.)
- **Transferable idea:** on uncertain acoustic conditions, **decoding-time changes can rival retraining** and are
  essentially free.

### 3.7 Deployment: ONNX export + int8 dynamic quantization
- **What:** Export the trained model to ONNX (separate encoder + decoder/joint graphs), then **dynamically
  quantize the encoder weights to int8** (activations quantized on-the-fly) while keeping the decoder in FP32.
- **Why:** The encoder is the compute bottleneck on CPU (profiling showed encoder ≫ decoder ≫ features). int8
  shrinks it and speeds matmuls (`MatMulInteger` / `DynamicQuantizeLinear`) with near-lossless accuracy. We kept
  the small LSTM decoder in FP32 because quantizing it buys little and risks accuracy.
- **Found:** int8 cost almost nothing in accuracy on the faithful harness — **FP32 10.49% → int8 8.76%** on
  Dev-500 across the pipeline iterations (int8 also produced a smaller, upload-friendly ~838 MB package vs the
  2.2 GB raw-FP32 ONNX package).

### 3.8 RNN-T SOS / blank initialization — the "deployment-correctness" bug class
- **What:** An RNN-T decoder must be primed with a start symbol. For this model that is the **blank id (1024)**,
  *not* token 0. Our first streaming wrapper primed it with `0`.
- **Why it's insidious:** GPU `transcribe()` (the training-side path) was *correct* and said encoder-only was
  great. But the **deployment `model.py`** had the wrong SOS, so the streaming model emitted garbage —
  literally the string `"Wh "` on 80/100 utterances — and made encoder-only look **broken** (11.7% vs the true
  1.3%). One-line fix (`_last_token = BLANK_ID`) restored it.
- **Lesson:** accuracy bugs live in the *wrapper* (token init, mel frontend, ONNX I/O, threading), not just the
  weights. **The deployment path is part of the model.** This is why §5's faithful-harness rule exists.

### 3.9 Latency engineering (TTFT / TTLT)
- **Where time goes (CPU profiling):** encoder kernels dominate (`Conv`, quantized matmuls); the mel frontend
  **recomputed features ~9×** (it re-ran the filterbank over the whole audio prefix each chunk — a real,
  fixable inefficiency); decoder dispatch is many small calls but not dominant on strong CPUs.
- **Thread sensitivity:** RTF improved 0.366 (1 thread) → 0.173 (2) → 0.130 (4) → 0.113 (8). Thread count is a
  real knob, but the **eval worker's core count is the gating unknown** — we design for safety at ~8 cores.
- **What sets TTFT:** it's **chunk-accumulation-bound** (~640 ms class) far more than compute-bound — which is
  why beam search didn't hurt it. Latency is largely *policy* (eager first partial, short flush tail, chunk size)
  → many Pareto points from one set of weights.
- **Measured:** TTFT p50 ≈ 1.0–1.2 s, TTLT p50 ≈ 0.2 s on the faithful streaming pass.

### 3.10 The scorer contract (things that silently fail a submission)
- **Min over two references**, per-utterance CER **clipped at 1.0**. Because of clipping, **empty outputs are
  pure loss** (100% each) — reducing empties is high-value.
- **`unk` handling:** the official scorer reconciles `unk` tokens between hypothesis and reference. If the model
  *emits a literal `unk`*, the two-reference alignment can diverge and trip an assertion
  (`preds from sgml-ref1 and sgml-ref2 are not identical`) — i.e. the submission **fails to score at all**. Fix:
  **strip standalone `unk`** from what the model returns (in `model.py`, never the scorer).
- **Golden rule:** never edit the scorer (`evaluate.sh`, `compute_metrics.py`, `local_decode.py`). Wrap, don't modify.

### 3.11 The empty-output ceiling (a structural insight)
- Encoder-only finetuning *cannot* fully fix empty outputs, because with the **joint network frozen**, its
  blank-emission propensity is fixed. We confirmed empties (~3–4%) do **not** shrink with more encoder epochs.
- **Implication for future work:** to move the empty floor and the hardest speakers, the real lever is
  **structural** (carefully unfreezing the joint/pred-net at low LR), not more of the same encoder training.

---

## 4. The experiment arc (how the story actually unfolded)

1. **Scoping & literature.** Concluded: finetune a *streaming-native transducer on all SAP data* is the main
   lever; rejected offline SOTA, SSL (wav2vec2/HuBERT underperform on SAP), and LLM-post-correction (kills latency).
2. **Zipformer anchor.** Full finetune; then the **beam-search decoding win** (−2.6 CER at no latency cost).
   Established the trustworthy baseline (Test1 ref 23.44%).
3. **Nemotron zero-shot.** Strong on clean speech, **failed on severe dysarthria** (43.4% Dev_diag, 18% empty)
   — motivating finetuning.
4. **Nemotron finetune.** Encoder-only vs full-unfrozen → **encoder-only wins** (43.4 → 23.6% Dev_diag, −19.8).
5. **Deployment crisis & fix.** Streaming wrapper made encoder-only look broken → traced to the **SOS/blank bug**
   → fixed → encoder-only confirmed best on the faithful CPU harness (~10.5% Dev-500 FP32).
6. **Quantize & package.** Encoder int8 → **8.76% Dev-500**, smaller offline package.
7. **Scorer assertion.** `unk`-token bug found and fixed in the wrapper; all gates pass.
8. **v2 accuracy workup (Phase 0).** Before spending on a retrain, we cheaply re-tested our assumptions on a
   reliable speaker-stratified validation set and found: **(a)** CER had **largely converged by epoch 3**
   (marginal gain collapsed from −1.46 to −0.13 pts) — so "just train longer" is weak; **(b)** checkpoint
   **averaging was beneficial, not contaminated** (no free re-ship win); **(c)** the long-audio drop is
   low-value and not severe-concentrated; **(d)** empties are capped by the frozen joint. Conclusion: **v1 is near
   the ceiling of its recipe**; the remaining levers are structural and require a real, gated training run.

---

## 5. The methodology lessons (the part worth internalizing)

1. **Faithful validation gate — the most expensive lesson.** Early on, two independent agents both validated a
   Nemotron submission with the *same hand-rolled decode script* and got ~25% CER; the real submission scored
   **51.5%**. Identical wrong number from a different submission proved the cause was **deterministic and in the
   prediction content**, not the worker. **Redundancy without independence of *method* adds no safety.** Rule now:
   *a submission may be uploaded only after the organizers' EXACT pipeline (real `local_decode.py` both passes →
   `evaluate.sh`) reproduces the claimed metric on Dev. Proxies are for exploration only; never to sign off.*
2. **Proxies are fine for *shape*, not for *ship*.** In Phase 0 we deliberately used a fast offline-transcribe
   proxy to read the *convergence curve shape* and to *compare models to each other* — valid uses — while
   explicitly reserving the faithful harness for any ship decision.
3. **Measure before you "fix."** We nearly removed beneficial checkpoint averaging on a reasonable-sounding
   theory. The data said the opposite. (Chesterton's Fence: understand why something exists before changing it.)
4. **State predictions before acting; stop on surprise.** When a result contradicts your model, your model has a
   false premise — debug the understanding, not reality. (The SOS bug, the averaging finding, and the convergence
   flattening all came from taking surprises seriously instead of explaining them away.)
5. **Distinguish belief from verification, and projection from promise.** Several strong-looking numbers were
   *projections* (e.g. "severe-set 23.6% should be mid-teens on representative Dev") clearly labeled as such until
   the faithful harness confirmed them.
6. **Cost discipline.** GPU pods are expensive: do all code/script prep locally first, define the exact
   stop-condition and decision metric *before* starting a pod, copy artifacts back, and stop immediately.

---

## 6. Where we landed & the open levers

- **Shipping artifact:** the Nemotron **encoder-only + int8** submission, **Dev-500 CER 8.76%**, validated on the
  official harness — our best entry, comfortably under the Zipformer Test1 reference (23.44%).
- **Phase 0 verdict:** v1 is near its recipe ceiling; the cheap wins (re-ship a single checkpoint, just train
  longer) are exhausted.
- **Remaining levers, all requiring a real gated run:** (1) **low-LR unfreeze of the joint/pred-net** (the only
  thing that can move the empty floor and hardest speakers); (2) **data** — speed perturbation (cheap, missing in
  v1) and severity-aware sampling (payoff is split-dependent, so measure); (3) a **fresh longer LR schedule** if
  testing convergence properly. Each must clear the pre-registered, paired, speaker-block-bootstrapped faithful gate.

---

## 7. Mini-glossary

- **CER / WER** — character / word error rate (edits ÷ reference length). CER is Track 2's primary metric.
- **RTF** — real-time factor (compute time ÷ audio duration); <1 means faster than real-time.
- **RNN-T (transducer)** — encoder + prediction network + joint; emits tokens *and* "blank" to advance time.
- **Blank / SOS** — the transducer's "emit nothing / advance" symbol; also used to prime the decoder at start.
- **Cache-aware streaming** — encoder keeps left-context state across chunks so streaming ≈ offline accuracy.
- **Greedy vs. beam search** — single-best vs. multi-hypothesis decoding.
- **SWA / checkpoint averaging** — averaging model weights across checkpoints for a flatter, better-generalizing minimum.
- **Dynamic int8 quantization** — weights stored int8, activations quantized at runtime; near-lossless, faster on CPU.
- **SpecAugment** — masking time/frequency bands of the spectrogram during training for robustness.
- **Speaker-disjoint split** — train/val share no speakers, so validation reflects unseen-speaker generalization.
- **TTFT / TTLT** — time to first / last visible token (the two latency metrics).
- **Faithful harness** — the organizers' exact decode+score pipeline; the only valid basis for a ship decision.
- **Etiology** — cause of the speech disorder (ALS, Cerebral Palsy, Down Syndrome, Parkinson's, Stroke); severity varies by etiology.
