# Research Summary

> Auditor's note: this synthesizes ~41 numbered research docs, handoffs, and experiment logs
> in the repo as of 2026-07-20. Numbered docs `research/NN` are the primary evidence trail;
> `PLAN.md` is stale (session-1). Where I state a metric, it is the *faithful* number (official
> `evaluate.sh` / sclite min-over-two-refs) unless flagged as a proxy.

## Project Goal
- Compete in **SAPC2 Track 2 — Streaming ASR on dysarthric speech** (Speech Accessibility Project Challenge 2).
- **Win condition:** sit on the **CER × latency Pareto frontier** of the sequestered `test2` set. Prize split equally among all Pareto-frontier teams.
- Hard constraints: **CPU-only inference**, **15000 s/submission** budget, chunked streaming interface (100 ms / 1600-sample chunks), Docker `xiuwenz2/sapc2-runtime:latest`.
- Metrics: **CER primary** (WER secondary), clipped at 100%/utt, min over two refs (with/without disfluency); **latency = TTFT + TTLT**, Pareto rank uses mean(TTFT, TTLT).

## Current Pipeline
- **Submission contract:** a `Model` class with 5 methods (`__init__`, `set_partial_callback`, `reset`, `accept_chunk`, `input_finished`), one per submission dir under `track2_starting_kit/`.
- **Two ingestion passes** (`local_decode.py`): Pass 1 batch/multiprocess → accuracy CSV; Pass 2 real-time 100 ms pacing → latency JSON.
- **Scoring is delegated, never reimplemented** — `evaluate.sh` → `steps/eval/*` → `utils/compute_metrics.py` (accuracy), `utils/compute_latency.py` (latency). House rule: wrap, don't modify.
- **Faithful-harness gate (learned the hard way):** a submission is validated only by the organizers' exact `local_decode.py` + `evaluate.sh` on Dev. Hand-rolled decode proxies caused a false negative once (Nemotron post-mortem 2026-06-24) — proxies are for exploration only.

## Models & Methods
- **Zipformer (A1) — the incumbent baseline / default ship.** ~66–70M, streaming-native (icefall streaming Zipformer), **fully fine-tuned on SAP dysarthric data** (16 ep, speed-perturb + SpecAug). Greedy → **modified beam search (paths=4)** upgrade measured.
- **Nemotron 0.6B cache-aware FastConformer-RNNT** (~618M, ~10× params). Encoder-only fine-tune, joint frozen; int8 encoder + fp32 decoder; ONNX export. **Submitted, lost to A1.**
- **Parakeet TDT-0.6b-v2 / RNNT-0.6b** — benchmarked zero-shot (most recent work, 2026-07-14). Offline-only (full-context `att_context=[-1,-1]`), not natively streamable.
- Adaptation levers tried on Nemotron: full encoder FT (v1), encoder-only + aug + severity (Arm A), **PEFT LinearAdapters** (armC), cheap decode-time fixes (blank-penalty, gain-norm). Contemplated but **not** run: joint-unfreeze (Arm B), RMS energy-normalization in Pass 1, ILM subtraction, TTS-synthesized dysarthric augmentation.

## Experiments Completed
| ID / doc | Purpose | Model | Data | Result | Conclusion |
|---|---|---|---|---|---|
| research/09 (E1) + Codabench | beam vs greedy decoding | Zipformer A1 | Dev 2k ruler → **Test1** | Dev: greedy 21.6% → beam-4 19.0%; **Test1: greedy 23.44% → beam-4 21.28%**, latency ≈ lower | **Beam-4 = strict win, SHIPPED** (−2.2 Test1 CER, latency-negative, 0 retrain) |
| research/12,15 | characterize Nemotron failures | Nemotron zero-shot / A1 | Dev_diag-425 | zero-shot 47.5% CER/25% empty; failures severity-driven, inverse to duration; empties = *confident blank* | Failure is **domain/severity**, not size/quant/streaming-bug |
| research/20,24,32 | v1 encoder-only FT + int8 gates | Nemotron v1 | Dev-500 / Dev_diag | rep-Dev **8.76%** (int8); severe Dev_diag **24.85%** | Big rep-Dev win; severe tail is the binding constraint |
| int8 `_t1` submission | real Codabench Test1 | Nemotron int8 | Test1 (10521) | **CER 27.97% / WER 37.71%** — timeout fixed (threads=1) | **Loses to A1 (23.44%)**; Test1 is severe-heavy |
| research/39 (Arm A) | enc-only + aug + severity | Nemotron | Dev_diag | 27.35% (worse than v1) | FAILED |
| research/41 (armC) | PEFT adapters, base frozen | Nemotron | Dev_diag | **28.19% (worst arm), empties ↑** | FAILED — PEFT hypothesis falsified |
| exp_nemotron_speed_001–003 | CPU RTF / thread topology | Nemotron ONNX int8 | pod real audio | threads=1 optimal under 20-worker Codabench topology; RTF 0.017 | Fixed the timeout; runtime, not accuracy |
| Parakeet bench (docs/results) | is zero-shot Parakeet better? | TDT/RNNT-0.6b, fastconformer_streaming | Dev_rand300 / Dev_streaming | offline 16–22%; **zipformer streaming 17.25% beats them**; streamable NVIDIA zero-shot 38% | **No** — dysarthric FT beats raw capacity |

## Results So Far
- **Best deployable model = fine-tuned Zipformer A1 with beam-4 — now a MEASURED Test1 point on Codabench.**
  - **beam-4** (submission 846354, 2026-07-13): Test1 **CER 21.28%** / WER 29.5, TTFT 1365 ms, TTLT 93 ms.
  - **greedy** (submission 806053, 2026-06-21): Test1 CER 23.44% / WER 31.51, TTFT 1438 ms, TTLT 95 ms.
  - Beam-4 **Pareto-dominates** greedy on *both* axes (lower CER *and* lower latency) — strict upgrade, zero retrain. This is the live #1 / recommended submission.
  - Note: the research/09 beam prediction (~21%) held on real Test1 (21.28%) — the offline→Test1 forecast was accurate here, in contrast to Nemotron's Dev→Test blowup.
- **Nemotron (10× params) lost:** Test1 27.97% > A1 23.44%. All rescue arms failed the faithful gate; v1 (23.58% Dev_diag) is the best Nemotron artifact and still loses.
- **Parakeet does not replace the baseline** on accuracy or latency (zero-shot, general-English).
- **Beam-4 helps most on the hardest etiologies** (CP, Stroke, DS) — wider search recovers uncertain acoustic frames.

## Repository Structure
- `track2_starting_kit/` — submission dirs; `streaming_zipformer/` is the pristine always-working baseline (config.yaml exposes chunk_size, left_context, beam width).
- `utils/`, `steps/`, `evaluate.sh`, `preprocess.sh`, `local_decode.py` — **organizer scoring pipeline; do not edit semantics.**
- `research/00–41` + `research/LOG.md` — the primary evidence/decision trail (chronological).
- `investigations/` — facts-vs-theories deep dives (esp. `nemotron_vs_zipformer.md`, `h7_fix_plan.md`).
- `experiments/` — per-run artifacts + `summary.csv` (only the speed/runtime rows are filled; accuracy exps live in `research/`).
- `benchmark/` + `docs/` — Parakeet benchmark scaffold + results (most recent work).
- `artifacts/` — packaged submission artifacts (nemotron int8).
- `track1_starting_kit/` — Track 1 kits (canary_qwen, parakeet, whisper); not the focus.
- Many root `*_HANDOFF*.md` / `STATE_OF_WORK.md` / `TEAM_BRIEF*.md` — agent handoffs; `STUDY_PLAN.md` is Q's learning curriculum.

## What's Working
- ✓ Zipformer A1 baseline: streaming-native, CPU-proven, dysarthric-FT, submitted and #1-worthy.
- ✓ Beam-4 decoding upgrade — **shipped and confirmed on Test1 (21.28% CER, −2.2 vs greedy, latency-negative)**; Pareto-dominates greedy A1.
- ✓ int8 quantization ≈ lossless (fp32 8.64% vs int8 8.76% Dev-500).
- ✓ CPU runtime / timeout solved (threads=1 under 20-worker topology; 44% budget margin).
- ✓ Faithful evaluation discipline + severity-enriched Dev_diag as the real Test predictor.

## Current Limitations
- ⚠ **No exp_000 baseline row was ever recorded via the official harness** — the pristine zipformer baseline's own Dev CER/latency through `evaluate.sh` is referenced but `experiments/summary.csv` accuracy columns are empty; accuracy results are scattered across `research/`, not the tracking scaffold.
- ⚠ **Dev→Test transfer is unreliable** (Nemotron slope +18 vs zipformer +1.8) — Dev gates can green-light a model that loses on Test.
- ⚠ Parakeet numbers use a **proxy scorer** (dual-ref macro-mean), *not* official sclite; no zipformer row on the exact same subset yet.
- ⚠ **Severe tail is the accuracy floor** (ALS/CP/DS ~34% CER, empties concentrated in ALS ~20%) and remains unsolved.
- ⚠ Heavy technical debt: 15+ overlapping root markdown docs, stale `PLAN.md`, redundant handoffs — hard to reconstruct current state without reading `research/` chronologically.

## Open Questions
- **H7 (the deepest, least-investigated):** *why* does a finetuned Nemotron score **18.46% offline (`transcribe`) but 28.19% through cache-aware chunked streaming export** — same weights, ~10-pt collapse? Cache-normalization vs full-sequence forward? att_context regime at export? Chunk size? No arm has targeted this; every fix so far touched the *encoder* and died on the export path.
- **H6 (unresolved):** are empties frozen-joint-bound, or encoder energy-sensitivity? Contradictory single-run evidence; the discriminating experiment (joint-unfreeze, Arm B) was contraindicated and never run.
- Is a **converged** encoder-only retrain (8–15 ep vs v1's 4, multi-lookahead + severity sampling + speed-perturb) worth GPU? Code fixes exist but were never run (research/33). Value depends on whether v1 loss was actually flat by epoch 4 — never verified.
- Would **cache-aware fine-tuning of a streamable FastConformer-RNNT** on dysarthric data beat the zipformer's 17.25% streaming CER? Unproven, expensive, same NVIDIA-streaming path that already lost.
- Does the **energy/quiet-audio** artifact (`normalize=NA`, empties ~2.4× quieter) admit a legal Pass-1 RMS-normalization win? Estimated small (~0.2 pt) but untested cleanly.

## Suggested Areas for Further Investigation
1. **Zipformer track, not Nemotron.** Beam-4 is already live (Test1 21.28%, the current #1); the zipformer is the higher-EV vehicle for further gains. Nemotron encoder-rescue is evidence-backed **STOP** (3 independent approaches failed).
2. **Attack H7 directly if any NVIDIA GPU is spent** — diagnose the offline→streaming export gap itself (a *decoder/joint or export-regime* change), not another encoder variant. This is the single unexplored lever and would explain the whole Nemotron loss.
3. **Push zipformer accuracy on the severe tail** (the actual Pareto lever): beam-8 latency check, RNN-LM shallow fusion (research/09 "next"), mild dysarthric augmentation, severity-aware sampling — on the model that already transfers well (+1.8 slope).
4. **Latency-axis Pareto sweep** on the zipformer (chunk_size / left_context / emission policy) — cheap, no retrain, directly moves the scored axis; barely explored.
5. **Close the process gaps:** record an official exp_000 baseline row; add a same-harness zipformer row to the Parakeet table; consolidate the stale root docs so state is reconstructable.

## Evidence References
- Objective / contracts / metrics: `Track2.md`, `SAPC-template/CLAUDE.md`, `utils/compute_metrics.py`, `utils/compute_latency.py`.
- Zipformer baseline + beam-4 win: `track2_starting_kit/streaming_zipformer/config.yaml`, `research/09_decoding_results.md`.
- Nemotron loss + full facts-vs-theories: `investigations/nemotron_vs_zipformer.md`, `research/12/15/20/24/32`, memory `int8-submission-status`.
- Failed rescue arms: `research/39_v2_armA_guardrail_FAILED.md`, `research/40_independent_review_and_pivot.md`, `research/41_v2_peft_armC_FAILED.md`.
- Convergence/accuracy levers (unrun): `research/33_accuracy_improvement_review.md`.
- Parakeet benchmark: `docs/results/parakeet_comparison.md`, `benchmark/README.md`, `docs/benchmark_plan.md`.
- Runtime/timeout: `experiments/summary.csv`, `experiments/exp_nemotron_*`, memory `eval-worker-cpu-confirmed`.
- H7 next-steps: `investigations/h7_fix_plan.md`, `STUDY_PLAN.md` (M2d).
- Stale-state caveat: `PLAN.md` (session-1, superseded by `research/37–41`).
