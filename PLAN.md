# PLAN.md — SAPC2 Track 2 Roadmap

Living roadmap. Reorient here at the start of every session. Update after every experiment.
Companion docs: `RESEARCH_NOTES.md` (why), `EXPERIMENT_LOG.md` + `experiments/` (results),
`CLAUDE.md` (contracts you must not break).

---

## North star
A Track-2 streaming submission on the **CER × latency Pareto frontier**, CPU-only, within the
15000 s budget. Win = sit on the frontier of test2. Levers, in expected ROI order:
**(1) in-domain fine-tuning → (2) augmentation → (3) decoding/latency tuning → (4) CPU quant/ONNX.**

## Current State (session 1 — 2026-06-19)
- ✅ Repo mapped; all organizer contracts read and documented (`CLAUDE.md`).
- ✅ Foundation docs created: this file, `RESEARCH_NOTES.md`, `EXPERIMENT_LOG.md`, `experiments/`.
- ⛔ **Baseline NOT executed.** Blockers: (a) no SAP data on host; (b) `setup.sh` k2/kaldifeat
  wheels are linux x86_64 / cp311 — won't install on this darwin box. Needs Docker image
  `xiuwenz2/sapc2-runtime:latest` or a linux x86_64 CPU host.
- ⛔ Untouched organizer code (intentional — Chesterton's Fence).
- ➡️ **Next gate**: secure data + linux/Docker host → run exp_000 baseline reference.

## Current State Update (Nemotron gate — 2026-06-23)
- ✅ Codabench topology clue incorporated: previous ingestion log reported `Number of workers: 20`.
- ✅ Offline reproduction completed in `exp_nemotron_speed_002`: 20 workers constrained to 20 logical CPUs.
- ✅ Result: `SAPC2_THREADS=1` is the correct batch-worker default under this topology. It was about 4.2x
  faster wall-clock than the submitted/default `threads=4` on the 120-row sweep.
- ✅ Exact-hypothesis hash audit found zero differences between `threads=1` and `threads=4` on a 40-row slice.
- ✅ Runtime-fix patch applied to extracted package at `/tmp/nemo_submission_codex_profile/model.py`; review artifacts
  saved in `experiments/exp_nemotron_speed_002/`.
- ✅ Local packaging helper written: `scripts/package_nemotron_runtimefix.py`. Its dry-run currently refuses
  safely because `/private/tmp` has only about 1.29 GiB free versus 1.64 GiB required with margin.
- ✅ Runtime-fix zip built on RunPod and guardrailed in `exp_nemotron_runtimefix_003`.
- ✅ Packaged candidate path:
  `/workspace/finetune/eval/nemotron_runtimefix_codex/artifacts/nemo_submission_worker1_runtimefix.zip`
  (803.3 MiB, SHA-256 `6fb803e08ee88385bcd7ca4348d6475c95d27f4900ac375916e76b1edd5a69f4`).
- ✅ Old-vs-runtimefix hash parity passed on 120 rows: 0 SHA/length/status diffs.
- ✅ Runtime-fix 20-worker 500-row guardrail passed: 500/500 OK, aggregate RTF 0.0172, throughput 58.09x.
- ⛔ Do **not** submit the old `SAPC2_THREADS=4` default package.
- ➡️ **Next gate**: submit the runtime-fix zip to Codabench, then compare Test1 CER/WER/latency against the
  failed Nemotron submission and A1 Zipformer baseline.

## Current State Update (small-model sweep — 2026-07-24)
**Context:** tested Q's "smaller = faster/better-generalizing model" hypothesis end-to-end on a live H200
pod (now STOPPED). All zero-shot, Dev_streaming (123u), proxy scorer. Full data: `research/46` §6;
memory `small-model-sweep-verdict`.

- ✅ **Sweep complete.** Zero-shot CER%: parakeet_realtime_120m@[70,1] **31.8** (best, 80ms) · FC-114M@480ms 33.7
  · zipformer-70M baseline 36.2 · FC-114M@1040ms 37.3 · FC-114M@default 38.1 · FC-114M@80ms 54.8 ·
  **zipformer-20M 68.7** · **FC-32M@80ms 72.4** (Dev_diag severe 80.6). Ref FT-zipformer 17.25 proxy / 12.14 official.
- ✅ **Hypothesis FALSIFIED.** Small models collapse in BOTH families (capacity is the binding constraint on
  dysarthric speech). No architecture is inherently better; the zipformer's edge is fine-tuning (~19 pts).
- ✅ **60M FastConformer does not exist** (only 32M medium + 114M large streaming hybrids published).
- ✅ Built `track2_starting_kit/fastconformer_medium32/` (Codabench-shaped sibling; NeMo streaming call still
  VERIFY-ON-POD — E1 actually ran via the pod's `stream_infer.py`, not this wrapper).
- ⚠️ Method scar: sherpa `weights/standard` == `weights/finetuned` by md5 (the 17.25% is FT, not zero-shot;
  true zero-shot is the `baseline` variant). Parakeet needs `<EOU>` stripping before scoring. Trust checksums, not labels.

### Plan from here (decide next session)
1. **DEFAULT (safe, no new spend):** ship the fine-tuned zipformer **beam-8** upgrade (Dev_streaming 11.62% vs
   shipped beam-4 11.62/12.14; Pareto-dominates at equal latency — memory `a1-provenance-and-beam8`). This is
   the banked win; small-model detour did not beat it.
2. **OPTIONAL BET (needs explicit Q go + fresh GPU pod):** fine-tune **parakeet_realtime_eou_120m** for a
   low-latency Pareto corner — only model hinting at a viable 80ms play (31.8% zero-shot → maybe ~14–17% FT).
   Shelf-ready spec in `research/46` §7. **Binding caveat:** same NeMo-streaming-FT transfer wall that sank
   Nemotron (research/37 §6, memory `nemotron-vs-zipformer-roadblock`); gate on held-out-severe; Dev win ≠ submit.
3. **DO NOT** revisit small models or the 32M/20M path — closed on evidence.
- ➡️ **Next gate:** Q chooses (1) package+submit beam-8, or (2) greenlight the parakeet-FT experiment. All sweep
  numbers are proxy-scorer (ranking only); any submission still needs official `evaluate.sh` sign-off.

---

## Phase 0 — Foundation & contracts  *(DONE)*
Goal: understand and lock the evaluation pipeline so nothing downstream breaks it.
- [x] Map repo, read interface + accuracy + latency + manifest code.
- [x] Document contracts (`CLAUDE.md`), research (`RESEARCH_NOTES.md`), tracking scaffold.
- [ ] Confirm with Q: data availability + execution environment (Docker vs linux host).

## Phase 1 — Baseline reproduction  *(BLOCKED on env/data)*
Goal: a trustworthy reference point. "Reproduce baseline Dev CER/WER + TTFT/TTLT, recorded as exp_000."
- [ ] Stand up `sapc2-runtime` Docker (or linux CPU host); run `streaming_zipformer/setup.sh`.
- [ ] Decode Dev via `local_decode.py`; score with `evaluate.sh` (stages 0–3).
- [ ] Record exp_000 (CER/WER/TTFT/TTLT/RTF) + git hash + config snapshot.
- [ ] Compute **implied RTF budget**: total Dev/test duration ÷ 15000 s → headroom for beam/quant.
- Output: filled exp_000 row; a working, repeatable decode+score command.

## Phase 2 — Data understanding (EDA)  *(needs data)*
Goal: know the corpus before modeling; build anti-overfit validation.
- [ ] Distribution of `etiology`, `speaker`, `duration`, transcript length; OOV vs BPE-500.
- [ ] Build a **speaker-disjoint internal dev split** from Train (test speakers are unseen).
- [ ] Per-etiology / per-speaker baseline CER to find where the model fails most.
- Output: `experiments/eda/` plots + notes; a fixed internal val split committed.

## Phase 3 — In-domain fine-tuning  *(highest ROI — now evidence-backed)*
Goal: close the LibriSpeech→dysarthric domain gap. **Confirmed by SAPC1**: every top-5 team
fine-tuned a foundation model; FT cut WER 17.82→8.11. Transducers beat SSL/Whisper on SAP.
See `RESEARCH_NOTES.md` §7 for the winner's full recipe.

**Recipe to port from the SAPC1 winner (Takahashi'25):** full FT (do NOT freeze decoder/joint) ·
all data/all etiologies (not disease-specific) · forced-align sentence segmentation of long audio ·
SpecAugment + speed-perturb 0.9–1.1 · 4-checkpoint weight averaging · beam/temperature tuning under
the 15000 s budget. Watch the repetition/stutter failure mode (with-disfluency ref).

**Model shortlist — evidence-derived in `research/05_synthesis_and_candidates.md`** (full survey in
`RESEARCH_PLAN.md` + `research/01–04`). Key lesson: the model MUST be **streaming-native** — offline SOTA
loses +1.9–77% WER when chunked; cache-aware streaming loses ~0.2%.
- **C1 (DO FIRST, anchor)**: full FT of the icefall streaming Zipformer-66M on SAP. Streaming-native,
  CPU-proven (5–9× RT, Q already ran it), int8≈lossless, **documented icefall finetune recipe**. New dir
  `track2_starting_kit/streaming_zipformer_ft/`. Measures the FT gain at zero feasibility risk.
- **C2 (higher-ceiling challenger, GATED)**: NeMo cache-aware streaming FastConformer-Hybrid (medium ~32M
  / large ~114M). Cache-aware ⇒ offline≈streaming accuracy; SAPC1-winner family; multi-latency Pareto
  points. **Gate before FT**: benchmark pretrained CPU-RTF + zero-shot SAP CER on the eval container.
- **C3 (PARKED)**: 600M Nemotron streaming — only if eval CPU has many cores (paper used 32) + int4 ONNX.
- **Out**: offline Parakeet-TDT/Qwen/Whisper-large (chunk-degrade / not CPU-RT), SSL (weak on SAP),
  LLM-GER (kills TTLT/budget).

Tasks:
- [ ] Build speaker-disjoint internal val (Phase 2); never tune to Train/Dev *text* (eval is "unshared").
- [ ] **exp_001 = C1** fine-tune on SAP Train; compare CER/WER + latency vs exp_000 on internal val.
- [ ] Ablate augmentation (SpecAugment, speed-perturb) and checkpoint averaging.
- [ ] If C1 gain is strong but ceiling-limited, run **exp_00x = C2** (NeMo streaming FastConformer).
- [ ] Promote only models that beat the prior best on speaker-disjoint val (not LB noise).
- Output: exp_001+ with CER delta vs baseline; pristine baseline dir untouched.

## Phase 4 — Streaming & latency tuning  *(no retraining)*
Goal: move along the latency axis cheaply via `config.yaml` + emission policy.
- [ ] Sweep chunk_size, left_context, greedy vs beam (paths 2/4/8), input_finished tail length.
- [ ] Partial-emission policy: emit first non-empty partial ASAP (TTFT keys on it).
- [ ] Plot CER vs mean(TTFT,TTLT); pick frontier points.
- Output: a CER–latency curve; chosen operating point(s).

## Phase 5 — CPU / quantization optimization
Goal: hit the time budget with margin; ideally also lower latency.
- [ ] int8 dynamic quantization (PyTorch) → CER delta + RTF.
- [ ] ONNX Runtime / sherpa-onnx export with int8 → compare RTF + CER + integration cost.
- [x] Thread tuning for the known Codabench accuracy-pass topology: 20 workers favors 1 thread per worker.
- [ ] Patch the submitted Nemotron runtime so worker processes default to 1 thread while local profiling can override.
- Output: a submission that is comfortably under 15000 s with best CER at chosen latency.

## Phase 6 — Advanced modeling  *(stretch, only if frontier still has room)*
- [ ] Etiology-aux multitask (train-time only — no test-time etiology in the interface).
- [ ] Shallow-fusion LM for rare-word recovery; tokenizer/BPE retrain on SAP text.
- [ ] Distillation to a smaller encoder if CPU budget is the binding constraint.

## Phase 7 — Consolidation & submission
- [ ] Pick 1–N frontier configs; package each (`model.py`+`setup.sh`+weights) per `CLAUDE.md`.
- [ ] Dry-run the exact Docker submission path; verify time budget on full test1 size.
- [ ] Keep ≥1 valid public LB submission at all times (organizer rule).

---

## Experiment discipline (every run)
Before: write **hypothesis**, metrics to compare, risk (esp. overfit on tiny dysarthric data).
After: fill `experiments/exp_XXX/` + a `summary.csv` row + an `EXPERIMENT_LOG.md` entry; compare
explicitly to the prior best; decide keep/reject; record the single next question.

## Decisions made (session 1)
1. **Data**: SAP corpus lives on a **remote linux host** (not this mac). Need the `DATA_ROOT` path.
2. **Compute**: run via **Docker `xiuwenz2/sapc2-runtime:latest` on linux**. (FT will need a GPU
   arranged separately; Track-2 eval is CPU-only inside the runtime image.)
3. **Refactor**: **minimal** — keep organizer files pristine; each new model = a sibling dir under
   `track2_starting_kit/`. No big package restructure.

## Open for Q (blocks autonomous execution)
- **Host access**: do I get a shell on the remote linux host (or run inside the Docker container)
  directly, or do I prepare scripts for you to run there? This determines whether I execute exp_000
  myself or hand you a turn-key harness.
- **Paths on host**: `DATA_ROOT`, project checkout location, and where SCTK should install
  (the `DATA_ROOT`/`PROJ_ROOT`/`SCTK_DIR` placeholders in `evaluate.sh`).
