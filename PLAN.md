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

## Current State Update (parakeet FT — Arm A + AdaLoRA — 2026-07-25/26)
**Ran option (2) above** on pod `1ppb7l0i5xuna8` (H100, now STOPPED). Full detail: `EXPERIMENT_LOG.md`
`exp_parakeet_ft`, memory `parakeet-ft-wrapper-streaming`. Base = `nvidia/parakeet_realtime_eou_120m-v1`.

**Results (faithful `local_decode` harness, real scorer):**
| Model | Dev_streaming CER | TTFT p50 | TTLT p50 | Dev_diag SEVERE CER | severe empties |
|---|---|---|---|---|---|
| **parakeet Arm A** (encoder-only FT) | **13.18%** | **640 ms** | **74 ms** | **29.96%** | **48/425 (11.3%)** |
| parakeet AdaLoRA (merged) | 17.51% | 640 ms | 74 ms | — | — |
| zipformer cs8 | 13.85% | 926 ms | — | — | — |
| zipformer beam-8 | 11.62% | 1157 ms | — | — | — |
| int8 zipformer | — | — | — | ~24.85% | — |

- ✅ **Low-latency corner is REAL:** Arm A Pareto-DOMINATES cs8 (better CER *and* TTFT). The earlier scary
  TTFT (1711ms) was a torch-thread oversubscription artifact (pod nproc=128 vs 13.6-CPU cgroup quota);
  wrapper now caps threads to the quota (committed) → clean 640ms. Confirmed real-time (26ms/chunk).
- ⚠️ **VERDICT MIXED, not a clean ship:** severe tail 29.96% w/ **11.3% empties** loses to zipformer (~24.85%),
  and **Test1 is severe-heavy** → Dev_streaming's 13.18% likely won't project to Test. Plan risk#2 realized
  (frozen joint → confident empties). AdaLoRA strictly dominated (same latency, worse CER) → **shelved**.

### Plan from here (parakeet line — decide next session)
0. **ERROR-ANALYSIS BATTERY — DONE 2026-07-26 (EXPERIMENT_LOG `exp_parakeet_ft_empties`; pod STOPPED).** Ran the
   full 5-theory battery on the 48 severe empties. Verdict: empties are **PARAKEET-SPECIFIC, not data-bound** — on
   the SAME 48 utts the fine-tuned **zipformer transcribes 28/48 (58%), 31% at CER<0.5, 9 PERFECT, mean CER 66%
   vs parakeet's 100%**. Ruled out streaming-gap (offline recovers only 21%) and EOU misfire (0). Cause = parakeet
   blank-propensity + `normalize='NA'` level sensitivity (the empty-dominated speaker sits ~10dB low). Free
   gain-norm is only a partial fix (21%→29% offline). **This is the mechanism of parakeet's severe-tail loss.**
1. **STRATEGIC CALL (supersedes "Arm B is next"):** parakeet's severe-tail loss is a self-inflicted pathology the
   banked zipformer never had; parakeet's only edge is latency (640 vs 926ms). To beat the zipformer on the
   severe/Test-heavy tail it needs BOTH input-normalization AND a joint-unfreeze (Arm B) surviving Dev→Test — two
   bets vs a model already ahead for free. **EV favors banking the zipformer** for the Test-predictive tail;
   parakeet is a latency-only play unless Q judges the low-latency corner worth the two-fix gamble.
2. **IF pursuing parakeet:** first add a **capped** input RMS-norm (+15-20dB cap + energy floor, to avoid the
   noise-hallucination failure mode seen at +39/+42dB) to the wrapper, validate through the REAL streaming harness
   (offline≠streaming here), THEN decide **Arm B — joint/pred-net unfreeze** (`nemo_finetune_v2.py --freeze
   joint_unfreeze`) on a hard Dev→Test gate. Arm B is now re-motivated by the blank-propensity finding, but the
   Nemotron scar (added joint capacity → Test regression) still binds. ~2.5h GPU.

   **UPDATE 2026-07-27 — Test1 board forces the priority (memory `test1-standing-and-pareto`).** We shipped
   `smfoundation` = beam-4 zipformer, now **#4/4 and Pareto-DOMINATED** (yac3xn 18.1%/592ms dominates all;
   we're dominated twice). Verified from the SAPC2 site: rank = **Pareto frontier**, latency = **mean(TTFT,TTLT)**,
   **multiple winners** possible. Parakeet Arm A ≈ **357ms mean** (TTFT 640/TTLT 74) — a 235ms margin under the
   entire field → a non-dominated **frontier-corner win at almost any plausible CER**. Beam-8 zipformer is DEAD
   for the frontier (better CER, worse on the latency axis we already lose). So the low-latency parakeet corner
   is our route on. Q call 2026-07-27: pursue parakeet-minimal; GPU pre-approved.
   - ✅ **Fix 1 (causal input gain) CODED + locally verified** in `track2_starting_kit/parakeet_realtime_ft/`
     (`config.yaml` input_gain block + `model.py` `_compute_gain`/`_extract_features`). Frozen-scalar gain from
     the first 100 ms (adaptive would break the frame-stability/cache invariant — Q chose no-delay freeze).
     Boost-only, target −25 dBFS (matches the offline probe), **+20 dB cap**, −45 dBFS floor; env-overridable
     (`SAPC2_INPUT_GAIN/TARGET_DBFS/GAIN_CAP_DB/RMS_FLOOR_DBFS`) for on-pod sweep. `py_compile` + gain-branch
     math verified locally (quiet→+13dB, normal→+0dB, silent→+0dB). Efficacy needs the pod (no NeMo locally).
   - ⏳ **Fix 2 (O(N²) feature recompute) DEFERRED** — decoupled on purpose. It refactors the streaming-loop's
     absolute-frame indexing; correctness (feature-cache equivalence) is **not locally verifiable** (no NeMo).
     It's a budget/validity fix, not an efficacy fix; the gate can **shard** the accuracy pass across workers
     (per `run_devdiag_norm_full.sh`), so ~17min/425u is tolerable for Dev validation. Fix 2 = a separate
     pre-submit task with its own on-pod numerical-equivalence assertion. Do NOT gate efficacy on it.
   - ⛔ **GATE RAN 2026-07-27 (pod `1ppb7l0i5xuna8`, now STOPPED) — causal gain fix FALSIFIED for the empties.**
     - ✅ **Latency corner HOLDS (the real win):** gain-on Dev_streaming (proxy scorer) = CER **12.99%**, empties
       5/123, TTFT p50 **669 ms** / TTLT p50 **73 ms** → mean ≈ **371 ms**. Non-dominated vs the Test1 board (min
       592 ms). Confirms parakeet's frontier-corner thesis (memory `test1-standing-and-pareto`).
     - ⛔ **Severe empties NOT recovered: 0/48** through the real streaming harness (vs offline probe 14/48=29%).
       Root cause (audio diagnostic, no model): **40/48 empties have a near-silent onset** — median first-100 ms
       = **−58.5 dBFS**, below the −45 floor → the causal freeze-from-first-chunk gain floors to **1.0 (no boost)**.
       Their full-utterance RMS is loud (median −29.6 dBFS), which is why OFFLINE (full-RMS) worked. STRUCTURAL:
       the empties ARE the quiet-onset utts, so any causal first-chunk estimator is blind to their level. A
       speech-onset-gated gain would add latency (kills the corner) AND break the frame-stability invariant, and
       even then caps at the ~29% offline ceiling. **The cheap severe-tail fix is dead.** Fix 1 left default-off.
     - ✅ **Fix 2 (incremental feature cache) is CORRECT** (equivalence gate: predict CSV byte-identical off vs on)
       but **NOT a speedup** (off=123s ≈ on=121s): feature extraction was never the bottleneck — per-chunk CPU
       conformer forward is. The 63-min unsharded Dev_diag was a single-process `gate.sh` artifact; the REAL
       accuracy pass is multiprocess (contract), so the O(N²) "budget risk" was overstated. Keep Fix 2 as free,
       proven-safe insurance for pathological long utts; it is not the budget lever.
     - **VERDICT:** parakeet is a **latency-corner-only** play. Severe CER stays mediocre (~30% / gain-off 29.96%
       control unchanged, since the empties — the main severe-CER driver — don't recover). Per the verified Pareto
       scoring (multiple winners; latency = mean(TTFT,TTLT)), the ~371 ms corner is a legitimate frontier WIN
       **regardless** of severe CER. So the open fork is Q's: (a) ship parakeet Arm A as a low-latency frontier
       entry **alongside** the banked zipformer (captures the unique corner; needs OFFICIAL `evaluate.sh` gate —
       today's numbers are proxy), (b) bank zipformer only, or (c) spend on **Arm B** (joint-unfreeze) as the only
       remaining severe-CER lever — GPU bet, Nemotron scar binds, and it targets blank-propensity not the
       input-level root cause, so uncertain it helps the empties.
   - ➡️ **(historical) original GATE plan:** restart pod `1ppb7l0i5xuna8` (STOPPED, disk
     persists → armA_full/ft_smoke_encoder_only.nemo + SAPC2 data + nemoenv present). Upload updated
     `model.py`+`config.yaml` only. **Part A (efficacy):** Dev_diag severe accuracy pass, `SAPC2_INPUT_GAIN=off`
     (control — must reproduce prior 29.96% / 48 empties, proving the edit is inert when off) vs `=on`; score
     via the REAL `evaluate.sh`/official scorer + empties count. **Part B (corner):** Dev_streaming real
     streaming pass, gain on; confirm CER + TTFT/TTLT still ≈ 357ms mean. **Decision:** does gain-on cut empties
     and move severe CER toward/below the zipformer's ~24.85% while holding the latency corner? Stop pod
     immediately after copying JSON/logs back. Expected pod runtime ~30-45min. Path layout VERIFY-ON-POD first.
2. **BEFORE any submit — fix wrapper O(N²) feature recompute** (`model.py accept_chunk` re-extracts the whole
   raw buffer each call). Invisible in the real-time streaming pass, but made the untimed ACCURACY pass take
   ~17min on 425 utts → a real 15000s/submission time-budget risk on full Test. Cache features incrementally.
3. Optional CER squeeze if Arm B clears severe: beam search (~-3pt historically, ~free latency), int8 encoder.
4. **Fallback if Arm B fails severe:** default remains ship the gated beam-8/beam-4 zipformer; parakeet line closes.
- ➡️ **Assets on pod `/workspace/parakeet_ft/`:** `armA_full/ft_smoke_encoder_only.nemo` (KEEP — ship-candidate
  + base for Arm B compare/beam/int8), `adalora_full/ft_adalora.nemo` (disposable), `gate.sh`, `ttft_probe.py`,
  gate outputs. Code committed: `b921f9f`, `7d28427` (main, unpushed).

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
