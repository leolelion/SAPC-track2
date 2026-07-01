# EXPERIMENT_LOG.md — SAPC2 Track 2

Append-only log. One entry per experiment. Machine-readable mirror: `experiments/summary.csv`.
Every entry must record: hypothesis, config snapshot path, git commit, metrics, conclusion, next.

Metric columns: **CER** / **WER** (Dev, batch/offline) · **CER_stream** (Dev_streaming if measured)
· **TTFT_p50** / **TTLT_p50** (ms) · **RTF** (CPU real-time factor) · notes.

> Latency Pareto figure of merit = `mean(TTFT_p50, TTLT_p50)`. Always report both axes.

---

## exp_000 — Baseline reference (PLANNED, NOT RUN)
- **Status**: blocked — no SAP data on this host; `setup.sh` wheels are linux x86_64 only
  (this box is darwin). Must run inside `xiuwenz2/sapc2-runtime:latest` or a linux CPU host.
- **Model**: `track2_starting_kit/streaming_zipformer` (LibriSpeech streaming zipformer, epoch-30).
- **Config**: chunk_size=16, left_context=128, modified_beam_search, num_active_paths=4,
  features.mode=incremental, input_finished tail=0.3 s. (`config.yaml` snapshot pending.)
- **Hypothesis**: establishes the *out-of-domain* reference CER on dysarthric Dev. Expect
  relatively high CER (LibriSpeech→dysarthric domain gap) — this is the headroom we attack.
- **Metrics**: CER=— WER=— TTFT=— TTLT=— RTF=— (to be filled on first real run)
- **Conclusion / Next**: once measured, this becomes the anchor for all deltas. First improvement
  to attempt: fine-tune on SAP Train (see PLAN Phase 3).

---

## exp_base_ft — TRUE current baseline (existing finetuned zipformer, on pod)
- **Status**: done (measured Session 3). Streaming zipformer-M 66M, ~2-epoch finetune, sherpa-onnx 1.13.3.
- **Metrics** (250 Dev utts, min-two-refs, greedy streaming): **CER 15.7% / WER 22.3%, 0 empty, RTF 0.147**.
  Per-etiology CER: PD 10.3 · Stroke 11.9 · ALS 12.9 · DS 20.3 · CP 27.1.
- **Note**: the "empty output" bug was sherpa-onnx **1.12.35**; fixed by pinning **≥1.13.3**. phase4b
  26%/empties was bug-contaminated. This 15.7% is the real number A1 must beat.

## exp_001 (A1) — ✅ DONE & WINS: speed-perturb + SpecAugment, 16-epoch finetune
- **VERDICT (2026-06-20)**: A1 last-5-epoch average = **CER 21.61% / WER 27.03%** on the 2k ruler,
  vs baseline `standard` **CER 28.45%** → **−6.84 CER pts, 24% relative**. Clear win.
  Trajectory: ep1 29.05 → ep9 22.97 → ep16-plain 23.40 (mild late overfit) → **avg-5 21.61** (best).
  Avg-5 smooths the late overfit (kept all ckpts → could average). Validation LOSS was flat throughout
  (red herring; loss ≠ CER). Winning model persisted (survives pod stop): `/workspace/finetune/onnx/a1/`
  ({encoder,decoder,joiner}.onnx + .int8 + tokens + bpe + a1_train.log). Integrity-verified by decode.
  Pod `3dwiczo41jeg1y` stopped after persist.
- **SUBMISSION PACKAGED + OFFLINE-VALIDATED (2026-06-21)** — the milestone this project never reached:
  - Self-contained Track-2 zip: `model.py` (no omegaconf, config baked in) + bundled ONNX + bundled
    sherpa-onnx wheels + offline `setup.sh` (`pip --no-index`). NO network / NeMo / HF download.
  - **Offline-validated**: extracted the actual zip into a clean venv with network blocked (dead proxies)
    → `setup.sh` installs sherpa from bundled wheel → ingestion harness transcribes correctly. Twice (fp32 & int8).
  - **int8 chosen** (= fp32 accuracy, 3.6x smaller, faster): CER 21.62% (2k) / 13.85% (streaming subset),
    zip **69 MB**. fp32 fallback also built (247 MB). Latency ≈ fp32 (TTFT~1234/TTLT~100/mean~643 ms this pod).
  - Artifacts: pod `/workspace/finetune/submission_a1_int8.zip`; Mac `/Users/o/Downloads/submission_a1_int8.zip`.
  - **Leaderboard context**: only 2 entries, both CER 34.59% on Test1 (vecxoz just matched baseline). Our CER
    crushes that on Dev; Test1 transfer UNVERIFIED (the key remaining unknown — only a real submission tells us).
  - chunk-8 TTFT-reduced variant exists as a future A/B.
- **TEST1 LEADERBOARD RESULT (2026-06-21, participant `smfoundation`) — #1 by a landslide on CER:**
  - **A1 int8: Test1 CER 23.44% / WER 31.51% / TTFT P50 1437.87 ms / TTLT P50 94.64 ms.**
  - Rank #1 vs vecxoz (34.56) and SAPC2-baseline (34.59) — **~11-point CER lead.** Submission ran on the
    offline worker (packaging held). **Dev→Test transfer healthy: 21.62% (2k Dev) → 23.44% (Test1), ~1.8 pt.**
  - **Correction to earlier latency claim**: on the real (slower) worker TTFT = 1437 ms = WORST on board →
    our mean latency (~766) is HIGHER than competitors (~710/724). NOT a Pareto-latency win; we rank #1
    because **CER is the dominant ranking factor**. TTLT edge (95 ms fast finalization) confirmed real.
  - **Strategic read**: ranking is CER-driven → push CER lower. Nemotron (~10% ceiling, see
    [[nemotron-higher-ceiling]]) is the priority next bet (gate on streaming viability). chunk-8 is a
    latency hedge (~641 ms mean but ~1-2 CER worse), only if scoring weights latency more than the board shows.

- **Status**: launched Session 3 (pod `/workspace/finetune/exp/a1_sp`, `run_a1.sh`, log `a1_train.log`).
- **Recipe**: from LibriSpeech epoch-0 (66.1M, causal chunk16/left128, streaming profile UNCHANGED);
  **3-way speed-perturb (0.9/1.0/1.1) = 1,008,225 cuts + SpecAugment**; base-lr 0.0045, lr-epochs 6,
  16 epochs, fp16, max-duration 600, on-the-fly feats, musan off (no cuts). Data: `data_a1/cuts_S`.
- **Hypothesis**: regularized + properly-trained recipe beats the thin ~2-epoch baseline (15.7% CER),
  esp. on the overfit-prone read speech; targets CP/DS (worst etiologies).
- **Start**: epoch1/batch0 loss 0.308, 9.2GB/H200. **Pending**: full run → ckpt-avg → ONNX export →
  official min-two-refs eval vs exp_base_ft.
- **Two pipeline fixes (reproducible)**: `finetune.py` `num_frames`→`duration` filter; `cuts_DEV` symlink.
- **Session-4 update (2026-06-20) — batch-0 "hang" ROOT-CAUSED + FIXED**:
  - The hang was **NOT** in the train step. It is the **TensorBoard `SummaryWriter`** writing event
    files into `--exp-dir`, which sat on `/workspace` (MooseFS). Its async writer thread either
    **stalls** on MooseFS (queue fills → main train thread blocks enqueuing the batch-0 scalar = the
    "hang, GPU idle") or, with the volume quota now full, **errors `EDQUOT`** and crashes. Same line,
    two faces. Full Python traceback pinned it to `event_file_writer.py:_run → record_writer.write → io.open`.
  - **Fix (VERIFIED)**: put `--exp-dir` on local/RAM storage, not MooseFS. Identical smoke config with
    `--exp-dir /dev/shm/...` trained **5 clean epochs** (loss 0.333→0.152, GPU busy, checkpoints saving).
    The train-step path (zipformer attn-entropy hook / autograd / CUDA sync) is healthy.
  - **Also fixed the data-starvation**: precompute features to `/dev/shm` (RAM) + drop `--on-the-fly-feats`.
    Parallel precompute uses spawn executor (validated, no segfault) → `/dev/shm/a1_feats`.
  - **Fixed launcher** staged at `/dev/shm/run_a1_fixed.sh` (exp-dir→`/dev/shm/exp_a1`,
    manifest-dir→`/dev/shm/a1_feats`, no on-the-fly). Could NOT edit pod `run_a1.sh` in place: MooseFS
    quota blocks even a 650-byte write to `/workspace`.
  - **MooseFS quota freed**: deleted `exp/standard2` (7.9 GB, Apr-5 superseded finetune, unreferenced,
    epoch-0 byte-identical to `standard/`, never exported) with user ok. `standard/` + `onnx/standard/` intact.
  - **Status now (relaunched, VERIFIED GPU-bound)**: precompute done (1,008,225 train / 47,929 dev cuts,
    72 GB feats in `/dev/shm/a1_feats`). Full A1 LIVE via canonical `run_a1.sh` (now the fixed version;
    original at `run_a1.sh.onthefly.bak`). Batch 200+, GPU 62 GB, loss converging (tot_loss 0.24→0.20),
    ~1.85 batch/s. Validation = full 47k Dev ×2 at batch0 + every 3000 (user chose to keep as-is).
    Rolling checkpoint backup loop (`/dev/shm/backup_loop.sh` → `/workspace/finetune/exp/a1_sp_backup`,
    keep-last-2 epochs + best-valid) guards against pod restart (exp_a1 is on ephemeral tmpfs).
    ETA: epoch-1 ckpt ~2 h, full 16 epochs ~24–36 h. Next: eval vs 15.7% baseline (exp_base_ft).
  - **Eval ruler built (Session-4)**: 2k speaker-stratified, etiology-balanced Dev subset (400/etiology,
    seed 42, 124 speakers) at `/workspace/finetune/eval/dev_eval_2k.csv`. Decode = sherpa-onnx 1.13.3
    greedy 100ms; scorer = official sapc-nemotron sclite min-two-refs (torchmetrics → `/opt/pylibs`).
    Harness validated: reproduces ~15.7% on the 3–6 s utt bucket (16.6%). Full balanced 2k is harder.
  - **Numbers on 2k ruler (sclite min-two-refs)**: baseline `standard` (~ep2) **CER 28.45% / WER 36.42%**;
    **A1 epoch-1 CER 29.05% / WER 36.68%** (1-epoch FT, ~= baseline as expected). CER rises steeply with
    duration (3–6 s 16.6 → 30 s+ 45.7). Full A1 pipeline (export→decode→score) validated end-to-end on
    epoch-1. Results: `/workspace/finetune/eval/RESULTS.md`. Awaiting full run for the real verdict.
  - **Leaner relaunch (Session-4, ~04:51)**: original run was GPU-underutilized (~31% util, ~28 h ETA).
    Killed at epoch 2 and relaunched with **max-duration 600→1200** (GPU util → ~60%, frames/batch 2×,
    no OOM, 68/140 GB) and **dev validation set shrunk 47,929→2,000 cuts** (`cuts_DEV_full.jsonl.gz` =
    backup; valid pass 47k→2k = ~4 s vs ~minutes; valid_interval not a CLI flag so shrank the set instead).
    Verified ~1.7× throughput → **ETA ~11–15 h**. Same recipe otherwise (sp+SpecAug, base-lr 0.0045, 16 ep).
    Completion watcher + rolling backup armed; on completion: avg ckpts → export ONNX → decode+score 2k
    ruler vs baseline 28.45% → persist → `runpodctl stop pod 3dwiczo41jeg1y` from Mac.

## exp_nemotron_speed_002 — Codabench 20-worker thread sweep
- **Status**: done (2026-06-23).
- **Git**: 16901c6. **Config/artifacts**: `experiments/exp_nemotron_speed_002/`.
- **Hypothesis**: the previous Test1 collapse is likely not warmup; it is batch accuracy-pass oversubscription
  from Codabench's `Number of workers: 20` combined with the current Nemotron default of 4 compute threads
  per worker.
- **Change vs prev**: moved from single-stream pod profiling to organizer-shaped stress testing:
  20 worker processes, model staged on `/dev/shm`, CPU affinity constrained to logical CPUs 0-19, callback no-op.
- **Metrics**: 120-row sweep, 120/120 OK for all settings.
  - threads=1: wall 32.67 s, aggregate RTF 0.029, throughput 34.21x, decode RTF p50 0.285 / p90 0.316.
  - threads=2: wall 60.93 s, aggregate RTF 0.055, throughput 18.35x, decode RTF p50 0.858 / p90 0.995.
  - threads=4: wall 137.09 s, aggregate RTF 0.123, throughput 8.15x, decode RTF p50 2.220 / p90 2.561.
- **Parity check**: 40-row exact SHA audit found zero hypothesis/status/length differences between
  threads=1 and threads=4.
- **Conclusion**: do not submit the current package again. Keep the model, but patch runtime policy:
  batch workers should default to 1 compute thread, and heavy CPU diagnostics should be disabled by default.
- **Patch status**: patched extracted package at `/tmp/nemo_submission_codex_profile/model.py`; saved reviewable
  artifacts `experiments/exp_nemotron_speed_002/model_worker1_runtimefix.diff` and
  `experiments/exp_nemotron_speed_002/model_worker1_runtimefix.py`. Did not rebuild the 820 MB upload zip because
  the local filesystem had only about 1.3 GiB free.
- **Local-first packaging prep**: added `scripts/package_nemotron_runtimefix.py` to build the patched zip from an
  extracted source package without staging a second 900 MB tree. Dry-run currently refuses safely because
  `/private/tmp` has 1.29 GiB free vs 1.64 GiB required with margin.
- **Next**: after freeing disk or packaging on the pod, build the runtime-fix zip and rerun exact-hash parity plus
  the 20-worker throughput guardrail before any Codabench submission.

## exp_nemotron_runtimefix_003 — packaged runtime-fix guardrails
- **Status**: done (2026-06-23). Pod stopped after artifact copy.
- **Config/artifacts**: `experiments/exp_nemotron_runtimefix_003/`.
- **Hypothesis**: the runtime-fix package should preserve transcripts while improving 20-worker startup/throughput
  by defaulting worker processes to 1 compute thread and disabling heavy diagnostics.
- **Package**: built on RunPod at
  `/workspace/finetune/eval/nemotron_runtimefix_codex/artifacts/nemo_submission_worker1_runtimefix.zip`.
  Size 803.3 MiB; SHA-256 `6fb803e08ee88385bcd7ca4348d6475c95d27f4900ac375916e76b1edd5a69f4`.
  Zip manifest clean: 15 files, no cache/backup entries.
- **Parity metrics**: old vs runtime-fix, 20 workers, threads=1, 120 rows:
  - old: 120/120 OK, aggregate RTF 0.0303, throughput 33.02x, decode RTF p50 0.290 / p90 0.339, worker load p50 9.62 s.
  - runtime-fix: 120/120 OK, aggregate RTF 0.0259, throughput 38.55x, decode RTF p50 0.287 / p90 0.319, worker load p50 4.43 s.
  - exact hash comparison: 120 common, 0 SHA/length/status diffs.
- **Throughput guardrail**: runtime-fix, 20 workers, threads=1, 500 rows:
  - 500/500 OK, 0 failed, 86 empty-text hypotheses.
  - total audio 4436.30 s, wall 76.37 s, aggregate RTF 0.0172, throughput 58.09x.
  - decode RTF p50 0.281 / p90 0.318, worker load p50 4.48 s.
- **Conclusion**: runtime-fix candidate is ready for a Codabench submission from a systems standpoint. Remaining risk
  is leaderboard/test distribution, not a known local packaging or throughput failure.
- **Next**: submit the runtime-fix zip, then compare Test1 CER/WER/latency against the failed Nemotron submission
  and the A1 Zipformer baseline.

<!-- Template for new entries:
## exp_00X — <short title>
- **Status**: done | running | blocked
- **Git**: <commit hash>   **Config**: experiments/exp_00X/config.yaml
- **Hypothesis**: <what should improve and why>
- **Change vs prev**: <one-line diff from the experiment it builds on>
- **Metrics**: CER=.. WER=.. CER_stream=.. TTFT_p50=..ms TTLT_p50=..ms RTF=..
- **vs baseline/prev**: CER Δ=.. latency Δ=..
- **Conclusion**: kept | rejected — why
- **Next**: <single next question this raises>
-->
