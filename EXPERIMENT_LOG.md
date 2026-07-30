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

## exp_parakeet_ft — parakeet_realtime_eou_120m dysarthric fine-tune (Arm A + AdaLoRA)
- **Status**: done (2026-07-25, pod 1ppb7l0i5xuna8). Base = nvidia/parakeet_realtime_eou_120m-v1 (EncDecRNNTBPEModel,
  cache-aware streaming [70,1]/80ms). Data: full SAP Train 331k utt / 875 spk, speaker-disjoint val 2k/7spk.
- **Wrapper (part of the model) — 3 bugs found + fixed, verified through the REAL local_decode.py**:
  1. change_decoding_strategy: NeMo 2.7.3 needs positional decoding_cfg; RNNT streaming crashed step-2
     (`'dict'.extend` on Hypothesis.timestamp) → fix = greedy_batch + compute_timestamps=False + preserve_alignments=False.
  2. feature feeding: old wrapper ran preprocessor on raw 100ms chunks → 0 tokens (base AND FT). Rewrote to feed
     FEATURE frames (ported CacheAwareStreamingAudioBuffer): chunk/shift=[9,16], pre_encode_cache=[0,9], drop_extra=2.
  3. AdaLoRA CUDA-init crash: get_peft_model split-device → pin_memory fork crash → fix m.to("cpu") + pin_memory=False.
- **Arm A (encoder-only FT, 4 ep, ckpt-avg)**: val CER 27.5%. **Faithful streaming Dev_streaming CER = 13.18%** /
  WER 16.94%, empty 5/123. Per-etiology CER: CP 18.99, ALS 14.3, Down 14.52, Stroke 14.78, Parkinson 6.33.
- **AdaLoRA arm (true adaptive-rank via HF peft, init_r12→target_r4, 0.836M adapters, merged→streaming-safe)**:
  val CER 31.3%. **Dev_streaming CER = 17.51%** / WER 22.87, empty 5/123 (CP 29.46, Stroke 25.8 — capacity-limited
  on hard etiologies). Verdict: technique validated + streaming-safe merge, but trails Arm A on measurable CER.
- **Latency (RESOLVED 2026-07-26 with capped wrapper)**: the earlier local_decode TTFT p50 1711ms was a
  MEASUREMENT ARTIFACT — pod exposes nproc=128 but a 13.6-CPU cgroup quota → torch oversubscribed 128 threads.
  Added a cgroup-aware torch.set_num_threads() cap to the wrapper __init__. Re-ran the REAL local_decode both
  passes on Dev_streaming: **TTFT p50 = 640ms (p90 2033, p95 2413); TTLT p50 = 74ms (p90 137, p95 249)**;
  CER unchanged 13.18% (deterministic). Cap alone cut TTFT 1711→640ms (2.7x). Validity: a thread-controlled probe
  confirmed per-chunk compute stays ~26ms (0% >100ms) even under the host's load-3771, so real-time pacing held
  (cgroup quota is honored regardless of neighbor load) — this 640ms is a clean number, not contention-inflated.
- **Metrics**: ArmA CER_stream=13.18% TTFT_p50=640ms TTLT_p50=74ms ; AdaLoRA CER_stream=17.51% (same latency,
  identical arch/merged weights) ; base zero-shot proxy=31.8% ; per-chunk compute≈26ms.
- **vs comparators (Arm A PARETO-DOMINATES cs8)**: ArmA 13.18%/640ms beats zipformer cs8 13.85%/926ms on BOTH
  CER and TTFT. beam-8 11.62%/1157ms wins CER by 1.6pt but at ~1.8x the TTFT → ArmA opens a non-dominated
  low-latency corner. TTLT 74ms is excellent (fast finalization after audio end).
- **Dev_diag-425 SEVERE (Arm A, 2026-07-26)**: CER **29.96%** / WER 37.81%, **empty 48/425 (11.3%)**.
  Per-etiology CER: ALS 41.26, Down 40.53, CP 29.47, Stroke 26.62, Parkinson 6.41. → realizes plan risk#2
  (severe-tail collapse + confident empties). WORSE than our int8 zipformer's severe ~24.85%, and Test1 is
  severe-heavy → this is the Test-predictive axis. The frozen joint (Arm A) caps the empty rate, consistent
  with the 48 empties.
- **Conclusion (MIXED, not a clean ship)**: Arm A opens a real low-latency corner on general Dev
  (13.18% @ TTFT 640ms, Pareto-dominates cs8) BUT loses the severe tail (29.96%, 11.3% empties) to the
  zipformer (~24.85%) — and Test1 is severe-heavy, so Dev_streaming's 13.18% likely does NOT project to Test.
  AdaLoRA strictly dominated → shelved. NOT a submission.
- **Next / gaps (ranked)**:
  1. **Arm B — low-LR joint/pred-net unfreeze** (nemo_finetune_v2.py --freeze joint_unfreeze): the DESIGNED
     empty-floor lever, directly attacks the 48 severe empties. The clear next GPU experiment for this line.
  2. **Fix wrapper O(N²) feature recompute** — accept_chunk re-extracts features over the whole raw buffer each
     call. Invisible in the real-time streaming pass but makes the untimed ACCURACY pass slow (Dev_diag-425 took
     ~17min) → a real 15000s/submission time-budget risk on full Test. Cache features incrementally. Do BEFORE any submit.
  3. Optional CER squeeze: beam search (zipformer greedy→beam ~-3pt), int8 encoder.
  Checkpoints at /workspace/parakeet_ft/{armA_full,adalora_full}/; wrapper now has cgroup torch-thread cap.

## exp_parakeet_ft_empties — root-cause of the 48 severe empties (NO GPU, CPU diagnostics)
- **Status**: done (2026-07-26, pod 1ppb7l0i5xuna8, since STOPPED). Motivation: Q asked *why* Arm B (joint-unfreeze)
  would fix the empties before spending ~2.5h GPU. Battery = 6 committed scripts (git 41f6431/90e9077/1da125d):
  parakeet_empty_probe, parakeet_offline_recovery, gain_norm_recovery, speaker_audio_probe, compare_empties_vs_model,
  run_parakeet_error_analysis.sh. Artifacts: experiments/exp_parakeet_ft_empties/*.json; Arm A .nemo exported off-pod
  to ~/Downloads/SAPC-artifacts/parakeet_armA/ (md5 f2cc2cc7be1c5e355f6ef3e536137b86).
- **Reproduced**: Dev_diag 425 utt, 48 empty (11.29%), CER 29.93% (≈logged 29.96%). Empties cost ~8.9 CER pts
  (29.93→21.00 if empties were merely average). Empties concentrated: ALS 27/110 (24.5%), CP 16/152 (10.5%), Down 5/67.
- **Theory tests (5 competing hypotheses)**:
  - **T3 streaming/chunking gap — FALSIFIED**: offline single-forward (same [70,1]) recovers only 10/48 (20.8%).
    The model blanks even with full-utterance context → not the H7 streaming path.
  - **T5 EOU misfire — FALSIFIED**: 0 eou-only.
  - **T1 blank-propensity — SUPPORTED**: non-empty errors are deletion-heavy, del:sub = 2.22 (del 9.83% / sub 4.43%).
  - **T2/C energy — PARTIAL**: the empty-dominated speaker `55c1784a` (22/28 empty, ~½ of all empties) sits ~10dB
    below the corpus (RMS −35 vs −25 dBFS), no clipping/DC; BUT within that speaker energy does NOT separate empty
    vs non-empty. This model is `normalize='NA'` → absolute level reaches the encoder → quiet speaker is OOD → blank.
  - **T4 data-bound — REJECTED for the bulk** (see cross-model below).
- **Free gain-norm test**: RMS-normalize empties to −25 dBFS (peak-safe) + offline transcribe → 20.8%→29.2%
  recovered (+4 utts, the very-quiet ones needing +8..+25dB). The +39/+42dB "recoveries" are likely noise
  hallucination. Verdict: partial free win, NOT a clean fix; 34/48 stay empty even at correct level.
- **DECISIVE cross-model check** (`compare_empties_vs_model.py` vs fine-tuned zipformer-beam4 diag decode): on the
  SAME 48 utts (each = 100% CER for parakeet), **zipformer is non-empty 28/48 (58%), CER<0.5 on 15/48 (31%), 9
  PERFECT (alexa/dog/period/football/hey siri…), mean CER 66%**. → the empties are **PARAKEET-SPECIFIC**
  (blank-propensity + normalize='NA' level sensitivity), **NOT data-bound**: a normalizing, different-blank model
  recovers a third cleanly. This IS the mechanism of parakeet's severe-tail loss (29.96% vs zipformer ~24.85%).
- **CORRECTION to prior turn**: I had called the empties "data-bound, Arm B won't help." The zipformer cross-check
  falsified that — they're model-specific and recoverable in principle, which RE-GROUNDS the Arm B (blank-propensity)
  rationale in the empties (not just the diffuse deletion-heaviness). Updated per the surprise.
- **STRATEGIC conclusion**: parakeet's severe-tail loss is a self-inflicted, parakeet-specific pathology the banked
  zipformer never had. Parakeet's only edge is latency (640 vs 926ms). To beat zipformer on the severe/Test-heavy
  tail, parakeet needs BOTH input-normalization AND a joint-unfreeze (Arm B) that survives Dev→Test — two bets vs a
  model already ahead for free. **EV favors banking the zipformer for the Test-predictive tail; parakeet = a
  latency-only play** unless Q judges the low-latency corner worth the two-fix gamble.
- **Next (no GPU committed)**: (a) if pursuing parakeet — add a CAPPED input RMS-norm (+15-20dB cap + energy floor
  to avoid noise hallucination) to the wrapper and validate through the REAL streaming harness (offline≠streaming
  here), THEN decide Arm B on Dev→Test; (b) else bank the gated beam-8/beam-4 zipformer and close the parakeet line.

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

## exp_d0_synth_forensics — D0 synthetic-corpus forensics (+ unplanned leakage audit)
- **Status**: done (2026-07-28/29, pod 38.80.152.148:30836, CPU-only, ran alongside D1 on the GPU)
- **Artifacts**: `experiments/exp_d0_synth_forensics/` (`NOTES.md` = full writeup, `d0_forensics.json`,
  `d0_leak_check.json`). New scripts: `scripts/build_knnvc_transcripts.py`, `scripts/d0_leak_check.py`.
- **Hypothesis**: the two synthetic corpora contain parakeet's failure region (short, quiet-onset utts)
  and carry new information → D2/D3 are worth GPU.
- **Pre-registered gates**: G-COVER (≥5% ≤3-word AND onset p25 ≤ −45 dBFS) · G-PROV (<80% SAP text
  match) · G-EOS (|F5 − kNN-VC trailing silence| < 300 ms).
- **Result**:
  - **G-COVER[kNN-VC] PASS** — 21.1% ≤3 words, onset p25 −56.3 dBFS.
  - **G-PROV[kNN-VC] FAIL** — 100.0% exact SAP-Train text match. kNN-VC is voice conversion of real SAP
    audio, so this is true by construction; it carries new ACOUSTICS, zero new lexical information.
  - **G-EOS PASS** — F5 320 ms vs kNN-VC 340 ms trailing silence (gap 20 ms). Trailing-silence mismatch
    is NOT the mechanical explanation for the D7 EOS collapse.
  - **G-COVER[F5] UNMEASURED** (the log prints FAIL — that verdict is a reporting artifact, do not cite
    it). F5 transcripts are not on the pod at all (the "candidate transcript files" discover found are
    tar-packaging manifests). F5 onsets are exact digital zero (verified 10/10 files, `nonzero=0/2400`),
    so `-inf` was filtered out of the percentile and counted as 100% ≤ −45 dBFS.
- **UNPLANNED, BINDING FINDING — both synthetic corpora contain Dev-derived material**: kNN-VC 18,797 wavs
  from 41 Dev speakers (9.2%); F5 11,931 wavs from 88 Dev-speaker buckets. Train/Dev speakers are disjoint
  (overlap 0), so provenance is decidable per file. **Any D2/D3 run must filter to Train-provenance first**,
  or the Dev gate — our only ship authority — is contaminated. A further 40,339 kNN-VC + 23,373 F5 wavs match
  neither split ("unknown"); treat as excluded until identified.
- **Conclusion**: D2 (kNN-VC) survives as **acoustic augmentation only** — cap ≤25% of steps, finish on real,
  expect modest gains on the 11.3-pt empty lever. D3 (F5) is neither promoted nor demoted: unproven, and its
  slot→text table must be found off-pod before it can be scoped.
- **Next**: does D1 (the joint unfreeze) move the empties at all? Everything data-side is downstream of that.

## exp_armB_parakeet — D1 Arm B, joint unfreeze (RUNNING)
- **Status**: running (started 2026-07-28 21:26Z; rung l0 = 4 epochs × ~45 min)
- **Base**: `/workspace/sweep/parakeet120.nemo` · train `/workspace/nemo_ft/train.json` (331,112, SAP Train)
  · val `/workspace/nemo_ft/val.json` (2,000)
- **Two model corrections found before/while launching** (both verified, both change how results read):
  1. **The base checkpoint already ships `fastemit_lambda=0.03`** — verified on the instantiated loss
     (`RNNTLossNumba.fastemit_lambda=0.03`), not just a config dump. So the runbook's "λ=0 control" does not
     exist, and the pre-registered ladder {0.003, 0.005, 0.01} would *lower* emission pressure below the base,
     i.e. the opposite of the intent. Rung `l0` inherits 0.03 and is still the correct control for the
     UNFREEZE (Arm A trained under the same 0.03). The follow-on rung goes **up** (λ=0.06), justified by the
     measured error profile below.
  2. **`val.json` is drawn from the organizers' Dev split, not a Train-derived speaker-disjoint slice** as the
     runbook states (paths under `processed/Dev/`; 2000/2000 ids join to `Dev.csv`, 0 to `Train.csv`). No
     train/val leakage (train is all `processed/Train/`), but **checkpoint selection has been seeing Dev** —
     for Arm A too. Caveat on the independence of any Dev-based ship gate.
- **Arm A baseline re-measured on the identical val set** (`scripts/val_metrics.py`, same pinned [70,1] ctx):
  CER **24.93%** · empties **324/2000 (16.2%)** · insertions **1.38%** · **deletions 24.76%** · subs 5.87%.
  Deletions outnumber substitutions **4:1** — independent corroboration of the confident-blank theory D1
  targets, and the reason the FastEmit direction is UP.
  Note the empty rate on this broader Dev-drawn set (**16.2%**) exceeds the Dev_diag severe slice (11.3%),
  which answers Stage-0/GATE-REP's "does the pathology generalize" on the PROCEED side.
- **D6 (short-command oversampling) — premise dented before spending a rung on it**: ≤3-word utterances are
  already **22.5% of train (74,606 utts)**, not rare. The internal-LM-bias-on-rare-short-phrases rationale does
  not fit this data; ×3 would push train to ~45% short. Manifest is built (`/workspace/nemo_ft/train_d6.json`,
  `scripts/d6_oversample.py`) but the rung is deprioritized pending a reason to believe it.
- **Gates**: GATE-TRAIN (proxy) = val CER < 24.93% AND empties < 324 AND insertions ≤ 1.59%.
  GATE-SHIP (only ship authority) = official two-ref sclite Dev_diag CER ≤ 24% AND mean(TTFT,TTLT) ≤ 420 ms.
- **Epoch 0**: val CER 0.2978 vs Arm A's epoch-0 0.2974 — no collapse (tripwire not tripped), no gain yet.
- **Rung l0 trained out** (4 epochs, 2026-07-29 00:39Z). Per-epoch val CER 0.2978 → 0.2806 → 0.2692 → 0.2640,
  monotone, no collapse. Top-5 averaging had only 4 checkpoints so it averaged **all** of them, epoch 0 included.

### RESULT — the joint unfreeze did NOTHING, and the baseline it was measured against was wrong

**GATE-TRAIN (proxy val, 2000 utts)** — passed on paper, failed on substance:

| | Arm A | Arm B l0 | delta |
|---|---|---|---|
| CER % | 24.9343 | 24.8119 | −0.12 |
| empties | 324 | 323 | **−1 of 2000** |
| insertions % | 1.3828 | 1.4692 | +0.09 |
| deletions % | 24.7605 | 23.8747 | −0.89 |

The gate printed `PASS= True`, because I wrote "empty count drops" with **no magnitude** — one utterance
satisfied it. D1's claim was that unfreezing the joint *collapses* the empty rate. It moved 0.3% relative.
Deletion:insertion imbalance stayed ~16:1 (Arm A 18:1). **A gate that noise can pass is not a gate.**

**GATE-SHIP (official two-ref sclite, Dev_diag severe n=425)** — and here the real error surfaced:

| run | CER | WER | empties |
|---|---|---|---|
| Arm A, hypotheses from 2026-07-26, **rescored today** | **18.69%** | 24.97% | 48 |
| Arm A + `input_gain` ON | 19.22% | 25.73% | 50 |
| **Arm B l0** | **18.74%** | 24.93% | 50 |

l0 is **0.05 pts worse than Arm A** — identical within noise, exactly as the proxy said. Latency
(l0, Dev_streaming): TTFT p50 **0.630 s**, TTLT p50 **0.072 s** → mean **351 ms**.

**The 29.93%/29.96% severe baseline in `PLANNED.md` and line 186 above does not reproduce.** Same Arm A
hypothesis CSV, official `evaluate.sh`, today → **18.69%**. The 29.9x figure came from the error-analysis
proxy scripts (single-ref, no min-over-two-refs, no `unk` reconciliation), never from `evaluate.sh`. The
~11-point gap is what two-ref min scoring buys.

**What this invalidates.** Every quantity derived from 29.93%: "empties cost 11.29 CER pts", "CER if empties
were merely averaged = 21.00%", "already beats zipformer's 24.85%", and the framing of the whole D-series as
an empty-tail rescue. The empty *count* (48/425) is real and was measured off the CSV; the CER *arithmetic*
built on it is not. **`24.85%` for zipformer severe has the same provenance risk and is being re-measured
through the official scorer before any comparison is believed.**

**Method failure, same family as the 2026-06-24 Nemotron post-mortem.** A proxy number was carried forward
as if it were the official one and became the reference every later decision was scored against. It is in
the house rules that proxies never authorize a ship claim — the unwritten half is that **proxies must never
become baselines either.** Caught only because l0's "11-point win" was too large to believe and the control
was one rescore of an existing CSV away.

### Dev_clean2k — a comparison slice neither model selected against

`val.json` spans only **7 speakers** across its 2000 utts, and `Val2k` is those same 2000 utts, so Val2k
is 100% contaminated for every parakeet arm (checkpoint selection ran on it) and is a narrow basis besides.
Dev_diag overlaps val.json by only 33/425 (7.8%), which is why the severe comparison above is trustworthy.

Built `Dev_clean2k.csv` on the pod: **2000 utts, 122 speakers** (≤17 per speaker), disjoint from *both*
`val.json` and `Dev_diag`, spanning all five etiologies (PD 570 · CP 471 · ALS 455 · Down 319 · Stroke 185).

| model | CER | WER | empties |
|---|---|---|---|
| **parakeet Arm A** | **13.51%** | 18.78% | 62/2000 (3.1%) |
| zipformer beam-4 | 18.19% | 24.27% | 20/2000 (1.0%) |

**Parakeet Arm A wins the broad clean slice by 4.68 CER pts**, on top of winning severe by 3.79 —
while carrying 3× the empty rate. So the empty tail was never the dominant CER term; the D-series was
aimed at a lever that the official scorer says is worth far less than the proxy claimed. D1 failing to
move the empties and parakeet winning *despite* them are the same finding from two directions.

**Val2k is retired as a benchmark.** Same zipformer, same official scorer, both 2000 Dev utts:
Val2k **29.78%** vs Dev_clean2k **18.19%**. An 11.6-pt swing from the speaker draw alone. Any conclusion
that leaned on a Val2k number is measuring those 7 speakers, not the Dev distribution.

- **Next**: re-measure zipformer beam-4 on Dev_diag severe officially (done: 22.48%); fe06 killed to free
  CPU — its CER premise is falsified by l0 and its remaining rationale is **latency** (TTFT p50 630 ms),
  which is worth a rung later but not ahead of establishing which model is actually ahead.

## exp_parakeet_onnx_ship — Arm A exported to ONNX and gated for Codabench
- **Status**: gates green except Dev_clean2k (running) — 2026-07-29, pod 1ppb7l0i5xuna8.
  Q chose Option 2 (ONNX export) over bundling the NeMo wheel tree: every submission that ever scored
  for us ran ONNX under onnxruntime, no NeMo submission ever has.
- **Checkpoint identity trap (caught before export)**: `track2_starting_kit/parakeet_realtime_ft/weights/
  parakeet_realtime.nemo` on the pod is **Arm B l0** (md5 cdf2dd9f), not Arm A (md5 f2cc2cc7 =
  `parakeet_ft/armA_full/ft_smoke_encoder_only.nemo`). Exporting from the submission dir would have
  shipped the joint-unfreeze arm, and parity would have passed anyway (both sides wrong; the two arms
  differ by 0.05 CER pts). Arm A is now linked into `/workspace/parakeet_ref_armA/`.
- **Four bugs found by the gates, all of the silently-wrong class**:
  1. **Cache layout.** NeMo's runtime `get_initial_cache_state` is LAYER-first `[17,1,70,512]`; the
     exported graph declares BATCH-first `[1,17,70,512]`. The exporter now aligns runtime tensors to the
     graph's static dims (unique permutation, batch at dim 0, ambiguity raises) and `model.py` re-checks
     the meta against the session at construction.
  2. **Step 0.** The graph applies `drop_extra_pre_encoded` ITSELF and unconditionally, so NeMo's bare
     9-frame first step underflows to a zero-length sequence (`Conv ... Invalid input shape: {0}`).
     Step 0 now carries a full-width pre-encode cache of zeros (the Nemotron convention).
  3. **Mel pad mode.** NeMo's `FilterbankFeatures.stft` pads with `constant`, not `reflect` (our localmel
     inherited reflect from the Nemotron one) — worth ~3.5 log-mel units on the first frames.
  4. **Mel tail mask.** NeMo masks every frame at/after `get_seq_len(n) = floor(n/hop)` to `pad_value`;
     with `center=True` that is always the final column, so an unmasked front-end differs by
     |log(log_zero_guard)| = 16.6 on the last frame of EVERY utterance. Cache stays unmasked; the mask is
     applied on the way out so it cannot freeze a zeroed column mid-utterance.
- **Encoder trim policy — MEASURED, not reasoned** (`scripts/probe_parakeet_enc.py`): graph does its own
  `drop_extra` and `valid_out_len`. `drop=nemo` gives 983/1013 frame mismatches; `drop=none` gives 0 with
  max|Δ| **5.17e-05**. Shipped config: `drop_policy=none, trim_policy=none`.
- **Parity vs the NeMo Arm A wrapper** (30 utts, 6 per etiology from Dev_diag):
  feat max|Δ| 8.9e-04 ✅ · enc 0 frame mismatches / 5.2e-05 ✅ · **text only 19/30 exact (63%)**.
  The text gate was a proxy of my own invention and it is the WRONG metric: greedy RNN-T is chaotic
  under any numeric difference (one flipped argmax diverges the prediction-net state for the rest of the
  utterance), and the disagreements run in both directions. Superseded by the official scorer below;
  `package_parakeet_onnx.py` now treats feat/enc as hard gates and text as advisory, with the reason and
  these numbers written into the code.
- **Official scorer (`evaluate.sh`, min-over-two-refs), Dev_diag severe n=425:**
  | build | CER | WER |
  |---|---|---|
  | banked Arm A (NeMo) | 18.69 | 24.97 |
  | ONNX fp32 | **18.73** | 25.05 |
  | ONNX int8 (ship candidate) | **18.93** | 25.17 |
  int8 costs 0.20 CER pts for 537 MB -> 264 MB. Still 3.55 pts better than zipformer beam-4 (22.48).
- **Latency (int8, threads=1, Dev_streaming, real-time paced): TTFT p50 628 ms / TTLT p50 70 ms ->
  mean 349 ms** (banked NeMo Arm A 356 ms; board best 592 ms; our shipped zipformer ~730 ms). The
  pre-registered rule picks the lowest thread count meeting <=420 ms, so threads=1 ships — which is also
  what the 15000 s wall-clock budget wants. No 2/4 sweep needed.
- **Offline validity (the recurring cause of submission death)**: `pip --no-index` into an empty target
  under DEAD PROXIES installs from the bundled wheels ✅; the extracted zip then decodes 17/20 non-empty
  with dead proxies + `HF_HUB_OFFLINE=1` ✅. numpy scare resolved empirically: ort 1.28 needs
  `numpy>=1.21.6`, satisfied by the runtime's 1.26.3, so the bundled numpy 2.4.6 wheel is never installed
  over it (`deps ready: ort 1.28.0 | numpy 1.26.3 | torch 2.4.1+cu124`). It only appeared when installing
  into an EMPTY dir.
- **Dev_clean2k int8 (n=2000, 122 speakers): CER 13.24% / WER 18.63%** — passes the <=15% gate and edges
  the banked NeMo Arm A (13.51%). vs zipformer beam-4 18.19%.
- **Wall-clock budget**: 200 utts (1830 s audio) single-process at the shipping `SAPC2_THREADS=1` took
  626 s wall (both passes + model load) -> RTF 0.342/worker. Test1 projection with stated assumptions
  (10521 utts x Dev's 8.01 s mean = 84,300 s audio; 20 worker processes; x1.677 pod->Codabench per-core
  correction measured in the Nemotron timing gate) = **~2,420 s vs the 15,000 s budget, ~84% margin**
  (gate: >=30%). Order-checks against Nemotron's 8,432 s projection for a 5x larger model.
- **Artifact (final)**: `parakeet_armA_int8.zip`, **200.6 MB**, sha256
  `82bc071adbd7e309cdd48bb22e147c41e8b009ca225933c0a3dc95a40267eca7`, 18 entries, model.py at root.
  Local copy + all gate JSON/logs: `/Users/o/Downloads/sapc2_parakeet_onnx/`.
  (The first zip, sha `851d326c...`/246 MB, carried a duplicated `wheels/wheels/` tree from a stray
  `cp -r`; removed and repackaged, then the offline `--no-index` install was revalidated from the
  repacked zip.)
- **ALL FIVE pre-registered ship criteria met.** Pod `1ppb7l0i5xuna8` stopped 2026-07-29.
- **NOT SUBMITTED** — uploading to Codabench is Q's call.
- **Cost note**: ~5.5 h of pod time, of which ~80 min was idle because an `exit 1` guard in my own chain
  script killed the remaining stages unnoticed. Chains now mark every stage with an rc instead of aborting.

---

## exp_parakeet_onnx_int8_POSTMORTEM — Test1 CER 100%, and the ISA gap that caused it
**Date**: 2026-07-29 (submitted, scored, root-caused the same day) · **Pod**: `1ppb7l0i5xuna8`

`parakeet_armA_int8.zip` (sha `82bc071a…`) was uploaded despite ALL FIVE pre-registered ship
criteria being green. Codabench result:

- **Test1: WER 1.0000 / CER 1.0000 (100.00%)**, all 10521 utts matched, nothing salvageable.
- Latency: `TTFT p50=6170.80, p90=18960.03 | TTLT p50=82.13, p90=127.09 (total_n=89)`.
- Test2 died in the scorer: `preds from sgml-ref1 and sgml-ref2 are not identical!
  len(preds_ref1) = 8304, len(preds_ref2) = 8303`.

### It was NOT a timeout and NOT slow — read the latency code before believing the latency
TTLT p50 82 ms proves the model kept up with real time. The TTFT p50 of 6170 ms is a
**degenerate artifact**: `utils/compute_latency.py:91-99`
(`_first_non_empty_or_last_event_time`) falls back to the LAST event timestamp when every
partial for an utterance is empty. So a huge TTFT here means *silence*, not latency. `total_n=89`
likewise is just the streaming pass's utterance count, not a failure count. I initially
misread 6170 ms as a 17x speed collapse; that was wrong and cost an investigation branch.

Also: **CER exactly 1.0000 does not by itself prove empty output** — `compute_metrics.py` clips
per-utterance error at 1.0, so insertion-heavy gibberish scores identically. The TTFT fallback
is what actually localized it to empty partials.

### What we eliminated by direct test (shipped zip + real `local_decode.py`, 10 real Dev utts)
| runtime | CPU path | ort | result |
|---|---|---|---|
| Mac | arm64 NEON | 1.24.4 | 9/10 non-empty ✅ |
| Mac | arm64 NEON | 1.28.0 | 9/10 ✅ |
| Docker `linux/amd64` (Rosetta) | **SSE4.2 only** | 1.28.0 + numpy 2.1.2 + torch 2.5.0 | 9/10 ✅ |
| pod gates (18.93% / 13.24%) | Xeon 8462Y+, **avx512_vnni** | 1.27.0 | ✅ |
| pod offline smoke, 20 utts | same | 1.28.0 | 17/20 ✅ |
| nemotron zip that scored 27.97% | **EPYC Milan, avx2 no vnni** | 1.27.0 | ✅ |
| **Codabench parakeet** | **EPYC Milan, avx2 no vnni** | **1.28.0** | ❌ **100%** |

Exonerated: zip contents, weights, `config.yaml`, wrapper logic, numpy 2.x, torch 2.5+,
ort 1.28.0 per se, speed, and the 20-worker topology as a *sufficient* cause.

### ROOT CAUSE: we never once ran the kernel the worker runs
Confirmed on-pod this session: `grep flags /proc/cpuinfo` gives
**Intel Xeon Platinum 8462Y+ (Sapphire Rapids) with `avx512f`, `avx_vnni`, `avx512_vnni`**.
The Codabench worker is **EPYC Milan (Zen 3): avx2, NO VNNI, NO avx512** (see memory
`eval-worker-cpu-confirmed`).

ORT dynamic quantization is **U8S8** (`DynamicQuantizeLinear` uint8 activations x int8 weights).
Without VNNI, `MatMulInteger` lowers to MLAS kernels built on **`VPMADDUBSW`, which accumulates
two u8*s8 products into a SATURATING int16** — worst case `255*127*2 = 64770 >> 32767`. With
VNNI (`VPDPBUSD`) the accumulator is int32 and cannot saturate. We shipped
`per_channel=True, reduce_range=False`; `reduce_range=True` would cap weights at 7 bits
(`255*63*2 = 32130 < 32767`) and make the AVX2 path safe.

So the gates were run on silicon on which the bug is *unreachable by construction*. Rosetta was
not a substitute either — it advertises SSE4.2 only (`AVX2: False`), a third distinct path.

**A wrong turn worth recording**: nemotron shipped the *identical* U8S8 recipe to the *same*
worker and scored 27.97%, and I treated that as falsifying the mechanism. It does not.
Saturation depends on the model's own activation magnitudes, so it is model-specific — another
model surviving proves nothing about yours.

### Two validation holes, both real independent of root cause
1. `scripts/run_parakeet_onnx_pod.sh:51` ran **unpinned** `pip download onnxruntime`. It resolved
   to **1.27.0** in June (nemotron, scored) and **1.28.0** on 2026-07-29 (parakeet, 100%).
2. The shipped `setup.sh` guard `if ! python3 -c "import onnxruntime"` skips the bundled wheel in
   any env that already has ort — i.e. **every accuracy and latency gate**, all of which ran
   inside `$VENV` (ort 1.27.0). Only the 20-utt `offline` stage ever used the shipped 1.28.0.
   Net effect: **the version we gated was never the version we shipped**, and nothing compared them.

Fixed in `run_parakeet_onnx_pod.sh`: `ORT_VERSION` pinned (default 1.27.0); `gate_lat`/`gate_acc`
now decode with `$WORK/offlinevenv/bin/python` and **refuse to run** unless its ort equals the
pinned bundle version; ISA check added as pre-registered ship criterion #6. The shipped
`setup.sh` was deliberately left alone — on the real worker ort is absent, so the guard passes
through and the bundled wheel installs correctly; the defect was in where we gated, not in it.

### Still unexplained (do not assume the fix covers it)
Test2's `len(preds_ref1)=8304 vs 8303` scorer abort. Test1 scored normally with the same empty
output, so this may be a latent defect rather than a symptom. We have no Test2 data locally and
cannot reproduce it. **If a future submission errors this way again with real transcripts, it is
a separate bug.**

---

## exp_parakeet_onnx_fp32 — the fix: fp32 encoder + pinned ort 1.27.0
**Date**: 2026-07-29/30 · **Pod**: `1ppb7l0i5xuna8` (started 21:30 UTC, stopped 23:16 UTC, ~1.8 h)

### What changed, and why it is not just "another green gate"
Two changes, both aimed at the root cause rather than at the symptom:
1. **fp32 encoder** replaces the int8 one. `build/fp32` is **md5-identical** to the shipped
   `build/int8` in `model.py`, `localmel.py`, `setup.sh`, `requirements.txt`, `config.yaml` and
   every weight file except the encoder — a pure encoder swap, no wrapper change.
2. **`onnxruntime==1.27.0` pinned** in the bundle (was unpinned, resolved to 1.28.0).

The argument for fp32 is **not** that a gate went green — that is exactly the belief that failed
last time. It is that **the failing instruction is no longer issued**: with no quantized MatMul
there is no `MatMulInteger`, no `VPMADDUBSW`, and no int16 saturation, on any ISA. The failure
mode is deleted from the artifact rather than re-measured.

### Gate results (all from the EXTRACTED zip, via the organizers' `local_decode.py` + `evaluate.sh`)
| criterion | gate | fp32 | int8 (shipped, failed) | verdict |
|---|---|---|---|---|
| parity feat/enc vs NeMo | pass | banked green | same | ✅ |
| Dev_diag severe CER | ≤20% | **18.733%** (WER 25.054%) | 18.93% | ✅ |
| Dev_clean2k CER | ≤15% | **not measured** — killed mid-run | 13.24% | ⚠️ see below |
| mean(TTFT,TTLT) p50 | ≤420 ms | **375.7 ms** (TTFT 665.3 / TTLT 86.1) | 349 ms | ✅ |
| Test wall-clock margin | ≥30% | **~69%** (~4,620 s of 15,000) | ~84% | ✅ |
| ISA exposure (new #6) | none | no int8 kernels at all | U8S8, non-VNNI unsafe | ✅ |

- **Dev_diag from the zip was byte-identical to the pre-zip decode** (18.733% / 25.054% both), so
  the packaged artifact and the build dir compute the same thing.
- **Latency is clean, not degenerate**: `n_utts_total=123, n_utts_with_timing=123,
  n_utts_with_mfa_start=123`. Contrast the failed submission's `total_n=89` with the all-empty
  TTFT fallback.
- **Throughput**: 200 utts / 1830 s audio, threads=1, **1195 s** wall vs int8's 626 s = **1.91x
  slower**, RTF 0.653/process. Projection reuses the int8 method (20 workers, x1.677 pod->Codabench
  per-core correction): 2,420 x 1.91 = **~4,620 s of the 15,000 s budget**.
- **Offline validity, this time proven properly**: fresh venv with `--system-site-packages` from the
  *base* interpreter, asserted to have **no** onnxruntime, then `setup.sh` under dead proxies +
  `HF_HUB_OFFLINE=1` installed
  `onnxruntime-1.27.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` — the same wheel
  filename the nemotron submission that scored 27.97% carried. `deps ready: ort 1.27.0`. 17/20
  non-empty on the smoke, matching int8's 17/20.
- **Third-ISA check, off-pod**: the downloaded zip decodes **9/10 non-empty on Mac arm64** (ort
  1.24.4 / numpy 2.2.4 / torch 2.6.0) with clean transcripts. So the artifact behaves the same on
  NEON, SSE4.2-only x86 (Rosetta), and VNNI x86.

### Artifact
`parakeet_armA_fp32.zip`, **480,939,149 B**, sha256
`b17ad0d8f6435d2c4fda2d7ea61bad708c94876bf9b68226852c9b0c9bb62f29`, 18 entries, `model.py` at root.
Local: `/Users/o/Downloads/sapc2_parakeet_fp32/` (+ `gates/` with every metric JSON and log).
**NOT SUBMITTED** — Q's call.

### The one gap, stated plainly
Dev_clean2k was killed ~5 min in when Q chose to stop the pod, so **criterion 3 is argued, not
measured**. The argument: int8 is a lossy approximation *of this exact fp32 graph*, and int8 scored
13.24%; for fp32 to miss ≤15%, quantization would have to have *improved* the model by >1.8 CER
points on that slice, while we separately measured fp32 beating int8 on the severe slice. Strong,
but it is a dominance argument, not a number. If Dev_clean2k matters for the ship decision it costs
~3 h of pod time to close.

### Process fixes landed in `scripts/run_parakeet_onnx_pod.sh`
- `ORT_VERSION` pinned (default 1.27.0) with the drift history written into the comment.
- `gate_lat` / `gate_acc` now decode with `$WORK/offlinevenv/bin/python` and **refuse to run**
  unless that interpreter's ort equals the pinned bundle version. Gating with `$VENV` is how a
  submission passed five gates on a runtime it did not ship.
- ISA match added as pre-registered ship criterion #6, with the VPMADDUBSW arithmetic spelled out.
- Shipped `setup.sh` deliberately **unchanged**: on the real worker ort is absent, so its guard
  passes through and the bundled wheel installs correctly. The defect was where we gated.

### Cost
~1.8 h pod (~$5.4). Both trained checkpoints exported and md5-verified off-pod during the run at no
extra cost: Arm A `f2cc2cc7be1c5e355f6ef3e536137b86`, Arm B l0 `cdf2dd9f0de63b510b6962dcd55b09e4`,
460,062,720 B each, in `/Users/o/Downloads/sapc2_checkpoints/`.

---

## exp_parakeet_onnx_fp32_test1 — SUBMITTED, and it lands ON the Pareto frontier
**Date**: 2026-07-30 · **Commit**: 82032b1 · **Artifact**: `parakeet_armA_fp32.zip` (sha256 `b17ad0d8…`)

### Result (Codabench Test1, 10521/10521 matched)
`WER 25.5019% · CER 19.0100% · TTFT p50 742.70 ms (p90 2514.74) · TTLT p50 91.06 ms (p90 154.19)`
`latency_n = mfa_n = 89` · scoring wall 54 s. **Latency figure of merit = mean(TTFT,TTLT) = 416.88 ms.**

### Standing vs the 2026-07-26 board
| # | team | CER | WER | TTFT p50 | TTLT p50 | mean lat |
|---|---|---|---|---|---|---|
| — | yac3xn | 18.10 | 25.12 | 1126.30 | 58.28 | 592.3 |
| — | **us (parakeet fp32)** | **19.01** | **25.50** | 742.70 | 91.06 | **416.9** |
| — | takagi | 19.52 | 24.94 | 1439.71 | 141.70 | 790.7 |
| — | takagi | 20.69 | 26.59 | 1226.86 | 135.91 | 681.4 |
| — | us (old beam-4 zipformer) | 21.28 | 29.50 | 1365.55 | 93.41 | 729.5 |

- **Non-dominated.** yac3xn beats us on CER by 0.91; we beat it on latency by 175 ms. Frontier goes
  from one point to two, and we own the low-latency corner — the exact route
  `test1-standing-and-pareto` predicted.
- We now **strictly dominate** both takagi entries and our own shipped beam-4 (better on *both* axes:
  −2.27 CER, −313 ms). The zipformer submission is retired as a frontier candidate.

### What this confirms
1. **The ISA gap was the whole int8 catastrophe.** Same weights, same wrapper, same ort 1.27.0; only
   the encoder quantization changed, and CER went 100.00% → 19.01% on the same non-VNNI EPYC Milan
   worker. Closes `parakeet-not-packaged-nemo-blocker`; keeps `gate-on-the-worker-isa` standing.
2. **Dev→Test transfer is tight**: pod gate Dev_diag severe 18.733% → Test1 19.01% (+0.28 pt), with
   Dev_clean2k at 13.24% (int8). Test1 sits at the severe end, as expected.
3. Predicted latency held: pod gate mean 375.7 ms → measured 416.9 ms (+41 ms on a slower worker).

### Remaining slack (not blocking)
TTFT p90 2514.74 ms vs p50 742.70 ms — a long tail from empties / late first-partial, unchanged in
kind from the Dev runs. Cutting it would push the latency corner further out, but the frontier
position is already secured; CER (−0.92 to take the CER axis too) is the higher-value target.

---

## exp_s0_s1_probe — the empty tail is 3.89 points, and the decode knob cannot reach it
**Date**: 2026-07-30 · **Pod**: `1ppb7l0i5xuna8` (started 14:41 UTC, stopped 14:52 UTC, **11 min, ~$0.55**)
**Commits**: `6d275f9`, `9e0e1d9`, `dc6ae97` · **Artifacts**: `experiments/exp_s0_s1_probe/`

Two CPU-only measurements, no GPU, no retraining. Both changed the picture.

### S0 — the first error decomposition ever run through the official scorer
`scripts/error_decomposition.py` on the banked hypotheses of the **exact artifact that scored
Test1 CER 19.01%** (`art32/Dev_diag.fp32.predict.csv`). Both self-gates passed: our SGML parse
reproduces `utils/compute_metrics.py` exactly, and our per-utterance sums reproduce the official
metrics — **CER 18.7332% / WER 25.0540%**, identical to the banked gate numbers.

Dev_diag severe, n=425, **28,986 ref chars, 5,430 errors**, 3 utterances hit the 1.0 clip.

| char-level | count | CER points |
|---|---|---|
| **deletions** | 3,780 | **13.04** |
| substitutions | 1,294 | 4.46 |
| insertions | 401 | 1.38 |

**del:sub 2.92:1 · del:ins 9.43:1.** Word-level: D 760 / S 631 / I 87 (del:sub 1.20:1) — the
char ratio being much higher says deletions remove *whole spans*, while substitutions cost a few
characters each.

**Empties: 48 utterances, 1,129 errors = 3.89 CER points (20.8% of the error mass).** The
retracted proxy claimed 11.29. The pre-pod estimate in `investigations/step01_runbook.md`
(≈3.5–4.0, from local char counts) was right.

**Where the mass actually is — by reference length:**

| words | n | CER pts | share of errors | empties |
|---|---|---|---|---|
| **13+** | 99 | **9.55** | **51.0%** | 2 |
| 7–12 | 82 | 4.00 | 21.4% | 9 |
| 4–6 | 136 | 3.92 | 20.9% | 15 |
| 2–3 | 72 | 0.85 | 4.5% | 8 |
| 1 | 36 | 0.41 | 2.2% | 14 |

**Half of all error mass is long utterances that are not empty at all.** The short/wake-word slice
the entire research program was organised around is 6.7% of the errors.

By etiology (CER points): ALS 6.40 (34.2% of errors) · CP 4.94 · **Parkinson 3.79** · Down 2.82 ·
Stroke 0.78. Parkinson's has the *lowest* CER (10.00%) but carries 38% of all reference characters,
so it still contributes more than Down Syndrome, whose CER is the worst at 29.88%.

**One speaker, `55c1784a`, is 14.1% of all errors** (28 utts, 2.64 CER pts, 21 of the 48 empties,
char-deletion rate 85.4%).

### S1 — blank-margin probe on 60 real Dev_diag utterances: pre-registered **NO-GO**
5,311 greedy steps, blank won 92.86%, 11/60 empty. Margins p10 **2.60** · p50 **9.05** · p90 15.05.
Flip fractions: β=1 → 3.1% · β=2 → 7.0% · β=3 → 12.1% · β=4 → 17.2% · β=6 → 27.0%.

The discriminating measurement came back on the **wrong side**:

| | empty utts (11) | non-empty (49) | delta |
|---|---|---|---|
| p50 blank margin | 9.391 | 9.049 | +0.342 |
| **p10 blank margin** | **6.346** | **2.611** | **+3.735** |

Empties are **more** confidently blank than ordinary blanks. Even the *least* confident blank inside
an empty utterance sits at 6.35, so reaching it needs β ≈ 6, which flips **27% of all blank
decisions globally** — an insertion flood. **A constant logit shift cannot separate the two
populations.** Per the pre-registered rule the grid was NOT run and the pod was stopped.

### What this retires
1. **The empty tail is not the CER problem.** 3.89 of 18.73 points, and unreachable by the decode
   knob. Combined with D1 (joint unfreeze moved it 48 → 50) and the `input_gain` patch (0/48), three
   independent attacks have now failed on it. Stop organising the programme around it.
2. **The local gain finding held up.** `investigations/step01_runbook.md` showed −30 dB attenuation
   changes nothing (per-feature normalisation) while SNR degradation reproduces the empties. The
   "quiet onset −58.5 dBFS" statistic is a correlate, not a cause.
3. **The real target is deletion inside long, non-empty output**: 13.04 of 18.73 CER points are
   deletions, and 51% of the error mass is utterances of 13+ words.

### Cost of the pre-registration
Writing the gate first turned a ~4.5 h session into an 11-minute one and stopped a grid whose
rationale had just been falsified. The instrument also caught its own calibration error twice before
the pod (a guessed 0–4 β grid was dead; 0–12 reached too high).
