# STATE_OF_WORK.md — Audit of existing SAPC2 work on RunPod (pod 3dwiczo41jeg1y)

Read-only audit, Session 3 (2026-06-19). Pod: 192 cores / 2 TiB RAM / **H200**. Everything below is
from artifacts on `/workspace`, not rebuilt. Tags: `[V]` verified from a file, `[?]` needs confirmation.

## 1. Repo lineage (resolves the "clone/rename" question)
All three dirs are the **same GitHub repo** `github.com/leolelion/SAPC-track2`, different checkouts: `[V]`
- `/workspace/SAPC-template` — `c48d945` (Mar 19). **Identical to our Mac copy.** Oldest. Template baseline.
- `/workspace/SAPC-track2` — `064cfd4` (Apr 8). Parakeet pipeline.
- `/workspace/sapc-nemotron` — `27bf6d1` (May 16). **Newest & most complete → the live checkout.**
- `/workspace/SAPC2` — not a git repo; holds data (`manifest/Train.csv` = 336k utts, `Dev.csv`).
**Conclusion**: no clone-from-Mac needed (pod is ahead); no rename needed (distinct dirs). Work in `sapc-nemotron`.

## 2. What's been built (the map)
**Track-2 streaming candidates** (`sapc-nemotron/track2_starting_kit/*/model.py`): `streaming_zipformer`
(icefall), `sherpa_zipformer` (finetuned→ONNX), `nemotron_streaming` (ONNX int8), `wenet_u2pp`,
`qwen3_asr`, `parakeet_ctc`, `moonshine`. `[V]`
**Offline / Track-1**: Qwen3-ASR (LoRA + full), ROVER system-combination, GEC, etiology classifier. `[V]`
**Finetune pipeline** (`/workspace/finetune/`, icefall): checkpoints `epoch-0.pt` (=LibriSpeech-30),
`epoch-1.pt`, `epoch-4.pt`, `best-valid-loss.pt`; exports in `finetune/onnx/standard/`. `[V]`
**Process WAS set up**: `sapc-nemotron/CLAUDE.md` mandates an experiment ledger, off-pod archiving,
"never invent metrics", logging failed runs, checking LEADERBOARD before new experiments. `[V]`

## 3. Known results (CER)
| Track | System | CER | Source |
|---|---|---|---|
| Offline | Qwen3-ASR | **6.5%** | rover_summary.json `[V]` |
| Offline | GEC 1-best (full Dev 47.9k) | 6.7% (oracle 4.3%; GEC *hurt*) | gec summary `[V]` |
| Streaming | **Finetuned Zipformer** | ~7.75–8% offline / ~11.5% streaming(dev100) | eval logs `[V]` |
| Streaming baseline | Kroko Zipformer (unadapted) | 26.2% (486 empty) | phase4b `[V]` |
| Streaming baseline | Nemotron (unadapted) | 24.4% (**1463 empty**) | phase4b `[V]` |
Per-etiology everywhere: PD easiest (~12–15%), CP hardest (~42%), then DS/Stroke/ALS. `[V]`

## 4. PROBLEM A — why the Nemotron Codabench upload failed (the frustration)
**Diagnosis: a fragile, network-dependent submission package — not a model-logic bug.** `[V setup.sh]`
The local emulation *passed* (RTF 0.011, CER 9.93%, 763/15000 s) → the failure was on the **platform**.
The `nemotron_streaming` submission:
- zip is **7.2 KB** → **weights are NOT bundled**; `setup.sh` **downloads them from HuggingFace at setup
  time** (`hf_hub_download danielbodart/...`). If the Codabench worker has **no/limited internet**, setup
  fails → ingestion fails. (Classic "works locally, dies on Codabench".)
- `setup.sh` also installs **`nemo_toolkit[asr]>=2.5,<2.6`** — heavy and fragile; install can fail/timeout
  on the worker, and the script hard-exits if `import nemo.collections.asr` fails.
- Builds a venv with `--system-site-packages` and pins torch — extra moving parts that can break.
**Fixes (high-confidence)**:
1. **Bundle the ONNX weights inside the zip** (no network at setup). Submission becomes self-contained.
2. **Drop the NeMo runtime dependency** — at inference the model needs only `onnxruntime` + a mel/feature
   extractor; NeMo is the single biggest install-failure source. Replace its preprocessor with a small
   local fbank (or sherpa-onnx, which bundles features).
3. **Validate the zip in the local `sapc2-runtime` Docker with networking disabled** before uploading.

## 5b. UPDATE (Session 3, verified on pod): PROBLEM B is SOLVED — it was a sherpa-onnx version bug
Ran the **exported** `finetune/onnx/standard/{encoder,decoder,joiner}.onnx` via **sherpa-onnx 1.13.3**
on Dev wavs → **correct transcriptions** ("now is our one moment of glory", "find my phone", etc.). `[V]`
The empty output happened on **sherpa-onnx 1.12.35** (the debug doc's own regression guess was right).
**Fix = pin sherpa-onnx ≥1.13.3.** Weights + export were always fine. Implications:
- The finetuned streaming Zipformer is a **working Track-2 model** → fastest path to a valid submission.
- The phase4b "empties / 26% CER" numbers are **contaminated by the 1.12.35 bug** → re-evaluate.
- Remaining real issues: (a) streaming-boundary truncation (start warmup + tail flush); (b) model only
  ~1–2 epochs finetuned, loss→0.035 = likely text-overfit → re-finetune with a proper recipe.
- Preliminary rough CER on 20 mixed-etiology Dev utts ≈ 26% but NOISY (n=4/etiology, char-metric, not
  min-two-refs) → must run a proper eval for the true number. `[?]`

## 5. PROBLEM B (original notes) — the fine-tuned Zipformer empty-output bug (now resolved, see 5b)
`inference_debug.md` `[V]`: the finetuned Zipformer trains well (loss 7.37→0.035) but ONNX/sherpa
inference emits empty/garbage (argmax pinned to `▁`/blank every frame).
**Root cause found (untested fix)**: the export/inference called `model.encoder(feat)` directly,
**bypassing `model.encoder_embed` (Conv2dSubsampling)** → encoder fed un-subsampled features (865 frames
for 8.67 s instead of ~215). Correct path: `encoder_embed(feat) → encoder(x)`. **Never verified.** `[V]`
Secondary concern: 1-epoch finetune converging to loss 0.035 looks like **text overfit, not acoustic
adaptation** → re-finetune 2–3 epochs, consider lower LR (0.001), compare `best-valid-loss.pt`.
⇒ The "finetuned Zipformer" numbers and the phase4b empties are likely **corrupted by this export bug**;
the true finetuned streaming CER is unknown until the export is fixed and re-evaluated. `[?]`

## 5c. Session-3 execution results (verified)
- **True baseline** (current finetuned zipformer, sherpa-onnx 1.13.3, greedy streaming, 250 Dev utts,
  min-two-refs): **CER 15.7% / WER 22.3%, 0 empties, RTF 0.147** (16 threads). Per-etiology: PD 10.3,
  Stroke 11.9, ALS 12.9, DS 20.3, CP 27.1. → real number; phase4b 26%/empties was the sherpa bug. `[V]`
- **Training env rebuilt** at `/workspace/.venv_train` (system torch 2.4.1+cu124 + k2
  1.24.4.dev20250715+cuda12.4 wheel + lhotse 1.33 + icefall editable). k2 CUDA + rnnt_loss verified,
  `finetune.py --help` runs. The April k2 source build at `/workspace/k2` was incomplete (no `_k2.so`). `[V]`
- **A1 recipe reality** (from finetune.py/datamodule args): SpecAugment = on-the-fly flag (easy);
  **speed-perturb is NOT a flag — needs regenerating cuts with 3-way perturb (data prep)**; **musan_cuts
  absent**; **--use-mux needs real LibriSpeech cuts** but `librispeech_cuts_*` are symlinks to the SAPC2
  cuts (so mux is a no-op / harmful) → anti-forget needs base cuts downloaded. `[V]`
  ⇒ "Full A1" needs 2 small prep steps (perturb cuts; fetch libri cuts) beyond just flipping flags.

## 5d. A1 launch outcome (Session 3) — data-starvation root cause + fix
A1 launched and trained correctly (loaded epoch-0 66.1M, 1,008,225 speed-perturbed cuts, SpecAug,
loss 0.308) BUT was **catastrophically data-starved**: GPU 0% util, ~20 s/batch, <50 batches in 17 min.
**Root cause (verified): `/workspace` is a MooseFS network FS** (`mfs#us-ca-2.runpod.net`, 1006T). With
`--on-the-fly-feats`, each batch does random small-file wav reads + fbank + speed-perturb resample over
the network → the H200 starves. Killed the run (GPU freed to 0 MiB). The pod has **2 TB RAM** (huge asset).

**Fix (next step): precompute features once, remove on-the-fly.** Standard icefall flow:
1. `compute_and_store_features` (lhotse Fbank, 80-dim, parallel over ~64–128 of the 192 cores) for the
   perturbed `data_a1/cuts_S.jsonl.gz` (and dev) → cuts WITH features.
2. **Store features on fast storage** — ideally `/dev/shm` (tmpfs, backed by the 2 TB RAM) or local disk —
   NOT random-read MFS. Packed lilcom-chunky reads are sequential/bucketed → no MFS random-read penalty.
3. Relaunch `run_a1.sh` **without `--on-the-fly-feats`**, manifest-dir pointing at the feature-augmented cuts.
   Expect GPU-bound training (then 16 epochs is hours, not days).
Open: confirm `/dev/shm` size + whether any local (non-MFS) disk exists; feature storage ~150–250 GB for 3×.
Note: SSH to pod went flaky (255s) intermittently — transient RunPod proxy; retry.

### 5d-update (Session 3 cont.): precompute works, but a NEW stall blocks training
- Env fix: **`lilcom` was missing** in `.venv_train` (the real cause of the earlier `BrokenProcessPool` —
  workers died on import; single-process surfaced the clean `ImportError`). `pip install lilcom` fixed it. `[V]`
- Feature precompute to `/dev/shm` works (smoke: 2000 train + 300 dev cuts, `has_features=True`). `[V]`
- Wiring: dev cuts must be named `librispeech_cuts_dev-clean/other.jsonl.gz` (valid uses `dev_clean_cuts()`). `[V]`
- **BLOCKER (unresolved): training stalls after batch 0.** With precomputed features, GPU engages
  (util 51%, 20 GB during sanity-check + batch 0) then **hangs at batch 0, GPU idle**, indefinitely.
  **Reproduces with `--num-workers 0` AND `4`** → NOT a dataloader/worker issue, NOT I/O. The hang is in
  the train-step path after the first batch. `[V]`
  - **Definitive next diagnostic**: `py-spy dump --pid <hung pid>` while stalled → exact hang location
    (suspects: a diagnostic/logging hook e.g. zipformer attn-entropy, autograd, or a CUDA sync). Then fix.
  - Fallback remains the **working 15.7% model** (untouched).

## 6. Assessment of "was the project set up properly?"
- **Rules/context were NOT the gap** — the prior CLAUDE.md had solid tracking discipline.
- **The real gaps**: (a) bugs weren't driven to resolution — the agent root-caused the empty-output bug
  then "stopped to write a document" and pivoted; (b) submissions weren't validated against Codabench's
  actual constraints (offline worker, dependency weight) before upload; (c) work sprawled across 7
  candidates / phase1b–4c / 3 checkouts with **no single source of truth** → state was easy to lose
  (hence the forgotten work). This file is the start of that source of truth.

## 7. Recommended path (lowest-risk to a WORKING submission first)
1. **Fix the Zipformer ONNX export** (`encoder_embed`→`encoder`) and verify **non-empty** output on a few
   Dev utts. Cheap, unblocks the current best Track-2 model.
2. **Re-evaluate the finetuned Zipformer** streaming CER+latency on a fixed Dev subset (real number).
3. **Build a self-contained, network-free Zipformer submission** (bundled ONNX, sherpa-onnx or minimal
   ORT, no NeMo) and **validate in local sapc2-runtime Docker offline** before uploading → get our first
   valid Codabench result.
4. **Re-finetune properly** (2–3 epochs, LR sweep) — biggest CER lever, the current one is under-trained.
5. Only then revisit Nemotron (higher ceiling) with the same self-contained packaging discipline.

## 8. Open items to confirm
- Does the Codabench worker actually block network at setup? (explains A definitively) `[?]`
- True finetuned-Zipformer streaming CER after the export fix. `[?]`
- Which checkpoint is best (`epoch-1` vs `epoch-4` vs `best-valid-loss`). `[?]`
