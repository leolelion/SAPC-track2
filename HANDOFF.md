# HANDOFF — SAPC2 Track 2 (read this first)

You are picking up a streaming dysarthric-ASR competition (SAPC2 Track 2: **minimize CER on a
CER×latency Pareto, CPU-only, real-time, ≤15000 s/submission**). A lot of prior work already exists.
**Do not start from scratch.** Read this, then `STATE_OF_WORK.md` and `research/05_synthesis_and_candidates.md`.

## TL;DR of where things stand
- **Working model exists**: a finetuned streaming Zipformer (66M) gives **CER 15.7% / WER 22.3%, 0 empty,
  RTF 0.15** on Dev (min-two-refs). This is the real baseline and the current best Track-2 system.
- The famous "empty output" bug was **just sherpa-onnx 1.12.35**; pin **sherpa-onnx ≥1.13.3** and it works.
- We're trying to beat 15.7% with a better finetune ("A1": speed-perturb + SpecAugment + more epochs).
- **BLOCKED**: A1 training **hangs at batch 0** (GPU engages then idles). Reproduces with num_workers 0 & 4
  → it's in the train-step path, not I/O. **Next step: `py-spy dump` the hung process** (details below).
- Don't submit yet — the user wants the improved model first. The 15.7% model is the safe fallback.

## Access
- Pod (RunPod, H200, 192 cores, 2 TB RAM): `ssh -i /Users/o/.runpod/ssh/RunPod-Key-Go -p 30723 root@38.80.152.249`
  - SSH is intermittently flaky (transient 255s); just retry.
  - `runpodctl ssh info 3dwiczo41jeg1y` if IP/port changed.
- Everything lives under **`/workspace`** (which is a **MooseFS network FS** — see "Gotchas").

## Repo situation (the "fresh repo" question)
- The same GitHub repo `github.com/leolelion/SAPC-track2` is checked out 3× on the pod:
  `/workspace/SAPC-template` (old, Mar), `/workspace/SAPC-track2` (Apr), **`/workspace/sapc-nemotron` (May, newest)**.
- The **local Mac repo** `/Users/o/Downloads/SAPC-template` (where these docs live) is the *oldest* (Mar) checkout.
- Recommendation: work on the **pod**. If you want a clean slate, consolidate into ONE working dir there;
  don't re-clone the stale Mac copy. The *value* is the data + env + models on the pod, not the repo name.

## VERIFIED facts you can trust (don't re-derive)
- **Data**: `/workspace/SAPC2/manifest/{Train,Dev}.csv` (cols `id,speaker,etiology,audio_filepath,duration,
  text,norm_text_with_disfluency,norm_text_without_disfluency`). Audio under `/workspace/SAPC2/processed/`.
  Train ≈ 336k utts / 743h; Dev ≈ 47,929 utts. 5 etiologies, **PD-dominant**; CP/DS hardest.
- **Eval worker** (Codabench, from one ingestion log): ≈24-core EPYC-Milan, 226 GB. NOT guaranteed →
  design for **real-time on ~8 cores int8 ONNX**.
- **Scoring** = min-over-two-refs CER/WER (sclite), CER primary; latency = mean(TTFT,TTLT) P50. (`utils/`)
- **Working finetuned model (ONNX)**: `/workspace/finetune/onnx/standard/{encoder,decoder,joiner}.onnx`
  + `bpe.model` + `tokens.txt`. Streaming zipformer-M, causal, chunk16/left128, 66.1M params.
- **Checkpoints**: `/workspace/finetune/exp/standard/epoch-{0,1,2}.pt` (epoch-0 = LibriSpeech base),
  `best-valid-loss.pt`. (Prior finetune was only ~2 epochs, LR 5e-4, no aug → text-overfit.)
- Prior offline (Track-1, not for us): Qwen3-ASR 6.5% CER, GEC/ROVER explored. LLM-GER is OUT for Track 2.

## How to reproduce the 15.7% baseline (works today)
```bash
python3 -m venv --system-site-packages /tmp/sherpa_env   # system has torch 2.4.1+cu124
/tmp/sherpa_env/bin/pip install sherpa-onnx               # gets >=1.13.3
# then OnlineRecognizer.from_transducer(tokens/encoder/decoder/joiner from finetune/onnx/standard),
# feed 100ms (1600-sample) chunks, greedy. See the eval script pattern we used (in chat / STATE_OF_WORK).
```
A 250-utt min-two-refs eval gave CER 15.7%. Build a proper fixed speaker-disjoint eval set + official
scorer (SCTK is at `/workspace/SCTK`) as the ruler for all experiments.

## The A1 experiment (the improvement attempt) — env + recipe
**Training env (rebuilt, works):** `/workspace/.venv_train` = system torch 2.4.1+cu124 + k2 wheel
`1.24.4.dev20250715+cuda12.4.torch2.4.1` + lhotse 1.33 + icefall (editable) + **lilcom**. k2/CUDA verified.
**Recipe (literature-backed, see research/06):** full FT from epoch-0, **3-way speed-perturb (0.9/1.0/1.1)
+ SpecAugment**, base-lr 0.0045, ~16 epochs, checkpoint-averaging, keep streaming config (chunk16/left128).
**Launch script**: `/workspace/finetune/run_a1.sh`. **Patched** `finetune.py` (icefall is a GigaSpeech
finetune recipe repurposed via cut filenames): filter uses `int(c.duration*100)` not `c.num_frames`.
**Perturbed cuts (done)**: `/workspace/finetune/data_a1/cuts_S.jsonl.gz` = 1,008,225 cuts (3× SAPC2).
**Data wiring**: train cuts must be `manifest_dir/cuts_S.jsonl.gz`; dev must be
`librispeech_cuts_dev-clean.jsonl.gz` + `dev-other` (valid uses `dev_clean_cuts()`); also `cuts_DEV.jsonl.gz`.

## THE BLOCKER + exact next step
Training (on-the-fly OR precomputed features) **hangs at batch 0**: GPU memory allocates, util spikes
during the OOM sanity-check + batch 0, then **util→0 and it sits at batch 0 forever**. Reproduces with
`--num-workers 0` and `4` → NOT dataloader/IO. It's in the train step after the first batch.
**Do this first:**
1. Relaunch the smoke run (precomputed feats in `/dev/shm/smoke`, see below), get the PID.
2. `/workspace/.venv_train/bin/pip install py-spy` then `py-spy dump --pid <pid>` while it's hung.
3. The stack shows the exact hang (suspects: zipformer diagnostics/attn-entropy logging hook, an
   autograd/grad-scaler step, a CUDA sync, or a finetune.py-specific path under torch 2.4.1).
4. Fix, confirm GPU stays busy + batches advance past 50, THEN scale up.
**Smoke setup (already on pod):** `/dev/shm/smoke/` has 2000 train + 300 dev precomputed-feature cuts.
Smoke launch = `run_a1.sh` args but `--manifest-dir /dev/shm/smoke`, NO `--on-the-fly-feats`,
`--exp-dir /workspace/finetune/exp/smoke`, `--max-duration 300`.

## After the stall is fixed: scale to full A1
- Precompute features for all 1.0M `data_a1/cuts_S` to `/dev/shm` (≈64 GB; /dev/shm = 117 GB).
  **Use `spawn` + a `if __name__=="__main__":` guard** for parallel precompute (fork-after-torch segfaults).
  Pin `OMP_NUM_THREADS=1`. (lilcom now installed.)
- Relaunch `run_a1.sh` WITHOUT `--on-the-fly-feats`, manifest-dir = the featured cuts dir.
- Verify GPU-bound (util high, fast batch rate) in the first 2 min before walking away.
- When done: average best checkpoints → export ONNX (icefall `export-onnx-streaming.py`) → eval with the
  official min-two-refs scorer vs the 15.7% baseline → log it.

## GOTCHAS (lessons — don't repeat)
- **`/workspace` is MooseFS (network)** → random small-file reads are slow. On-the-fly feature extraction
  STARVES the GPU. Always **precompute features to `/dev/shm` (RAM)** for training.
- **Pin sherpa-onnx ≥1.13.3** (1.12.35 gives empty output for this model).
- **lilcom** must be installed in the training venv (else cryptic `BrokenProcessPool`).
- **Parallel feature precompute**: fork-after-torch segfaults → use `spawn` + `__main__` guard, threads=1.
- **Smoke-test pipelines on tiny data + watch GPU util in the first 2 min** before any full run.
- Submission packaging must be **self-contained & network-free** (bundle ONNX, pin versions, NO NeMo,
  no HF download at setup) — that's why prior Codabench uploads failed. Validate in the Docker image offline.
- One source of truth: keep `STATE_OF_WORK.md` current. Log every run (incl. failures) with config+commit+env.

## Suggested first actions next session
1. Reconnect to pod; `py-spy` the batch-0 hang → fix it (the one thing blocking everything).
2. Build the fixed speaker-disjoint Dev eval + official scorer (the ruler).
3. Finish A1 (precompute→train→eval), compare to 15.7%, log it.
4. In parallel (independent): package the 15.7% model as a self-contained submission, validate offline
   (don't submit until A1 either beats it or is ruled out).

## Pointers
- `STATE_OF_WORK.md` — full audit + the blocker detail (§5d-update).
- `RESEARCH_PLAN.md`, `research/01..06` — literature survey + candidate decision + finetuning recipe.
- `EXPERIMENT_LOG.md` — `exp_base_ft` (15.7%) and `exp_001` (A1) entries.
- `PLAN.md` — phased roadmap. `CLAUDE.md` — contracts + house rules.
