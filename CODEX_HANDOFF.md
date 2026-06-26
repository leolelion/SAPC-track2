# Codex Handoff

> Written 2026-06-27. Supersedes the older `HANDOFF_NEMOTRON_CODEX.md` (which was from the
> earlier "is Nemotron viable / speed" era — still useful for the packaging recipe, but this
> file is the current state). Self-contained: you should not need the Claude chat.

## Goal
**SAPC2 Track 2 — streaming ASR on dysarthric speech, CPU-only, latency-scored, CER primary.**
We finetuned NVIDIA `nemotron-speech-streaming-en-0.6b` (Cache-Aware FastConformer-RNNT, NeMo
`EncDecRNNTBPEModel`) on the SAP dysarthric corpus and are now **deploying it as an ONNX-on-CPU
streaming submission** that must beat the current #1 baseline (a SAP-finetuned streaming
**Zipformer at 23.44% CER on hidden Test1**) on **both CER and latency**.
Final ranking = Pareto on **mean(TTFT, TTLT)** and **CER (min-over-two-refs, sclite)**.

## Current State
The finetune is DONE and strong; we are mid-way through the **deployment loop**, debugging a
streaming artifact. Immediate task in flight: **the "SOS-token-init" fix test** (see below).

What's done:
- Full 2-arm finetune (encoder-only + full-unfrozen), all 309k utts / 683 h, on the RunPod H200.
- Held-out eval, ONNX export at the deploy context `[70,1]`, and a **working streaming deployment
  harness** validated through the organizers' real `local_decode.py` (streaming CER + latency).
- Discovered a stream-start **"Wh " artifact** on the encoder-only model. An independent review
  (5th of 5 such reviews this project) found my root-cause was wrong and identified a **real
  SOS-token bug** in the deploy `model.py`. **A test of that fix is queued/running right now.**

## Repository / Branch State
- **Local branch:** `main`. **Do NOT push to `origin`** (it's the read-only upstream
  `xiuwenz2/SAPC-template`). All our commits are pushed to the **fork**:
  `git push fork HEAD:claude/asr-research-docs` (fork = `https://github.com/leolelion/SAPC-track2.git`).
- `git status`: **0 modified, ~28 untracked.** The untracked files are pre-existing planning docs
  + a `__pycache__` + a few older bench scripts; **all our real work IS committed** (scripts/ and
  research/16–22 are tracked). **Do not delete/reset/clean anything** — preserve all untracked files.
- Last commit: `c291893 deploy: SOS-init bug test ...`. The work is a linear series of commits;
  read `git log --oneline -30` for the full journey.

## Important Files (open these first)
1. **`research/22_deployment_results.md`** — the deployment results + the deploy-model decision.
2. **`research/20_fullrun_results.md`** — the full-finetune numbers (enc-only vs full-unfrozen).
3. **`scripts/run_sos_fix.sh`** — the IN-FLIGHT experiment: the SOS-token fix test. **This is the
   next result you need.** (`scripts/autorun_sosfix.sh` is the Mac-side launcher.)
4. **`scripts/nemo_finetune.py`** — the finetune script (NeMo). Holds all the training fixes.
5. **`scripts/build_deploy_sub.sh` / `run_deploy_fix2.sh` / `run_test_unfrozen.sh`** — the deploy
   harness build + the "Wh" debugging runs.
6. Context chain: `research/16` (characterization) → `17` (gate plan) → `18` (smoke) → `19`
   (pre-run review) → `21` (audit + ranked next steps) → `22` (deployment).
7. On the POD: `/workspace/finetune/nemo_submission/model.py` — the **deployed streaming harness**
   (local-mel + ONNX cache loop). Holds the SOS bug (`self._last_token = np.array([[0]])`).

## What Is Working
- **Finetune pipeline** end-to-end (NeMo venv, manifests, training, val, top-5 ckpt avg, eval).
- **The finetuned models** (held-out, NeMo `transcribe()` @ `[70,1]`, severe-enriched Dev_diag-425,
  vs same-path zero-shot 43.4% / 18% empty):
  - **encoder-only: 23.6% CER / 8% empty** (best on offline transcribe; frozen decoder+joint)
  - **full-unfrozen: 25.2% / 6%** ; internal-dev: enc 8.01% vs full 11.06% (full = overfit signature)
  - No catastrophic forgetting (Parkinson's/mild 11.2 → 4.8%).
- **The streaming deployment harness** (real `local_decode.py`, CPU, 100 ms chunks). NeMo's ONNX
  export I/O **exactly matches** our existing `model.py`, so deployment was adapt-not-rebuild; the
  only `[70,1]` change is `CHUNK_NEW` 56→16.
  - **Streaming CER on Dev_100 (clean read speech):** zero-shot base **5.2% clean**;
    FT **full-unfrozen 1.2% CLEAN**; FT **encoder-only 11.7% BROKEN** ("Wh " on 80/100).
- **Latency measured** (`compute_latency.py`, Dev_streaming 123 utts), full-unfrozen:
  **TTFT p50 1.20 s / p90 2.53 s, TTLT p50 0.29 s / p90 0.34 s** — better than the old zero-shot
  Nemotron submission (TTFT 2.27 s). (Track 2 is mean(TTFT,TTLT)-ranked.)

## What Is Broken / Failing
- **The "Wh " stream-start artifact** on the **encoder-only** model: a spurious leading `"Wh "`
  on ~80% of utterances under streaming (`local_decode`), inflating its streaming CER to 11.7% on
  Dev_100 (vs ~5% clean). The base and full-unfrozen do NOT have it.
- **LIKELY ROOT CAUSE (verified against NeMo source, not yet empirically confirmed): an SOS-token
  bug in `model.py`.** It primes the RNN-T prediction net with `self._last_token = np.array([[0]])`
  (= embedding of BPE piece #0), but NeMo's RNN-T uses **SOS = blank id (1024)** with
  `blank_as_pad=True`. So every utterance is decoded from the wrong start token.
  - Evidence: NeMo `rnnt.py predict(None)` uses a zero/blank SOS, `_SOS = blank_index`. Our two
    earlier "Wh" fixes (drop warmup frames; skip first chunk) **never touched `_last_token`**, so
    they could not distinguish this bug from an "encoder-rep" hypothesis — they were
    non-discriminating, and my earlier "frozen-joint mismatch" conclusion was under-determined.
- **The fix being tested now:** set `self._last_token = np.array([[BLANK_ID]])` (BLANK_ID=1024).
  **`scripts/run_sos_fix.sh` is running this on BOTH arms.** Expected: if SOS-bug, "Wh" vanishes on
  enc-only AND base stays clean → **salvages the better model (enc-only)**, possibly improves
  full-unfrozen too. If "Wh" persists → the encoder-rep story gains support and we keep full-unfrozen.

## Constraints / Rules
**Track / scoring:**
- CPU-only at eval. **15000 s per submission.** Size is NOT a blocker (an 821 MB zip ran before).
- Streaming: audio arrives in **100 ms (1600-sample) float32 16 kHz mono** chunks; emit partials.
- **Offline Codabench worker: NO network.** Bundle weights+wheels, `pip --no-index`. **No NeMo,
  no HuggingFace download** in the submission — that caused **7 prior upload failures** (numpy
  C-ABI break from `nemo_toolkit`). Use the **local-mel preprocessor + bundled `onnxruntime` wheel**
  (the working recipe is `nemo_submission/` on the pod + `HANDOFF_NEMOTRON_CODEX.md`).
- **NEVER edit the scorer**: `utils/compute_metrics.py`, `utils/compute_latency.py`, `evaluate.sh`,
  `steps/eval/*`, `local_decode.py` semantics. Wrap, don't modify.

**Repo / git:**
- `git add .` is forbidden — stage files individually. Commit msgs end with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` (Codex can use its own).
- Push to **fork** branch `claude/asr-research-docs`, never `origin`.

**Pod / infra (CRITICAL):**
- RunPod pod id **`3dwiczo41jeg1y`**, host **38.80.152.249**, **1×H200, $4.39/hr, often GPU-scarce**
  (starts can fail with "not enough free GPUs"; the autorun retries for ~6 h).
- **SSH:** `ssh -i /Users/o/.runpod/ssh/RunPod-Key-Go -p <PORT> -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@38.80.152.249`.
  **The SSH PORT changes on every pod restart** — get it from
  `runpodctl get pod 3dwiczo41jeg1y -a` (parse the `…:<PORT>->22 (pub,tcp)` field). There is a
  ~25 s race after RUNNING before the port is assigned — poll it.
- **`/workspace` = MooseFS network volume `sapc2-training` (id `p649ll2a2w`, US-CA-2) → PERSISTS**
  across stop/start/recreate. **The container disk RESETS each cycle** (system pip installs are lost).
  So: keep everything on `/workspace`. The **NeMo venv lives at `/workspace/nemoenv`** (persistent,
  idempotent — `setup_nemo_venv.sh` reuses if present).
- **Stop the pod when done** (cost). Autorun pattern uses a **PID-lock** at `/tmp/sapc_autorun.lock`:
  only the newest autorun stops the pod (prevents a stale watcher from killing a live run — that
  happened once, killed a full run). **Only run ONE autorun at a time; kill strays first**
  (`ps aux | grep autorun_`).
- **Mac gotchas:** the system `head` is a broken perl tool — **never pipe to `head`** (use
  `sed -n`/`awk`). **zsh does NOT word-split unquoted vars** — put ssh options inline, not in `$O`.

**Modeling decisions already made:**
- Deploy context **pinned `[70,1]`** (160 ms lookahead, low latency). Trained CASED targets
  (`norm_text...` ablation was a wash; cased = NVIDIA-native style, scorer normalizes both sides).
- Streaming chunk constants for `[70,1]`: `CHUNK_NEW=16` mel, `CACHE_FRAMES=9`, `last_channel_cache=70`,
  `valid_out_len=2`, `drop_extra_pre_encoded=2` (from NeMo `streaming_cfg`).

## Commands to Reproduce
All heavy work runs on the pod via Mac-side **autorun** wrappers (wait-for-GPU → run → pull log →
stop pod). General pattern:
```bash
cd /Users/o/Downloads/SAPC-template
# 1) ensure no stray autorun:
ps aux | grep autorun_ | grep -v grep
# 2) launch (always as a backgrounded harness task so you get notified on completion):
bash scripts/autorun_sosfix.sh        # the IN-FLIGHT SOS-fix test
# 3) results land in /Users/o/Downloads/sos_fix.log (pulled by the autorun) and on the pod at
#    /workspace/finetune/nemo_ft/sos_fix.log
```
Manual pod inspection (get PORT first):
```bash
runpodctl get pod 3dwiczo41jeg1y -a              # find :<PORT>->22, and RUNNING/EXITED
ssh -i /Users/o/.runpod/ssh/RunPod-Key-Go -p <PORT> -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null root@38.80.152.249 'tail -40 /workspace/finetune/nemo_ft/sos_fix.log'
runpodctl pod stop 3dwiczo41jeg1y                # ALWAYS stop when done
```
**Build a `[70,1]` streaming submission dir** (the deploy recipe; see `run_sos_fix.sh::build_sosfix`):
```bash
# on pod, in NeMo venv:
SUB=/workspace/finetune/nemo_ft/deploy_X
cp -r /workspace/finetune/nemo_submission "$SUB"
rm -f "$SUB"/weights/encoder_model.onnx* "$SUB"/weights/decoder_model.onnx*
cp /workspace/finetune/nemo_ft/export_<MODEL>70_1/* "$SUB"/weights/
mv "$SUB"/weights/encoder-model.onnx       "$SUB"/weights/encoder_model.onnx
mv "$SUB"/weights/decoder_joint-model.onnx "$SUB"/weights/decoder_model.onnx
sed -i 's/^CHUNK_NEW = 56/CHUNK_NEW = 16/' "$SUB"/model.py
# THE SOS FIX under test:
sed -i 's/np.array(\[\[0\]\], dtype=np.int32)/np.array([[BLANK_ID]], dtype=np.int32)/g' "$SUB"/model.py
# then run the REAL harness:
LD=/workspace/sapc-nemotron/track2_starting_kit/local_decode.py
cd "$SUB" && SAPC2_THREADS=4 python3 "$LD" --submission-dir "$SUB" \
  --manifest-csv /workspace/SAPC2/manifest/Dev_100.csv \
  --streaming-manifest-csv /workspace/SAPC2/manifest/Dev_streaming.csv \
  --data-root /workspace/SAPC2 --out-csv /dev/shm/x.csv --out-partial-json /dev/shm/x.json
# CER: two-ref via /workspace/sapc-nemotron/utils normalizer (see nemo_eval_diag.py / run_sos_fix.sh::cer_check)
# Latency: python3 /workspace/sapc-nemotron/utils/compute_latency.py --partial-json /dev/shm/x.json \
#          --manifest-csv /workspace/SAPC2/manifest/Dev_streaming.csv
```
**Key pod paths:** base `.nemo` = `/workspace/finetune/nemo_ft/nemotron-speech-streaming-en-0.6b.nemo`;
finetuned `.nemo`s = `full_enc/ft_smoke_encoder_only.nemo` (enc-only) & `full_unfrozen/ft_smoke_full.nemo`;
ONNX exports = `export_{full,unfrozen,base}70_1/`; data manifests = `/workspace/SAPC2/manifest/`
(`Train.csv`, `Dev.csv`, `Dev_100.csv`, `Dev_streaming.csv`, `Dev_diag.csv` = severe-enriched 425).

## Debugging History (so you don't repeat it)
- **Whole-project arc:** zero-shot Nemotron scored 51% on Test1 → diagnosed (via faithful local
  harness) NOT a packaging/speed/quant bug but **domain failure** (empties on severe dysarthria) →
  finetuning is the fix (literature + our own evidence). All in research/10–15.
- **Finetune build** shook out (cheap gates): Noam-LR-is-a-scale trap (→ explicit AdamW+cosine);
  Lhotse dataloader `num_buckets=None` (→ classic dataloader `use_lhotse=False`); Lightning
  `val_check_interval > batches` (→ epoch-based val); `torch.load weights_only` (→ False);
  checkpoint-avg buffer bug (→ float params only). A **stale autorun once `pod stop`ped a live run**
  → added the PID-lock guard. NeMo's `speech_to_text_cache_aware_streaming_infer.py` is **Hydra-based
  and hangs on `--help`** — don't try to drive it with argparse flags; we built our own harness instead.
- **"Wh" fixes that FAILED (do not retry):**
  1. `run_deploy_fix.sh`: drop first 2 encoder frames on chunk 0 → "Wh" persisted.
  2. `run_deploy_fix2.sh`: skip ALL chunk-0 emission (it's leading silence) → "Wh" persisted.
  Conclusion: it's NOT a warmup/timing artifact. Both fixes ignored `_last_token`, so they were
  non-discriminating w.r.t. the SOS hypothesis.
- **`run_test_unfrozen.sh`:** full-unfrozen streams CLEAN (1.2%, 0 "Wh") → led to (provisionally)
  choosing full-unfrozen to ship. **The 5th independent review then flagged this is premature**
  (full-unfrozen is the worse model on every offline axis; "Wh" likely the SOS bug = a 1-line
  harness fix). Hence the SOS test now in flight.

## Likely Root Cause
**The deploy `model.py` SOS-token init is wrong:** `self._last_token = np.array([[0]])` should be
`np.array([[BLANK_ID]])` (BLANK_ID = 1024 = blank, since `blank_as_pad=True`). NeMo's RNN-T starts
each sequence from the blank/zero SOS, not the embedding of token 0. The finetuned encoder's
first-frame activation, decoded from the wrong SOS, yields the spurious `"Wh "`. **Evidence:** NeMo
source (`rnnt.py`, `rnnt_greedy_decoding.py` `_SOS = blank_index`); both prior fixes never touched
`_last_token` and both failed; base streams clean because its encoder's first frame doesn't cross the
joint's (token-0-biased) boundary. **Not yet empirically confirmed — that's the running test.**

## Recommended Next Steps (ordered checklist)
1. **[IN FLIGHT] Get the SOS-fix result** (`scripts/run_sos_fix.sh` via `autorun_sosfix.sh`; log at
   `/Users/o/Downloads/sos_fix.log` or pod `…/nemo_ft/sos_fix.log`). Decision rule:
   - enc-only "Wh" GONE + CER ≈ base → **SOS bug confirmed; ship encoder-only (the better model)**.
   - "Wh" persists → keep full-unfrozen; the encoder-rep story stands.
   - Either way, persist the conclusion to a new `research/23_*.md`.
2. **Severe + representative STREAMING CER** for the chosen model: run `local_decode` on
   **`Dev_diag.csv` (severe-enriched 425)** and a representative Dev sample, with **speaker-level
   (blockwise) bootstrap CI**. This converts "beats zipformer 23.44%?" from projection to fact.
   (Adapt `nemo_eval_diag.py`'s CER to read `local_decode` out-csv; add a bootstrap.)
3. **TTFT investigation** (it's HALF the rank): measured 1.20 s but the published ONNX-CPU impl of
   this model reports ~0.56 s algorithmic delay — check the `model.py` emit cadence / partial-callback
   buffering; cheap latency wins move Pareto rank as much as CER.
4. **int8 quantize the ENCODER only** (keep decoder+joint FP32 — lit-backed), measure CER delta on
   dysarthric audio. Helps the 15000 s CPU budget + size.
5. **Offline packaging dry-run** in a clean/offline container (the 7-failure risk): local-mel +
   bundled `onnxruntime` wheel, `pip --no-index`, NO NeMo/numpy-wheel. Then a **faithful Dev gate**
   (`local_decode` + `evaluate.sh`) BEFORE any Codabench upload. (Recipe: pod `nemo_submission/` +
   `HANDOFF_NEMOTRON_CODEX.md` + `submission-offline-packaging` memory.)
6. **Then** (optional, higher ceiling): a **converged retrain** — 8–12 epochs, full-dev val,
   clean-tail checkpoint averaging, and **multi-lookahead training** (`att_context_probs` over the
   list) to restore the latency lever we pinned away. Drop the losing arm to halve cost. Then
   **severity-aware sampling** for residual ALS/DS empties.
   **Deprioritize:** LoRA (no forgetting observed), beam/LM fusion (raises TTFT on a latency track),
   TTS augmentation (heavy).

## Open Questions
- **Does the SOS fix remove "Wh" and salvage encoder-only?** (the running test answers this.)
- Does the exported ONNX **decoder embed `targets` internally** (we inferred yes from input names)?
  If it expects a pre-embedded or already-blank SOS, the fix's exact form may differ — inspect the
  decoder ONNX graph if `[[BLANK_ID]]` doesn't fully fix it.
- **Real (streaming, dysarthric) CER** for the chosen model is unmeasured beyond clean Dev_100 — does
  it actually beat 23.44%? (severe Dev_diag transcribe said 23.6–25.2%; streaming likely similar.)
- **int8-vs-fp32 CER delta on dysarthric audio** — unmeasured.
- Is **1.20 s TTFT competitive**, and can the wrapper get to ~0.56 s?
- The v1 finetune is **under-converged** (4 epochs, noisy early-stop) — how much CER is left on the
  table by a proper converged + multi-lookahead retrain?
