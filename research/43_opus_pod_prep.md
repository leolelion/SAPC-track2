# 43 — Opus pod-prep for the Zipformer cheap-wins session (2026-07-21)

> Local-only prep session (darwin box, no SAP data, Linux-only wheels absent → **no
> faithful-harness runs executed here**). This note stages the next cost-gated pod so
> that session is *run → copy → stop*. Executes `OPUS_STEP_BY_STEP_PLAN.md` §3–§8.
> **No CER/latency numbers are claimed anywhere in this file — none were measured.**

Base commit: `0af119c`. Follows the repo house rules: pristine `streaming_zipformer/`
untouched; every variant is a sibling dir; scoring code (`local_decode.py`,
`evaluate.sh`, `utils/compute_*`, `steps/eval/*`) wrapped, never modified.

---

## What this session accomplished (all local, all verified)

### Task A — hardened the speaker-disjoint split tool
`scripts/make_speaker_disjoint_dev.py` (stdlib-only) gained two optional audit outputs:
- `--heldout-speakers-out <path>` — the held-out speaker ids, one per line, sorted.
- `--summary-json <path>` — row/speaker counts, per-etiology row counts for pool &
  heldout, the streaming id-parity flag, and the leakage assertion result
  (`no_speaker_leakage`, `speaker_leakage: []`).

Verified: `py_compile` clean; synthetic self-test (5 etiologies × 4 speakers × 2 utts)
proved **zero pool/heldout speaker overlap**, all 5 etiologies present on both sides,
streaming-id parity, and both new outputs correct. The pre-existing behaviour
(speaker-column with id-prefix fallback, optional `--streaming-csv`, id-matched
streaming subsets) was confirmed intact.

### Task B — config-only Zipformer latency/beam variants (5 dirs)
Each is `cp -a` of the pristine dir with **only `config.yaml` values changed**
(model.py/setup.sh byte-identical to the baseline). YAML validated for all:

| dir | chunk_size | left_context_frames | method | num_active_paths |
|---|---:|---:|---|---:|
| `streaming_zipformer` (pristine, unchanged) | 16 | 128 | modified_beam_search | 4 |
| `streaming_zipformer_beam8` | 16 | 128 | modified_beam_search | **8** |
| `streaming_zipformer_cs8_lc128` | **8** | 128 | modified_beam_search | 4 |
| `streaming_zipformer_cs16_lc64` | 16 | **64** | modified_beam_search | 4 |
| `streaming_zipformer_cs16_lc256` | 16 | **256** | modified_beam_search | 4 |
| `streaming_zipformer_cs32_lc128` | **32** | 128 | modified_beam_search | 4 |

Rationale: `chunk_size` is a *direct* TTFT lever (`model.py:141-145`: `is_ready`
needs `chunk_size*2 + pad` frames) with zero retraining; `left_context_frames` trades
accuracy vs per-step compute; beam-8 is the never-measured accuracy datapoint
(research/09 predicted ~18.3% CER at unknown latency). Together they map the
CER × mean(TTFT,TTLT) surface the Pareto prize actually scores.

### Task C — earlyemit variant: **intentionally SKIPPED (not applicable)**
Plan §5 gate: create an earlyemit variant *only if* `model.py` suppresses early
non-empty partials. It does not. `_decode_available()` (`model.py:368-377`) calls
`self._partial_callback(text)` **unconditionally after every decoded chunk** — no
`if text:` guard, no buffering, no "wait N chunks" policy. Grep for
`partial_callback|decoding_result|is_ready|done` surfaces no suppression path. TTFT is
gated purely by `is_ready` needing `chunk_size*2 + pad` frames — inherent chunk-size
latency already swept by the cs8/cs16/cs32 variants above. So there is nothing an
earlyemit variant could change. It was removed from the pod sweep loop.

### Task D — RNN-LM shallow-fusion **skeleton** (blocker documented, not overbuilt)
`streaming_zipformer_lm/` is a `cp -a` clone whose `config.yaml` adds an **inert**
placeholder block under `decoding:`:
```yaml
  lm_scale: 0.0                   # sweep grid on pod: 0.05, 0.1, 0.2, 0.3
  lm_path: lm/epoch-best.pt       # relative to this submission dir; trained on SAPC2 text
```
**Blocker (do not skip):** `model.py` does **not** read `lm_scale`/`lm_path` — these
keys are currently no-ops (`lm_scale: 0.0` ⇒ byte-identical to the pristine baseline).

**Prior art discovered this session (research/08, 2026-06-21) — read before any LM work:**
- `scripts/build_lm.sh` — trains a **KenLM n-gram** from transcripts. NB it reads a
  `norm_text_without_disfluency` column that is NOT in the Track-2 manifest schema
  (`utils/manifest.py`: id,speaker,etiology,audio_filepath,duration,text) — it targets
  the older sapc-nemotron manifest, so it is **not** directly reusable for the zipformer.
- `scripts/decode_lm.py` — decodes via **sherpa-onnx** `OnlineRecognizer.from_transducer(
  lm=…, lm_scale=…)`, i.e. a **natively-supported online shallow-fusion path**.

This surfaces **two candidate LM routes**, and the earlier "streaming path may not support
fusion" caveat applies to only one of them:
- **Route A (icefall-python, the current shipped runtime).** `streaming_zipformer/model.py`
  uses icefall's `decode_one_chunk`/`streaming_decode` on the PyTorch `epoch-30.pt`. Whether
  the *streaming* beam search takes an LM is the open question (offline `beam_search.py` has
  `modified_beam_search_lm_shallow_fusion`; the streaming path historically may not). Probed
  by `scripts/pod_train_rnnlm.sh` Stage 0 (feasibility gate — exits 3 if no hook).
- **Route B (sherpa-onnx).** `decode_lm.py` already demonstrates `from_transducer(lm=,
  lm_scale=)` works online — **lower fusion risk** — but requires exporting the zipformer
  checkpoint to ONNX (encoder/decoder/joiner.onnx), a step the current submission does not do.

Prep staged this session (both routes share the corpus step):
1. **Corpus (DONE + self-tested, pod-independent):** `scripts/prepare_lm_text.py` extracts a
   clean one-transcript-per-line corpus from the `text` column of any manifest(s). Correct
   for the Track-2 schema; NEVER pass Dev_heldout/Test (eval leak). Repeatable `--manifest-csv`,
   optional `--dedup`/`--lowercase`/`--stats-json`. `py_compile` clean; self-test verifies
   pooling, empty-drop, whitespace-collapse, dedup, lowercase.
2. **Route-A training recipe (staged):** `scripts/pod_train_rnnlm.sh` — Stage 0 feasibility
   gate FIRST (grep the streaming decode path for an LM hook; `exit 3` if absent so no GPU is
   wasted), then corpus build (calls #1), then a VERIFY-gated icefall `rnn_lm` train stub
   (paths must be checked against the unpinned icefall HEAD), then the wiring+sweep checklist.
   `bash -n` clean.
3. **Wiring + sweep (pod-only, gated):** wire the LM into `streaming_zipformer_lm/model.py`
   **(sibling only)** using the exact API Stage 0 reveals; set `decoding.lm_path`; sweep
   `lm_scale ∈ {0.05,0.1,0.2,0.3}`; gate on faithful `Dev_heldout` CER **and** per-etiology
   (an LM prior can worsen severe-tail deletions/empties — stop or move to density-ratio/ILME
   if the ALS/DS tail regresses; do not ship a tail-regressing LM).

If neither route supports shallow fusion cleanly, **do not hack the scorer or baseline** —
write the blocker into `research/44…` and let the latency sweep (Task B variants) carry the
session. **Recommendation:** run Route A's Stage-0 gate first (free); if it blocks, evaluate
Route B (sherpa-onnx export) rather than hacking icefall's streaming beam search.

### Task E — pod script now matches the actual variant set
`scripts/pod_run_lm_and_latency_sweep.sh` (`bash -n` clean):
- New `maybe_run_and_score` wrapper: runs a variant **iff its dir exists**, else prints
  `[SKIP] … not present` and continues. LM and beam8 are now guarded (so a partial
  variant set never aborts the run); pristine baseline (`exp_000`) stays unconditional.
- `earlyemit_cs16` removed from the sweep loop (Task C).
- Documented at the top: **`evaluate.sh` lines 36-38 (`DATA_ROOT`/`PROJ_ROOT`/`SCTK_DIR`)
  are unconditional assignments that clobber exported env vars** → they must be edited
  on the pod. Filling those path placeholders is config, not a scoring-semantics change.

---

## Exact files changed this session
- `scripts/make_speaker_disjoint_dev.py` — added `--heldout-speakers-out`, `--summary-json` (created by Fugu, extended here).
- `scripts/pod_run_lm_and_latency_sweep.sh` — skip-if-missing wrapper, dropped earlyemit, documented evaluate.sh precondition.
- `scripts/prepare_lm_text.py` — NEW: LM corpus extractor (Track-2 `text` column, stdlib, self-tested).
- `scripts/pod_train_rnnlm.sh` — NEW: Route-A icefall rnn_lm recipe with a Stage-0 feasibility gate.
- `track2_starting_kit/streaming_zipformer_beam8/` — NEW (config: paths=8).
- `track2_starting_kit/streaming_zipformer_cs8_lc128/` — NEW (config: chunk_size=8).
- `track2_starting_kit/streaming_zipformer_cs16_lc64/` — NEW (config: left_context=64).
- `track2_starting_kit/streaming_zipformer_cs16_lc256/` — NEW (config: left_context=256).
- `track2_starting_kit/streaming_zipformer_cs32_lc128/` — NEW (config: chunk_size=32).
- `track2_starting_kit/streaming_zipformer_lm/` — NEW (config: inert lm_scale/lm_path skeleton).
- `research/43_opus_pod_prep.md` — this note.
- Pristine `track2_starting_kit/streaming_zipformer/` — **unchanged** (verified via git).

---

## Cost gate — what must be approved before ANY pod starts

**File list to upload/sync** (small; all config + scripts, no weights):
- `scripts/make_speaker_disjoint_dev.py`
- `scripts/pod_run_lm_and_latency_sweep.sh`
- `track2_starting_kit/streaming_zipformer_beam8/`
- `track2_starting_kit/streaming_zipformer_cs8_lc128/`
- `track2_starting_kit/streaming_zipformer_cs16_lc64/`
- `track2_starting_kit/streaming_zipformer_cs16_lc256/`
- `track2_starting_kit/streaming_zipformer_cs32_lc128/`
- `track2_starting_kit/streaming_zipformer_lm/` (skeleton; only used once LM wiring lands)
- `research/42_fugu_independent_review.md`, `research/43_opus_pod_prep.md`
- `experiments/summary.csv`
(Each submission dir still needs its `setup.sh` run on the pod to pull icefall + weights.)

**Exact run command (after approval, on the pod):**
```bash
cd /workspace/SAPC-template
export DATA_ROOT=/workspace/SAPC2_DATA        # adjust to the real pod data path
export PROJ_ROOT=/workspace/SAPC-template
# EDIT evaluate.sh lines 36-38 (DATA_ROOT/PROJ_ROOT/SCTK_DIR) — exports do NOT reach it.
bash scripts/pod_run_lm_and_latency_sweep.sh 2>&1 | tee experiments/pod_run_lm_and_latency_sweep.log
```

**Minimum experiment sequence:** smoke (first 20–50 heldout utts if subset-able without
touching scoring) → guardrail (full `Dev_heldout` for pristine beam-4) → then variants.

**Decision metric:** a variant ships only if it is **non-dominated on the
speaker-disjoint `Dev_heldout` gate** vs beam-4 on `(CER, mean(TTFT,TTLT))`. An in-pool
Dev win that regresses `Dev_heldout` does **not** ship (research/37: Dev→Test transfer
is the binding risk).

**Expected runtime:** unknown precisely (never run on this data); dominated by
`local_decode.py` Pass-2 real-time 100 ms pacing over `Dev_heldout` × up to 7 variants.
The LM block is skipped until wiring exists, so the first pod pass is baseline + beam8 +
4 latency clones. Budget the session to the point where the decision metric is known,
then stop — do not leave the pod running.

**Stop condition / artifacts to copy back:** once every attempted variant has
`experiments/exp_*/` with `accuracy.log`, `latency.log`, `*.predict.csv`,
`*.partial.json`, `config.snapshot.yaml`, `git_commit.txt` — plus
`experiments/pod_run_lm_and_latency_sweep.log` and the generated
`Dev_heldout*.csv` manifests — scp them back and **stop the pod immediately**.

---

## Post-pod (local) follow-up
Parse the logs into `experiments/summary.csv` rows, then write
`research/44_zipformer_lm_latency_results.md` with the CER/WER/TTFT/TTLT table, a sorted
Pareto view, per-etiology if available, and a ship / no-ship call. If a variant dominates
beam-4 on the heldout gate, package it for Codabench (memory `submission-offline-packaging`);
if none wins, beam-4 stays live and the next bet is a severity-aware Zipformer retrain
(a funded GPU session), **not** Nemotron.
