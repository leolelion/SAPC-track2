# Parakeet Arm A — ship runbook (written 2026-07-29, local prep, no pod spend yet)

Objective: get the banked parakeet **Arm A** onto Codabench as a low-latency frontier entry.
Arm A already satisfies the pre-registered win condition on the official scorer:

| | Arm A | zipformer beam-4 (shipped) |
|---|---|---|
| Dev_diag severe CER | **18.69%** (WER 24.97) | 22.48% |
| Dev_clean2k CER (2000u / 122 spk) | **13.51%** | 18.19% |
| mean(TTFT, TTLT) | **356 ms** (638 / 74) | ~611 ms |

Test1 board's best latency is 592 ms, so 356 ms is a non-dominated corner at almost any CER.
Weights: pod `1ppb7l0i5xuna8` (STOPPED, disk persists) `/workspace/parakeet_ft/armA_full/ft_smoke_encoder_only.nemo`.

---

## Done locally (2026-07-29) — no pod needed

1. **Thread policy fixed.** `parakeet_realtime_ft/model.py` sized torch threads to the *cgroup
   quota*. On the eval worker (24 vCPU, quota == nproc) that is a no-op, so each of ~20 ingestion
   worker processes would have taken 24 threads → 480 threads on 24 vCPU. Under this exact topology
   `exp_nemotron_speed_002` measured threads=1 at 4.2x the wall-clock throughput of threads=4, with
   byte-identical transcripts. Quota is now a **ceiling**; the value comes from `SAPC2_THREADS`
   (env) → `config.yaml runtime.num_threads` → default **1**.
   Verified: `python3 scripts/test_thread_policy.py` (8/8).
2. **Contract smoke repaired** — it had been broken since the input-gain commit and nobody noticed,
   so the 5-method interface went three commits with no local gate while `model.py` was edited.
   Two root causes, both fixed rather than patched:
   - the smoke hand-duplicated `__init__`'s attribute list, which rotted every time a config
     attribute was added → the config/env setup is now one NeMo-free method, `_init_runtime_cfg()`,
     called by both `__init__` and the smoke;
   - the mock predated the bug#2 feature-frame rewrite → the mock is now NeMo-shaped and the smoke
     drives the **real** `_setup_streaming()`.
   Verified: `python3 scripts/smoke_parakeet_ft_wrapper.py` (28/28), including a feature-cache
   on/off bit-equivalence check. That check was negative-controlled: injecting `_feat_margin` 0 or 1
   (geometrically wrong) is caught; 2/3/4 are all correct (minimum correct is
   `ceil(n_fft/2/hop) = 2`; shipped is 4) and agree.

---

## BLOCKER — parakeet has never been packaged for Codabench, and NeMo is the reason

`track2_starting_kit/parakeet_realtime_ft/setup.sh` makes **two network calls**: `pip install
nemo_toolkit[asr]==2.7.2` from PyPI, and `ASRModel.from_pretrained(...)` from HuggingFace. It has
only ever run on a pod where both work. There is no `weights/` in the dir and no packaged zip.

Two pieces of our own evidence point in opposite directions, and I have not resolved which holds:

- Memory `submission-offline-packaging` records that the project's **recurring** submission failure
  was exactly this — "network-dependent setup.sh: HuggingFace downloads, heavy `nemo_toolkit[asr]`"
  — solved on 2026-06-21 by bundling wheels and installing `--no-index`.
- The organizers' own `streaming_zipformer/setup.sh` installs k2/kaldifeat from `huggingface.co`
  URLs, which implies network *is* reachable at setup time.

Reconciling theories (do not pick one without evidence): (a) the worker is hard-offline and the
baseline is served from a cache/whitelist; (b) network works but a heavy dependency tree times out or
resolves against the runtime's torch 2.5.0+cu124 badly. Both make bundling the derisked path.

The circumstantial evidence is strong regardless: **every submission we have ever successfully
scored — A1 greedy, beam-4, Nemotron int8 — ran ONNX via sherpa-onnx/onnxruntime. None ran NeMo.**

So "package Arm A" is not a packaging task. It is a **decision**, and it is Q's:

| | Option 1 — bundle NeMo | Option 2 — export to ONNX |
|---|---|---|
| Work | `pip download nemo_toolkit[asr]==2.7.2` + full tree as wheels; `--no-index` install; bundle the FT `.nemo` | export cache-aware streaming encoder + prediction net + joiner to ONNX; rewrite `model.py` against onnxruntime; int8 |
| Size | large (NeMo tree is heavy; A1 zip was 69 MB, Nemotron 803 MB) | small; int8 ≈ fp32 accuracy at ~3.6x smaller |
| Risk | dependency resolution against torch 2.5.0+cu124; unproven for us | export fidelity — but `h7-fix-plan` **resolved** this: our Nemotron export was faithful (base 43.4 ≈ NeMo-stream 43.8 ≈ our export 43.4) |
| Precedent | none of our scored submissions | all of them |
| Re-gate needed | rerun the Dev gate on the extracted zip | full re-gate: export changes the numerics |

Option 2 is the better-evidenced path but is real engineering; Option 1 is faster if the worker
does have network. **Do not start a pod until Q picks one** — the pod work differs completely.

---

## Pod plan once Q decides (single session, stop immediately after the decision metric)

Restart `1ppb7l0i5xuna8` (disk persists: Arm A ckpt, SAPC2 data, `nemoenv`). Upload only the changed
`model.py` + `config.yaml` + `scripts/test_thread_policy.py` + `scripts/smoke_parakeet_ft_wrapper.py`
(get explicit approval for the file list first — sandbox policy).

**Step 1 — thread sweep (the one number that is still guessed).** The default of 1 is chosen for the
budget, but if per-chunk compute at 1 thread exceeds the 100 ms chunk period the stream falls behind
real time and the 356 ms corner — our entire reason to ship this — collapses.

```bash
for T in 1 2 4; do
  SAPC2_THREADS=$T python3 track2_starting_kit/local_decode.py \
    --submission-dir ./parakeet_realtime_ft \
    --manifest-csv  $DATA/manifest/Dev_streaming.csv \
    --streaming-manifest-csv $DATA/manifest/Dev_streaming.csv \
    --data-root $DATA --out-csv dev_t$T.csv --out-partial-json dev_t$T.json
done
./evaluate.sh --start_stage 3 --stop_stage 3 --partial-json dev_t$T.json \
  --manifest-csv $DATA/manifest/Dev_streaming.csv
```

**Pre-registered:** pick the LOWEST thread count whose mean(TTFT,TTLT) stays **≤ 420 ms**. If even
threads=4 misses it, the corner is thread-bound — stop and report, do not raise the default blind.

**Step 2 — wall-clock budget check.** Accuracy pass at the chosen thread count under ~20 concurrent
workers, wall-clock extrapolated to Test size. Gate: projected wall **≤ 15000 s with ≥ 30% margin**.

**Step 3 — full gate on the EXTRACTED ZIP, not the source dir.** House rule
`validate-against-real-harness`: real `local_decode.py` (both passes) → `evaluate.sh` → official
sclite, on Dev_diag severe + Dev_clean2k, run from the unzipped submission artifact.

**Pre-registered ship criterion — write it before the run, submit IFF all four hold:**
- severe CER **≤ 20%** (Arm A is 18.69%; >20% means packaging changed the model)
- Dev_clean2k CER **≤ 15%** (banked 13.51%)
- mean(TTFT,TTLT) **≤ 420 ms**
- projected Test wall-clock **≤ 15000 s** with ≥ 30% margin

Any miss → do not submit, report the delta. **Never submit to test a hypothesis.**

**Stop condition.** Copy back metrics JSON + logs, stop the pod the moment the four numbers exist.

---

## Known and accepted, not blockers

- **48/425 severe empties (11.3%)** remain. Arm A wins both slices *carrying* them. Joint unfreeze
  (D1) and causal input gain were both falsified against this tail; see `experiments/PLANNED.md`.
  The untried lever is a **zipformer fallback on empty finalization** — on those same 48 utterances
  the fine-tuned zipformer transcribes 28/48 (9 perfect), it fires on ~11% of utterances, runs after
  audio end so it never touches TTFT, and TTLT is scored at p50 so the tail does not move it. Costs
  package size and a second model in memory. Upside, not a ship blocker — do it after shipping.
- Input gain ships **default-off** (falsified: 0/48 recovered through the real harness).
- Feature cache (Fix 2) ships **on**: proven equivalent, not a speedup — free insurance on long utts.
