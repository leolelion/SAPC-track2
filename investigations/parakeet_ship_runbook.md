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

## DECIDED 2026-07-29 (Q): **Option 2 — export to ONNX.** Local build is done.

Rationale below stands; the decision resolves it. Everything needed for the pod session now
exists locally and is exercised by a local smoke that passes 30/30 with no weights:

| file | what it is | runs where |
|---|---|---|
| `track2_starting_kit/parakeet_onnx/model.py` | ORT wrapper: NeMo-free, 5-method contract, feature-frame stepping ported from the validated NeMo sibling, hand-rolled RNN-T greedy | Codabench |
| `track2_starting_kit/parakeet_onnx/localmel.py` | mel front-end, parameterised from `streaming_meta.json` (the Nemotron one hard-coded its constants) | Codabench |
| `.../config.yaml`, `setup.sh`, `requirements.txt` | runtime knobs; `--no-index` offline install; empty requirements | Codabench |
| `scripts/export_parakeet_onnx.py` | `.nemo` → encoder/decoder ONNX + tokens + filterbank/window + `streaming_meta.json` | pod |
| `scripts/parity_parakeet_onnx.py` | fidelity gate: feature / encoder-tensor / transcript parity vs the NeMo wrapper | pod |
| `scripts/quantize_nemotron_encoder.py` | reused as-is for int8 encoder (generic over a submission tree) | pod |
| `scripts/package_parakeet_onnx.py` | zip + offline-validity asserts (no-network setup.sh, root layout, external `.onnx.data`, parity-set policies) | pod |
| `scripts/run_parakeet_onnx_pod.sh` | the whole paid session in order, stage-selectable | pod |
| `scripts/smoke_parakeet_onnx.py` | local contract/cadence/policy/feature-cache smoke, mock ORT | Mac ✅ 30/30 |

**Two knobs deliberately left unset.** NeMo applies `drop_extra_pre_encoded`
(`cache_aware_stream_step`) and `valid_out_len` (`streaming_post_process`) *outside*
`encoder.forward` — which is what gets exported. Whether NeMo's exporter bakes either into
the graph is an empirical fact, not something to reason out from the source, and getting it
wrong yields a fluent-but-wrong transcript (the Nemotron failure mode). So they are policies
(`encoder.drop_policy`, `encoder.trim_policy`), the parity script sweeps the 2×2 against NeMo
reference tensors, and the packaging script REFUSES to build a zip whose config does not match
the parity winner.

## Original blocker analysis (kept — it is why Option 2 was chosen)

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

## Pod plan (Option 2, single session; stop the moment the decision metrics exist)

Restart `1ppb7l0i5xuna8` (disk persists: Arm A ckpt, SAPC2 data, `nemoenv`). Upload the new
`track2_starting_kit/parakeet_onnx/` dir + the five `scripts/*parakeet_onnx*` files (get explicit
approval for the exact file list first — sandbox policy), then:

```bash
DATA=/workspace/... PARITY_WAVS='/workspace/data/parity/*.wav' \
  bash scripts/run_parakeet_onnx_pod.sh          # all stages, gated, in order
# or a subset:  STAGES="export parity" bash scripts/run_parakeet_onnx_pod.sh
```

Stage order and what each one settles:

1. **wheels** — `pip download onnxruntime` (cp311/manylinux) into the submission dir. Network is
   used HERE, on the pod, never at Codabench setup time.
2. **export** — `.nemo` → `weights/{encoder,decoder}_model.onnx` + tokens + NeMo's own filterbank/
   window + `streaming_meta.json`. The exporter asserts blank id against `decoder.blank_idx` and
   `joint.num_classes_with_blank`, and refuses any preprocessor `localmel.py` cannot reproduce.
3. **parity (fp32)** — the fidelity gate, and the only way `drop_policy`/`trim_policy` get set.
   Gate: encoder frame counts match NeMo on every step, max|Δ| ≤ 1e-2, transcripts ≥ 90% exact.
4. **int8** — dynamic int8 on the encoder only (fp32 decoder), the Nemotron recipe that scored.
5. **parity (int8)** — transcripts only (tensor tolerances are meaningless post-quantization).
   Gate: ≥ 85% exact vs NeMo.
6. **package** — zip with contents at ROOT; refuses to build if setup.sh touches the network, if
   `requirements.txt` would install anything, if external `.onnx.data` is missing, or if config.yaml
   disagrees with the parity winner.
7. **offline** — extract the zip, install into a fresh `venv --system-site-packages` with
   `--no-index` under DEAD PROXIES + `HF_HUB_OFFLINE=1`, and decode through the real
   `local_decode.py`. If it transcribes under those conditions it is offline-valid.
8. **gate_lat** — thread sweep 1/2/4 on the streaming pass. Pre-registered: take the LOWEST thread
   count whose mean(TTFT,TTLT) stays **≤ 420 ms**. If even threads=4 misses it, the corner is
   thread-bound — stop and report, do not raise the default blind. (Also gives the wall-clock
   budget check: projected Test wall **≤ 15000 s with ≥ 30% margin**.)
9. **gate_acc** — house rule `validate-against-real-harness`: real `local_decode.py` (both passes)
   → `evaluate.sh` → official sclite, on Dev_diag severe + Dev_clean2k, run **from the extracted
   zip**, never the source dir.

**Pre-registered ship criterion — written before the run, submit IFF all of these hold:**
- parity text stage ≥ 90% exact vs the NeMo wrapper (fp32) and ≥ 85% (int8)
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
