# Runbook — Step 0 (error decomposition) + Step 1 (blank-penalty sweep)

**Date:** 2026-07-30 · **Status:** ready, awaiting a CPU pod · **Cost:** ~4–5 h CPU, **no GPU**
**Decision context:** the challenge ranks on the **CER × latency Pareto frontier** (Q, 2026-07-30).
We are already non-dominated on Test1 (CER 19.01 / mean latency 416.9 ms) and own the low-latency
corner. Nothing here is required to hold that position — it is an attempt to take the CER axis too.

---

## Why these two, and why before any GPU spend

**Step 0 — we have never done error analysis with the official scorer.** Every decomposition we own
came from proxy scripts, and the load-bearing one was **11 CER points wrong** (single-ref, no
min-over-two-refs, no `unk` reconciliation). It produced the "empties cost 11.29 points" claim that
justified D1–D6; D1 then spent GPU hours falsifying a theory built on a number that did not exist.
Before spending again we measure the error where the metric actually lives.

The correction matters because **official CER is char-weighted micro** — `sum(edit_distance) /
sum(ref_chars)`, per-utterance clipped at 1.0 (`utils/metrics/cer.py:_cer_update_min_two_refs`).
Empties are short (median ≤3 words for 22/48), so counting *utterances* massively overstates them.
Off the local artifacts, the 48 Dev_diag empties carry ≈1,043 ref chars against ≈26–30k total →
**≈3.5–4.0 CER points, not 11.3**. Step 0 replaces that estimate with the exact number.

**Step 1 — the deletion asymmetry has a decode-time knob we did not previously have.**
Measured error profile (Dev val-2000, `exp_armB_parakeet`): **deletions 24.76% · substitutions 5.87%
· insertions 1.38%** → del:sub 4:1, **del:ins 18:1**. A calibrated ASR has sub > del. Ours is
inverted: the joint over-emits blank.

The documented counter-pressure is a constant subtracted from the blank logit before the argmax.
`parakeet_improvement_framework.md` §7b ruled this out as "an integration task behind the greedy
pinning gate" — true of the **NeMo** path, whose `greedy_batch` exposes no per-step logit hook.
**That objection died when we shipped ONNX.** `track2_starting_kit/parakeet_onnx/model.py`
hand-rolls the RNN-T greedy loop (`_decode_frames`), so the hook is one line in code we own. Nobody
noticed the constraint had lifted.

Properties that make it the right first spend:
- **No training, no GPU.** Same weights, same graph, same wrapper — one env var.
- **Moves both Pareto axes the same direction.** Earlier emission lowers TTFT as well as CER.
  Every other lever in the portfolio trades latency for CER.
- **18:1 insertion headroom** before over-emission becomes the binding cost.
- Ships as `0.0` = byte-identical decode, so a failed sweep costs nothing but pod hours.

**A softmax temperature is NOT a second knob.** `argmax(z/T) == argmax(z)` for any `T > 0`. Under
greedy decoding an additive shift on one class is the only thing the argmax can see. β is complete.

---

## What changed in the repo (all local, already syntax-checked)

| file | change |
|---|---|
| `scripts/error_decomposition.py` | **new.** Step 0. Self-verifies against the official metric classes before printing anything. |
| `track2_starting_kit/parakeet_onnx/model.py` | `self._blank_penalty` (env `SAPC2_BLANK_PENALTY` > `config.decoding.blank_penalty` > `0.0`); applied in `_decode_frames`; echoed in the ready-line. |
| `track2_starting_kit/parakeet_onnx/config.yaml` | `decoding.blank_penalty: 0.0` + rationale. |
| `scripts/run_blank_penalty_sweep.sh` | **new.** The whole pod session, staged and gated. |

The repo copy of `model.py` was **md5-identical to the shipped zip** (`0c9a06b67ec0ba5e59c44c8b8320f25b`)
before the patch, so the patch applies to exactly what scored 19.01%.

At `blank_penalty = 0.0` the value is falsy, the branch is skipped, no array copy is made, and the
decode path is the scored one instruction-for-instruction.

---

## Verification already done locally (do not redo on the pod)

- `error_decomposition.py` self-test on a synthetic SGML pair exercising C/S/D/I, all three `unk`
  rules, an empty hypothesis and a ref1/ref2 disagreement: **gate A** (our SGML parse reproduces
  `utils/compute_metrics.py:parse_sgml_csdi` exactly) and **gate B** (our per-utterance sums
  reproduce `CharErrorRateMinTwoRefs` / `WordErrorRateMinTwoRefs` to 1e-6) both **PASS**, char ops
  sum exactly to the CER.
- Gate B **caught a real bug on first run**: sclite's own C/S/D/I ops are not minimal under
  torchmetrics' cost model (a synthetic pair scored S+D+I=3 by sclite has Levenshtein distance 2),
  so word ops read off the SGML do **not** reproduce the official WER. The script now recomputes
  both levels the way the metric does and keeps sclite's ops as a labelled secondary view.
- `model.py` compiles; `config.yaml` parses; the argmax flip behaves as intended across β.
- Sweep script passes `bash -n`; the β→filename→β round-trip is checked.

---

## Pod preflight (each of these has bitten us before)

1. **`evaluate.sh` placeholders.** The repo ships `DATA_ROOT` / `PROJ_ROOT` / `SCTK_DIR` as
   `/path/to/...`. Confirm the pod checkout is patched, or stage 2 scores nothing.
2. **Refs built.** The sweep runs `--start_stage 2`; `ref1.<split>.norm.trn` / `ref2.<split>.norm.trn`
   must already exist for `Dev_diag` and `Dev_clean2k`. If the disk was reset, run stage 1 once.
3. **Gate the runtime you ship.** `GATEPY` must be `$WORK/offlinevenv/bin/python` with
   `onnxruntime == 1.27.0`. The script refuses otherwise. This is the exact hole the 100%-CER int8
   submission walked through: five green gates on an interpreter the zip did not carry.
4. **Sweep the EXTRACTED zip**, not the build dir. The script checks `$EXTRACT/model.py` exists and
   contains the hook — a stale extract silently sweeps a wrapper with no knob and returns a flat curve.
5. **`SAPC2_THREADS=1`, one process per β.** `local_decode.py` is single-process (AudioSender +
   Decoder threads), so K betas cost K cores. 7 in parallel on 24 vCPU is safe.

---

## Commands

```bash
export REPO=/workspace/SAPC-template DATA=/workspace/data/SAP \
       WORK=/workspace/parakeet_onnx ART=/workspace/artifacts/blank_penalty

# --- grid: 7 betas x Dev_diag (425 utts), decodes in parallel ~45 min, then serial scoring
STAGES="decode_grid score_grid" bash scripts/run_blank_penalty_sweep.sh

# --- read the printed curve, pick FINAL = "0.0 <best> <neighbour>", then:
FINAL="0.0 1.0 1.5" STAGES="decode_final score_final latency" \
  bash scripts/run_blank_penalty_sweep.sh
```

Scoring is **serial on purpose**: `evaluate.sh` writes into the shared `$DATA/eval` tree, and stage 2
**overwrites** `$DATA/eval/sctk/<split>.ref[12].sgml` every run. The script copies each SGML aside
immediately and runs the decomposition before the next β touches it.

**Timing** (from the measured fp32 rate, 1,195 s / 200 utts at threads=1 ≈ 6 s/utt):
Dev_diag 425 utts ≈ 42 min per β, all β in parallel ≈ **45 min**. Dev_clean2k 2,000 utts ≈ 3.3 h per
β, 3 in parallel ≈ **3.3 h**. Streaming pass on Dev_streaming (123 utts) ≈ 15 min each. Total ≈ 4.5 h.

`β = 0.0` on Dev_clean2k is in the plan deliberately: fp32's Dev_clean2k was **never measured** (the
one criterion left argued-not-measured in `exp_parakeet_onnx_fp32`). This closes it for free.

---

## Pre-registered decision rule — written before the run, per house rule

Baseline is the **shipped artifact at β=0**: Dev_diag severe CER **18.733%**, mean(TTFT p50, TTLT
p50) **375.7 ms**, projected Test wall-clock **~4,620 s**.

**ADOPT a β IFF, through the organizers' `evaluate.sh` on both Dev slices:**

| | criterion | why this threshold |
|---|---|---|
| A1 | Dev_diag severe CER **≤ 18.43%** | ≥0.30 pt better. Dev→Test transfer was +0.28 pt last time; smaller is indistinguishable from transfer noise. |
| A2 | Dev_clean2k CER regresses **≤ 0.20 pt** vs the β=0 run **from the same session** | stops us tuning the severe tail at the broad slice's expense. |
| A3 | mean(TTFT p50, TTLT p50) **≤ 420 ms** | the corner is why parakeet exists. |
| A4 | projected Test wall-clock **≤ 7,500 s** | 50% margin. More emissions = more prediction-net calls per chunk. |
| A5 | the winning β's **two grid neighbours also beat baseline** | a lone spike on a 425-utterance slice is noise, not a mechanism. |

**OR**, as a pure corner extension (we are ranked on Pareto and own the low-latency corner):
mean latency improves by **≥50 ms** with Dev_diag CER within **+0.10 pt**, and A2/A4/A5 hold.

**Otherwise: keep `blank_penalty: 0.0`, report the curve, stop.** Never submit to test a hypothesis.

**Honest caveat on tuning against Dev.** `val.json` is Dev-drawn, so Arm A's checkpoint selection has
already seen Dev; tuning β on Dev adds to that. A5 is the mitigation — we accept a β only where the
CER-vs-β curve is smooth, which is what a real mechanism looks like, and we prefer the **smaller** β
on a plateau. This does not eliminate the bias; it bounds it. Say so in the log.

---

## Watch items (second-order effects, none of them assumed away)

1. **`<EOU>`.** The checkpoint is a joint ASR + `<EOU>` model and `<EOU>` is a real vocab token, only
   stripped on output. Penalising blank raises the chance `<EOU>` wins the argmax, consuming a symbol
   slot and advancing the prediction state. If CER degrades while the empty count *falls*, check the
   raw hypotheses for `<EOU>` inflation before blaming the mechanism.
2. **Wall clock.** Deletion-heavy decoding is *cheap*; fixing it costs prediction-net calls. A4 is a
   real gate, not a formality.
3. **Streaming pass ≠ accuracy pass.** If per-chunk compute crosses 100 ms the stream falls behind
   real time and TTLT regresses. This only shows up in the paced pass — hence stage `latency`.
4. **All-empty latency fallback.** `compute_latency.py` reports a fallback TTFT when everything is
   empty; the int8 catastrophe's "6,170 ms TTFT" was a decode failure wearing a latency number. Check
   `n_utts_with_timing == n_utts_total` in every latency log.
5. **β scale is unknown a priori.** These are unnormalised joint logits; the grid spans 0–4 to find
   the responsive region. If every β below 4 changes nothing, the blank margin is larger than the
   grid — extend upward rather than concluding the lever is dead. If β=0.5 already destroys accuracy,
   bisect downward.

---

## Stop condition

The moment the grid + confirmation table is written, **copy `$ART` back and stop the pod.** Decision
metrics are all in `decomp_*.json`, `eval_*.log`, `latency.*.log`. No further exploration in the same
session — that is how sessions turn into GPU bills.

Then: `experiments/summary.csv` row, `EXPERIMENT_LOG.md` entry, and update
`experiments/PLANNED.md` (this work is the step-0/step-1 pair, ahead of D2–D5).

---

## What this does NOT do

It does not retrain anything. If the sweep returns flat, the conclusion is *"the blank margin on this
checkpoint is not reachable by a constant shift"* — which is itself worth knowing, because it means
the deletion bias is distributional rather than a calibration offset, and the next spend is the GPU
session (resume the un-converged Arm B curve × FastEmit λ sweep × proper top-k checkpoint averaging
on a speaker-disjoint val), not more decode-time tuning.
