# Step 2 runbook — localise the deletion mass, then try a beam

Status: **prepared locally, NOT run.** Written 2026-07-30 after `exp_s0_s1_probe`.
Global protocol: `/Users/o/Downloads/CLAUDE.md`. Repo contracts: `SAPC-template/CLAUDE.md`.

---

## 1. Why this exists

`exp_s0_s1_probe` was the first error decomposition ever run through the official scorer.
It moved the target:

| | |
|---|---|
| char deletions | **3780 = 13.04 CER pts** |
| char substitutions | 1294 = 4.46 pts |
| char insertions | 401 = 1.38 pts |
| empties | 48 utts = **3.89 pts only** |
| 13+ word utterances | 99 utts = **51.0% of all error mass**, 2 empty |

Dividing the two levels against each other is the finding that reframes the programme:

```
3780 char deletions / 760 word deletions = 4.97 chars per deleted word
28986 ref chars      / ~5900 ref words   = 4.91 chars per word
```

They match. Insertions match too (401/87 = 4.6). **The deletions are whole words.** The
words the model does emit are ~2 chars off — phonetic neighbours, not garbage. So this is
a **coverage** failure, not a phonetic one, and it lives inside long non-empty output.

The empty tail — which D1, `input_gain` and the blank penalty were all aimed at — is
3.89 points and all three attacks failed on it. It should stop being the organising
principle. See memory `empty-tail-is-minor`.

## 2. Two mechanisms killed locally, for free, before proposing anything

| candidate | verdict |
|---|---|
| `max_symbols_per_step` capping emissions | `null` -> checkpoint value or 10. Slow dysarthric speech has a LOW symbol/frame rate. Not binding. **Dead.** |
| `input_finished` dropping the final remainder | Real: `if (total - self._feat_idx) >= samp` silently discards a short tail. But `samp` is subsampling-scale (~8 mel frames), so the loss is bounded at **~80 ms**. Cannot be 760 words. **Footnote, not cause.** |

## 3. The three live mechanisms, and the measurement that separates them

| | mechanism | position signature | run signature | fix |
|---|---|---|---|---|
| **M1** | duration mismatch — dysarthric speech is 2–4x slower, encoder left context is fixed in FRAMES, so it spans a fraction of the usual linguistic history | rising | either | longer left context **at export**, no retraining, no added lookahead |
| **M2** | blank prior mismatch — many frames per word, joint defaults to blank (blank won 92.86% of probe steps) | flat | singletons | beam search / training |
| **M3** | prediction-net drift — one drop puts the LM state off-distribution, causing more | rising | long runs | training only |

`scripts/error_decomposition.py` now reports all three signatures from alignments we
**already have**. No decode, no GPU, no new data:

- **S2a position** — normalised reference index of every deleted word, non-empty
  utterances with >= 4 reference words (empties delete everything by construction and
  would flatten the histogram), plus the 13+ bucket alone.
- **S2b runs** — consecutive-deletion run lengths, and the share of deleted words sitting
  in runs of 3+.
- **S2c rate** — Spearman of per-utterance CER against duration, speaking rate
  (ref chars/s) and word count; quartile tables for each; and a rate split **within** the
  13+ bucket, which holds utterance length roughly fixed so rate has to do its own work.

## 4. The lever: RNN-T beam search

Greedy takes blank at a frame and can never revisit it. A beam keeps the emitting
continuation alive so a later frame can pay for it. That is the mechanically correct
form of the counter-pressure the blank penalty applies by brute force — and the S1 probe
already showed a constant shift cannot separate the populations it would need to
(empty p10 margin 6.35 vs non-empty 2.61).

Implementation: `track2_starting_kit/parakeet_onnx/model.py`, time-synchronous decoding.
Per encoder frame, each live hypothesis is expanded; its BLANK continuation retires to the
finished set (blank is what advances time in RNN-T, and the prediction state does *not*
advance on a blank); its top non-blank continuations advance the state and stay live.
Depth `beam_max_symbols` is a **blank-only pass**, so every hypothesis that retires has
paid for its blank and carries a valid path log-prob — no hypothesis crosses a frame
boundary on an unpriced alignment.

`beam_length_norm: mean` is the second half of the lever and is on-target by construction:
an RNN-T beam is biased toward SHORT output because each extra token multiplies in another
probability < 1, and short output is exactly our failure mode.

**Cost:** up to `(beam_max_symbols + 1) * beam` decoder runs per frame. The encoder is
untouched and dominates wall clock, so the end-to-end multiplier is far below `beam` —
but the 15000 s budget already cost us one submission to a timeout, so it is measured,
never inferred.

### Repo changes

| file | change |
|---|---|
| `track2_starting_kit/parakeet_onnx/model.py` | `_BeamHyp`, `_log_softmax`, `_prune`, `_decode_frames_beam`; dispatch in `_stream_step`; beam state in `reset()`; knobs in `_init_runtime_cfg`; ready-line echoes them |
| `track2_starting_kit/parakeet_onnx/config.yaml` | `beam: 1`, `beam_max_symbols`, `beam_expand`, `beam_prune_logp`, `beam_length_norm` + rationale |
| `scripts/error_decomposition.py` | `_dp_table` extracted, `seq_align_trace`, `deletion_profile`, `spearman`, `report_deletion_localization`; per-utterance `del_positions`/`del_runs`/`duration_f` |
| `scripts/test_beam_decode.py` | NEW — beam correctness, no weights needed |
| `scripts/test_deletion_localization.py` | NEW — localization correctness, no SGML needed |
| `scripts/run_deletion_beam.sh` | NEW — staged pod driver |

**At `beam: 1` nothing is constructed and `_stream_step` dispatches to the untouched
greedy path.** Stage `parity` proves that on the pod against the banked shipped decode.

## 5. Local verification record (all actually executed 2026-07-30)

```
python3 -m py_compile track2_starting_kit/parakeet_onnx/model.py      -> compile OK
python3 scripts/test_beam_decode.py                                   -> ALL 4 PASS
python3 scripts/test_deletion_localization.py                         -> ALL 4 PASS
bash -n scripts/run_deletion_beam.sh                                  -> syntax OK
```

`test_beam_decode.py` drives both decode paths over a fake decoder session returning
scripted logits:

1. **peaked joint** — beam-3 reproduces greedy exactly (`'a b c'`). Regression guard: bad
   state threading or blank bookkeeping diverges here even with no search to do.
2. **the deletion case** — blank beats the token by 0.1 at frame 0, so greedy commits and
   outputs `''`; the emitting path is rewarded at frame 1 by more than it gave up, and
   beam-2 recovers `'a b'`. This is the mechanism, in miniature.
3. **emission budget** — `beam_max_symbols` 1 vs 2 yields 1 vs 2 words.
4. **length norm** — two paths scored -0.4870 (len 1) and -0.8870 (len 2); raw picks the
   short one, `mean` picks the long one.

> Test 3 failed on the first run (`max_symbols=1 -> ''`). Cause was the **test table**, not
> the beam: every state had the same blank logit, and since every path ends a frame with
> exactly one blank, all scores tied and `max()` returned the first. Fixed by varying blank
> cost by state. Recorded because a tie that looks like a result is exactly the kind of
> thing that ships a wrong number.

> Test 4 first passed **vacuously** (`''` vs `''`, asserted only `>=`). Rewritten with
> hand-computed scores and exact assertions.

`test_deletion_localization.py` covers the risky part — `seq_align` is what Gate B uses to
reproduce the official CER/WER, and it now shares `_dp_table` with the new trace:

1. 400 random pairs: trace op counts equal `seq_align`'s, and the non-insertion ops
   consume reference indices `0..len(ref)-1` exactly once each, in order.
2. hand-checked profiles: tail deletion -> one run of 3 at positions 0.5/0.7/0.9;
   scattered -> two singletons; empty hypothesis -> whole reference as one run.
3. planted signal: a late-deleting population gives `[0, 0, 9, 36, 63]` and a uniform one
   `[33, 18, 30, 27, 15]` — **the report can see the difference it is being asked to
   decide.**
4. `spearman` endpoints, zero-variance and short-input guards.

Not verified locally, by construction: anything needing SAP audio, the ONNX weights, or
`torchmetrics`. Those are what the pod is for.

## 6. Pod preflight

Verified on pod `1ppb7l0i5xuna8` last session (disk persists across stop):

| | path |
|---|---|
| repo worktree | `/workspace/SAPC-step01` on `fork/parakeet-onnx-ship` |
| data root | `/workspace/SAPC2` |
| shipped zip | `/workspace/onnx_ship/art32/` |
| extracted artifact | `/workspace/onnx_ship/extract32` |
| offline venv (ort 1.27.0) | `/workspace/onnx_ship/offvenv32/bin/python` |
| banked hypotheses | `/workspace/onnx_ship/art32/Dev_diag.fp32.predict.csv` |

Checklist:

1. `git -C /workspace/SAPC-step01 pull` — must land the commit carrying this runbook.
2. `chmod +x evaluate.sh preprocess.sh steps/eval/*.sh` — a worktree checkout loses the
   executable bit.
3. **Copy the patched wrapper into the artifact:**
   `cp /workspace/SAPC-step01/track2_starting_kit/parakeet_onnx/{model.py,config.yaml} /workspace/onnx_ship/extract32/`
   The extract is the *shipped* tree and has no beam hook; the driver's preflight refuses
   to run the decode stages without it.
4. Do **not** use `/workspace/SAPC-template` — it is a divergent lineage (`main` @ c48d945
   with a modified 3-arg `utils/metrics/wer.py`). Gate B is only meaningful against the
   scoring chain verified locally.
5. `DECOMPPY` must be the interpreter with `torchmetrics` (Gate B imports it). `python3`
   was correct last session.

## 7. Commands

```bash
# --- phase 1: free measurement + the inertness proof (~15 min, ~$0.75) ---
STAGES="decomp_banked parity" \
REPO=/workspace/SAPC-step01 DATA=/workspace/SAPC2 \
EXTRACT=/workspace/onnx_ship/extract32 \
GATEPY=/workspace/onnx_ship/offvenv32/bin/python \
ART=/workspace/artifacts/step02 \
BANKED_CSV=/workspace/onnx_ship/art32/Dev_diag.fp32.predict.csv \
bash scripts/run_deletion_beam.sh

# --- STOP. Read section 8. Only then: ---
STAGES="decode_grid score_grid" ... bash scripts/run_deletion_beam.sh
# --- and only for a config that cleared the grid: ---
STAGES="decode_final score_final latency" FINAL="1:2:none <winner>" ... bash scripts/run_deletion_beam.sh
```

## 8. How to read S2a — decided BEFORE the numbers exist

| position | runs | reading | what to do |
|---|---|---|---|
| flat | singletons | **M2**, per-frame blank prior | beam is aimed correctly — **run the grid** |
| rising | singletons | M1 (left context) with a per-word prior | run the grid, AND queue the longer-left-context re-export, which is cheaper |
| rising | long runs | **M3 / M1** — within-utterance collapse | a beam does not fix state drift. **Do not run the grid.** Report and take it to the GPU session |
| flat | long runs | bursty dropouts uncorrelated with position | ambiguous — report, do not self-authorize the grid |

Supporting evidence for M1 in the same output: `cer_vs_rate_cps` clearly stronger than
`cer_vs_ref_words`, and a slow/fast gap **inside** the 13+ bucket.

## 9. Pre-registered decision rule

Lives in the driver's header and is repeated here verbatim so it cannot be quietly
renegotiated after the numbers land. Baseline = the shipped artifact at beam=1: Dev_diag
severe CER **18.733%**, mean(TTFT p50, TTLT p50) **375.7 ms** on Dev, **~4620 s** projected
Test wall-clock.

**HARD DISQUALIFIER, any branch:** projected Test wall-clock > 12000 s.

| | B1 — replace the shipped point |
|---|---|
| B1a | Dev_diag severe CER <= **18.43%** (>= 0.30 pt; Dev->Test transfer was +0.28 pt) |
| B1b | Dev_clean2k CER regresses <= 0.20 pt vs the beam=1 run **in the same session** |
| B1c | mean(TTFT p50, TTLT p50) <= **420 ms** |
| B1d | projected Test wall-clock <= **7500 s** |
| B1e | plateau: the next-smaller beam also beats baseline. A lone spike is noise |

| | B2 — extend the frontier with a SECOND submission |
|---|---|
| B2a | Dev_diag severe CER <= **17.73%** (>= 1.00 pt — it must clearly buy its latency) |
| B2b | mean(TTFT p50, TTLT p50) <= **1000 ms** |
| B2c | B1b and the hard disqualifier hold |

B2 exists because the challenge ranks on the **Pareto frontier**: a higher-latency,
lower-CER point is worth owning even when it does not dominate the greedy one. B2a/B2b are
a **judgement call, not a measurement** — they state what latency we are willing to pay for
a CER point. Q owns those numbers and should change them before the run, not after.

Otherwise: keep beam=1, report the curve, submit nothing.

**Mechanism check, independent of the adopt rule:** a beam that is working must move
`dDel` strongly negative. If CER improves mainly by shedding insertions or empties, the
win is real but the explanation is not ours — say so rather than banking a right answer
for a wrong reason.

## 10. Watch items

1. **Parity must pass first.** If beam=1 is not byte-identical to the banked decode, the
   control is not the shipped model and every delta below is meaningless. Stop there.
2. **Grid wall times are parallel** — comparable to each other, not a clean projection.
   The `latency` stage re-runs serially; only that number goes in a decision.
3. **Chunks over 100 ms** in the streaming log mean the stream has fallen behind real time.
   TTLT then degrades for a reason the beam cannot fix, and the accuracy pass will not
   show it.
4. `n_utts_with_timing == n_utts_total` in each latency log. An all-empty TTFT fallback
   looks like a latency number and is a decode failure (the int8 lesson).
5. **Partials can retract** under a beam — the best hypothesis can get shorter. Contract-
   legal and scored on the final string, but if TTFT moves oddly, this is why.
6. `beam_expand` defaults to `beam`. Widening it costs no extra decoder runs (one run
   returns the whole vocab row); if the grid looks starved, widen it before widening `beam`.

## 11. Stop condition

Stop the pod as soon as one of:

- S2a reads **M3 / rising + long runs** — the beam is the wrong lever; do not spend the grid.
- The grid is scored and no config clears B1 or B2.
- A config clears the rule and its Dev_clean2k + latency confirmation is banked.

Then `runpodctl stop pod <id>`, verify `EXITED`, copy `$ART` back, strip licensed
reference/hypothesis text before anything is committed to the public fork, and write the
`EXPERIMENT_LOG.md` entry with the numbers that were actually observed.

Budget: phase 1 ~15 min. Full grid + confirmation ~3 h ≈ $9 at $2.99/h. The phase-1
decision point is deliberately at ~$0.75.
