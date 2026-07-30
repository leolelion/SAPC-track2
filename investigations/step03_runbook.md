# Step 3 runbook — H-F speaker forensics + H-A lookahead re-export

Pre-registered **before** the pod starts (house rule: write the success criterion first).
Companion to `investigations/step01_runbook.md` (blank penalty) and `step02_runbook.md` (beam).

Driver: `scripts/run_ha_hf_pod.sh`. Gates are duplicated in its header so a reader on the pod
sees them without this file.

---

## What Phase 0 established locally, for $0

`scripts/analyze_rate_and_speakers.py` re-reads the banked `*.summary.json` from
`error_decomposition.py` (per-utterance `ref_chars` / `errors` / `duration` / `speaker`,
reference text stripped). It self-gates on reconstructing the official micro CER from those
records: **delta 0.0e+00** against the stored 18.7332%.

### 1. The rate axis is real, and smaller than it looked

| bucket | n | rate (ref char/s) | CER% | 95% CI | pts | ref% | err% | charD | empty |
|---|---|---|---|---|---|---|---|---|---|
| Q1 slowest | 107 | 0.9–3.0 | 39.70 | [33.4, 46.6] | 3.69 | 9.3 | 19.7 | 858 | 20 |
| Q2 | 106 | 3.1–4.8 | 31.03 | [24.0, 38.7] | 5.14 | 16.6 | 27.4 | 1155 | 15 |
| Q3 | 106 | 4.8–7.2 | 18.19 | [13.7, 23.5] | 5.22 | 28.7 | 27.9 | 1012 | 8 |
| Q4 fastest | 106 | 7.2–14.3 | 10.31 | [8.2, 12.7] | 4.69 | 45.4 | 25.0 | 755 | 5 |

The 3.9x difficulty gradient is confirmed. But **error mass is flat** across the quartiles, and
the slow quartile holds only **9.3% of reference characters**. Fixing Q1 perfectly is worth
**−3.69 points**; fixing it to Q4's level ≈ **−2.7**. Every rate-targeted lever (front-end time
compression, rate-conditioned decode) is capped there.

**74% of the reference characters are in Q3+Q4.** Reaching them is what H-A is for.

> This is the same trap the repo already fell into once, in the opposite direction: S0's "51% of
> error mass is 13+ word utterances" was a mass statistic read as a difficulty one. Rate is a
> difficulty statistic. Reading it as a mass statistic would be the mirror-image mistake.

### 2. Dev_diag's noise floor

Whole-set unpaired 95% CI is **[16.26, 21.63]** — ±2.7 points on n=425. Paired resolves ~±1.5.
Any historical verdict decided on a sub-point unpaired difference was inside noise (D1's
"−0.05 worse" is a null, not a falsification). **Every gate in this runbook is paired.**

### 3. The beam verdict does not survive stratification

Paired, `b4m2mean` vs greedy, on banked step02 artifacts:

| bucket | greedy | beam4+mean | delta | 95% CI |
|---|---|---|---|---|
| Q1 slowest | 39.70 | 34.05 | **−5.65** | [−8.85, −2.72] |
| Q2 | 31.03 | 27.87 | **−3.15** | [−5.79, −0.76] |
| Q3 | 18.19 | 22.01 | +3.82 | [+0.63, +7.45] |
| Q4 fastest | 10.31 | 13.58 | +3.27 | [+1.23, +5.61] |

The whole-set +1.52 was **two opposite effects cancelling**. Fast-quartile damage is significant
in **8/8** configs tested — solid. The slow-quartile benefit is significant in **1 of 4**
(`b4m2none` −3.04 marginal, `b2m2mean` −0.73 n.s., `b2m2none` +1.70 n.s.), and with 16 tests at
95% one false positive is expected. **Hypothesis, not fact.** Oracle value of perfect
rate-switching ≈ **−1.05 points**, and the switch would need an online rate estimator (emission
density per encoder frame is the candidate self-signal) — logged as H-G, not scheduled here.

### 4. Speaker `55c1784a` is not a severity case

CER **86.65%**, **21 of 28** utterances empty, non-empty CER still 38.86%, **14.1%** of all error
mass, leave-one-out shift **−2.14 points**. Other ALS speakers: **14.28%**, 5 empties in 82.

|  | n | dur med | rate med | ref words | empty |
|---|---|---|---|---|---|
| 55c1784a | 28 | 6.29 | 4.11 | 6.0 | 21 |
| all others | 397 | 7.00 | 4.92 | 6.0 | 27 |
| its empties | 21 | 6.38 | 4.21 | 6.0 | 21 |
| its non-empties | 7 | 5.91 | 4.01 | 5.0 | 0 |

Duration, rate and word count are **excluded** — its empties look like its non-empties and like
the rest of the corpus. The cause is in the waveform, which is why H-F needs the pod.

---

## H-F — acoustic forensics (stage `forensics`, ~3 min)

`scripts/probe_speaker_audio.py`. Per-utterance level / structure / channel descriptors from the
raw wav (stdlib `wave` + numpy, no new deps), then four contrasts with a rank-based separation
statistic (AUC; 0.5 = none, ≥0.80 or ≤0.20 = strong):

- **C1** target vs all other speakers
- **C2** target vs same-etiology speakers (controls for the disorder)
- **C3** empty vs non-empty corpus-wide (does the same descriptor drive the other empties?)
- **C4** the target's empties vs its own non-empties (is there a per-utterance trigger?)

**No pass/fail gate — this is forensics.** Stop condition: report written. Decision tree:

| result | reading | next |
|---|---|---|
| C1+C2 strong, C3 flat | that speaker's recording session | data-side; will not generalise to Test |
| C1+C2 strong, C3 strong | front-end gap that also drives the corpus empties | highest value: one fix, two problems |
| C1+C2 flat | audio is in-family; the model fails this talker | no front-end fix exists; training only |
| C4 strong | per-utterance trigger, not per-speaker | a decode/front-end condition is available |

Smoke-tested locally against a synthetic fixture with a known injected low-SNR signature:
recovered it at AUC 0.00 in C1/C2/C3 and correctly reported **NONE** on the null contrast C4.

> Prior result that this does **not** revive: the frozen-scalar `input_gain` lever was already
> falsified against the empties (0/48 recovered, official CER worse at 19.22% vs 18.69%). That
> tested a **level** fix. It did not test SNR or channel. A level finding here is not a reason to
> re-enable `input_gain`.

---

## H-A — lookahead re-export

Shipped runs `att_context_size=[70,1]` = 80 ms = **one encoder frame** of future. Precedent in the
same family (`exp_sweep_fc114_*`): FastConformer-114M zero-shot **54.77 → 33.75** CER moving
[70,1] → [70,16]. Deletions here are whole words at every rate, positionally flat — the signature
of a per-frame context deficit, and positional flatness does **not** bear against right-context
(a lookahead limit is uniform in position; it was *left*-context exhaustion that flatness killed).

### G-HA0 — screen, kill-only (stage `probe`, ~8 min)

`scripts/probe_lookahead_support.py` answers two separate questions:

- **structural**: was the encoder *trained* multi-context (`att_context_size` list-of-lists,
  `att_context_probs`, `att_context_style`)? `set_default_att_context_size()` accepts any context
  on a cache-aware encoder — that is not evidence the weights support it. And our finetune pinned
  a **single** context (`nemo_finetune_v2.py --train-ctx "[70,1]"`), so even a multi-context base
  may have had its other contexts degraded by our own training.
- **functional**: a 30-utterance offline transcribe at each context, punct-stripped single-ref CER.

**Rule: within +3.0 proxy points of [70,1] to survive.** The screen can only **veto**. This repo has
an 11-point proxy error on its record (the retraction in `experiments/PLANNED.md`); a good proxy
number here buys nothing but permission to spend the real gate.

**Miss → H-A is a GPU arm** (retrain pinned at the new context). Stop the pod; do not export.

### G-HA1 — ONNX parity at the new context (stage `parity`, ~5 min/context)

≥90% exact match vs the NeMo wrapper **set to the same context**, ≥20 utterances. The NeMo side is
a patched *copy* of `parakeet_realtime_ft`; the pristine dir is never edited. This is the stage
that caught four export/wrapper bugs in the ONNX ship. `drop_policy`/`trim_policy`/`first_step_pad`
were verified at [70,1] **only**, so the export stage prints the new geometry (`chunk_size`,
`pre_encode_cache_size`, `valid_out_len`, `drop_extra_pre_encoded`) to make a policy change visible
rather than silent. Miss → the export is wrong, not the model.

### G-HA2 — accuracy (stages `decode` + `score`, ~65 min/context decode)

Official scorer, Dev_diag severe, **paired** vs the banked greedy baseline 18.733%:

> **delta CER ≤ −1.00 points AND the 95% paired CI excludes 0.**

1.00 matches the B2a threshold from the beam grid: the point must clearly buy the latency it costs.

### G-HA3 — on-target

> **The gain must appear in Q3+Q4, not only Q1+Q2.**

H-A's entire premise is reaching the 74% of characters that rate-based levers cannot. A gain
confined to the slow quartiles means H-A duplicates the cheaper front-end lever and should not be
paid for in latency.

### G-HA4 — Pareto (stage `latency`, winner only)

Shipped: Test1 **19.01% @ 416.9 ms** mean(TTFT,TTLT). Dev→Test CER transfer measured **+0.28**.
Live frontier neighbour: yac3xn **18.10% @ 592 ms**.

| context | lookahead | projected Test mean | needs |
|---|---|---|---|
| **[70,3]** — primary | 240 ms | ~537 ms | Dev_diag ≤ 17.8 → **dominates yac3xn on both axes** |
| [70,6] — exploratory | 480 ms | ~630 ms | Dev_diag ≤ 17.8 **and** must beat 18.10, else dominated |

Nothing is submitted from this session. Any ship still needs Dev_clean2k plus the full
faithful-validation gate (`validate-against-real-harness`).

---

## Session order and stop conditions

1. `STAGES="forensics probe"` — H-F first: minutes, independent, and if the session dies we still
   hold 2.14 points' worth of diagnosis.
2. Read G-HA0. **Kill → stop the pod.**
3. `STAGES="export parity decode score"` with only the surviving contexts.
4. Read G-HA2/G-HA3. **Either misses → stop the pod and report.**
5. `STAGES="latency"` for a survivor only.
6. Copy `$ART` back, **stop the pod immediately**.

Budget: two contexts end-to-end ≈ 1 h 45 m ≈ **$5.50** on the $2.99/h pod.

## Files

- `scripts/analyze_rate_and_speakers.py` — Phase 0 analyzer (local, self-gated)
- `scripts/probe_speaker_audio.py` — H-F forensics
- `scripts/probe_lookahead_support.py` — H-A structural + functional screen
- `scripts/run_ha_hf_pod.sh` — session driver, gates in the header
- `investigations/step03_runbook.md` — this file

Nothing in `utils/`, `steps/eval/`, `evaluate.sh` or `local_decode.py` is touched (house rule:
wrap, never modify).
