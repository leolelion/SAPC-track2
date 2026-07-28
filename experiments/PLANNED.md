# experiments/PLANNED.md — forward experiment registry

Companion to `summary.csv` (results, past) and `EXPERIMENT_LOG.md` (narrative). **This file is the
only place a not-yet-run experiment lives.** An experiment moves out of here the moment it produces a
number: create `exp_<id>/`, add a `summary.csv` row, write the `EXPERIMENT_LOG.md` entry, then set the
row below to `done` with a pointer. Never delete a row — killed experiments stay, marked `killed`,
with the reason. That is the record of what we ruled out.

**House rules that bind every row:** pre-register the success criterion *before* the run · proxy
scorer never signs off a submission (`validate-against-real-harness`) · stop the pod the moment the
decision metric is known.

---

## The thing all of this is aimed at

Parakeet Arm A on Dev_diag severe (n=425, official scorer):

| | value |
|---|---|
| total CER | **29.93%** |
| non-empty CER (377 utts) | 21.00% |
| empties | **48/425 = 11.3%** |
| **empty contribution to CER** | **+11.29 pts** |
| CER if empties merely averaged | 21.00% — **already beats zipformer's 24.85%** |

Empty rate by etiology: ALS 24.5% · CP 10.5% · Down 7.5% · **Parkinson 0% · Stroke 0%**.
Empty slice shape: **22/48 are ≤3 words** (wake-words/commands), median duration 4.67 s, onset
(first 100 ms) median **−58.5 dBFS** on a loud body (full-utt RMS median −29.6 dBFS).

**Win condition:** severe CER **≤ 24%** while holding mean(TTFT,TTLT) **≤ 420 ms**.

**The constraint that orders everything below:** the empty tail is an RNN-T confident-blank pathology
seated in the **joint network**, which Arm A left frozen. No data intervention can move a frozen
blank boundary. **D1 therefore gates D2–D6.**

---

## Registry

Status vocabulary: `blocked` · `ready` (code exists, gate written, awaiting pod) · `running` ·
`done` · `killed`.

| ID | Name | Depends on | Cost | Status | Decision metric |
|---|---|---|---|---|---|
| **D0** | Synthetic-data forensics | pod (data lives there) | ~0, CPU, minutes | **ready** | 3 gates below |
| **D1** | Arm B control — joint unfreeze + FastEmit | D0 not required | ~2.5 h GPU | **ready** | val CER + val empty count |
| D6 | Short-command oversampling | folds into D1 | free | ready | A/B inside D1 |
| D2 | Arm B + kNN-VC subset | D0 pass, D1 pass | ~3 h GPU | blocked | severe CER, empty count |
| D3 | Arm B + F5-TTS subset | D0 pass, D1 pass | ~3 h GPU | blocked | severe CER, empty count |
| D4 | Sequential synth → real re-anneal | D2 or D3 partial | ~4 h GPU | blocked | severe CER vs best of D2/D3 |
| D5 | TORGO + UASpeech targeted | D1 pass + license | ~3 h GPU | blocked | severe CER, empty count |
| ~~D7~~ | F5 + kNN-VC mixed | — | — | **killed** | known EOS collapse; mechanism = our primary failure mode |

---

### D0 — Synthetic-data forensics
**Question.** Do the two synthetic corpora actually contain the failure region, and do they carry new
information? Everything downstream is a coin-flip until this is answered, and it costs no GPU.

**Runs.** `scripts/d0_synth_forensics.py` on the pod. Runbook: `investigations/d0_runbook.md`.
**Inputs.** `/workspace/data/processed/SAPC2_v3_synth` (F5-TTS, 139,500 wav / 217.4 h) ·
`/workspace/data/processed/SAPC2_v3_knnvc` (kNN-VC, 203,427 wav / 150 h) · SAP `Train.csv` ·
`experiments/exp_parakeet_ft_empties/empty_slice_refs.json`.

**Pre-registered gates** (write the verdict against these, not against vibes):

| Gate | Measure | Pass | Meaning if FAIL |
|---|---|---|---|
| **G-COVER** | fraction of synth utts ≤3 words, and onset-dBFS p25 | ≥5% short **and** onset p25 ≤ −45 dBFS | corpus lacks the failure region; D2/D3 demote from "empty fix" to generic robustness, expected gain drops to ≈0 on the 11.3-pt lever |
| **G-PROV** | exact-text match rate of synth transcripts vs SAP Train | <80% | synth is a re-weighting of SAP, not new data; cap mix fraction hard, expect small gains |
| **G-EOS** | \|median trailing-silence F5 − kNN-VC\| | <300 ms | large gap is the mechanical candidate for the observed EOS collapse; fixable by trailing-silence normalization, and the D7 mixing ban could then be revisited |

**Stop condition.** Script writes `d0_forensics.json` and prints the three verdicts. Copy back, stop
reading. If Q is not also running D1 in the same session, stop the pod.

---

### D1 — Arm B control (joint unfreeze + FastEmit)
**The gate for the whole program.** If unfreezing the joint does not collapse the empty rate, the
confident-blank theory is wrong and the parakeet CER line closes — no amount of added data rescues it.

Fully specified already: `investigations/arm_b_runbook.md`. Trainer exists:
`scripts/nemo_finetune_v2.py --freeze joint_unfreeze --fastemit-lambda ...`.
Ladder: (1) λ=0 isolates the unfreeze · (2) λ=0.005 primary bet · (3) 0.01, back off to 0.003 if
insertions rise. Stop at the first rung that clears.

**Pre-registered.** GATE-TRAIN: val CER beats Arm A's 27.5% **and** val empty count drops.
GATE-SHIP: real harness severe CER ≤24% **and** mean latency ≤420 ms.
**Kill.** Empties stay above ~20% of the 48-slice **and** non-empty CER worsens → theory wrong,
parakeet becomes latency-only, ship the zipformer.

**Watch.** Insertion rate, not just CER — FastEmit's failure mode is over-emission (regresses at
λ ≥ 0.02, arXiv 2010.11148 Table 2). And the Nemotron collapse scar: graded unfreeze, differential LR
(joint = encoder × 0.1), warmup→cosine not Noam, top-5 CER-checkpoint averaging.

---

### D6 — Short-command oversampling *(folds into D1, no separate pod)*
RNN-T predictors over-delete rare short phrases (internal-LM bias, arXiv 2108.10752); 22/48 of our
empties are ≤3 words. Implemented as an opt-in manifest pre-pass:
`prep_nemo_manifest_v2.py --oversample-short-words N --oversample-short-mult K` (default **off**,
so it is a no-op unless requested). Runs as an A/B arm inside the D1 session.
**Falsifier.** D1 + short-oversample beats D1 alone on the 48-empty slice.

---

### D2 / D3 — Arm B + one synthetic corpus
**Never both in one epoch** (D7 is killed). kNN-VC first: stronger literature prior (VC augmentation
beats speed/tempo perturbation, arXiv 2505.14874 / 2506.19823) and its ~2.7 s mean duration sits
closer to the short-utterance failure slice than F5's ~5.6 s.

**Scoping rule, not negotiable.** Combined synthetic (343k utts) ≈ doubles SAP's 331k. Subsample to
~60 h, ALS/Down-weighted, capped at **≤25% of training steps**, and finish on real data.
**Falsifier.** Arm B + synth beats Arm B alone on the 48-empty slice *and* does not raise non-empty CER.
**Kill.** Non-empty CER degrades more than the empty slice improves → distribution drift dominates.

---

### D4 — Sequential, real-last
Q's Experiment C, with the order corrected: train on synthetic, then **re-anneal on real SAP**.
Finishing on the true distribution is what undoes synthetic drift; finishing on synthetic bakes it in.
Only worth running if D2 and D3 split (one helps, one hurts) or if both are partial.

---

### D5 — TORGO + UASpeech
Highest expected value per hour of any data branch, and the only one with a compliance kill switch.

Why it fits despite the usual criticism: the standard knock on these corpora is that they are
isolated-word and unnatural. **Our failing slice is short commands and wake-words from low-energy
speakers.** UASpeech is a dysarthric short-command corpus (~103 h, 15 speakers, mostly CP); TORGO
(~23 h, 8 dysarthric) carries **ALS**, our worst etiology at 24.5% empty. Combined ~126 h — small
enough to oversample without swamping SAP.

**Rules: RESOLVED 2026-07-28 — external data is ALLOWED.** Organizers' FAQ, verbatim:

> *Q: Can I use extra data?* **A: Yes.** The distributed train and dev corpora were recorded using the
> same protocol as the test corpora, and are therefore expected to be better-matched to the test data
> than any other source. Competing teams may find, however, that system performance benefits from the
> use of other datasets; if so, teams are encouraged to describe the other datasets used, and the
> method of their use.

Two riders that bind D2–D5, not just D5:
- **"Better-matched than any other source"** is the organizers stating the domain-mismatch risk
  outright. It ratifies the ≤25%-of-steps cap on any non-SAP data and the real-last ordering (D4).
  SAP stays the primary distribution; everything else is a supplement.
- **Describe what we used.** Any shipped submission using TORGO/UASpeech/synthetic owes a written
  description of the dataset and the method. Keep `EXPERIMENT_LOG.md` submission-grade.

**Remaining blocker (one, and it is soft):** both corpora require signed licenses. Neither is a
`wget`. Start the request early — it is the long pole, and it runs in parallel with D0/D1 for free.

Use as **fine-tuning supplement into the joint** — not pretraining, not evaluation.

---

## Retired / rejected (kept so we do not re-derive them)
- **D7 — F5 + kNN-VC mixed.** Killed on Q's prior observation: EOS collapse. Our checkpoint is a joint
  ASR + `<EOU>` model, so early-`<EOU>` truncation is a blank-side failure — the same pathology D1
  exists to fix. Do not feed it. G-EOS in D0 tests the mechanical explanation.
- **Zipformer data work.** Banked model is Pareto-dead on the latency axis (beam-8 improves CER but
  loses the axis we already lose). Spend here buys no rank.
- **Nemotron, FastConformer-32M, zipformer-20M.** Closed on evidence; see `summary.csv`.
