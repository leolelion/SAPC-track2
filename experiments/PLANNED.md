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

> **RETRACTED 2026-07-29.** The table below was NOT produced by the official scorer, despite saying so.
> It came from the error-analysis proxy scripts (single-ref, no min-over-two-refs, no `unk`
> reconciliation). Arm A's *same* hypothesis CSV rescored through `evaluate.sh` gives **CER 18.69% /
> WER 24.97%**, not 29.93%. Every derived quantity here — the +11.29-pt empty cost, the 21.00%
> non-empty CER, "already beats zipformer's 24.85%" — is therefore void. The empty *count* (48/425)
> is real; the CER arithmetic on top of it is not. Kept, struck, as the record of what we believed.
> See `EXPERIMENT_LOG.md` → `exp_armB_parakeet` for the control that caught it.

| | ~~value~~ (void) |
|---|---|
| ~~total CER~~ | ~~29.93%~~ → **18.69% official** |
| ~~non-empty CER (377 utts)~~ | ~~21.00%~~ |
| empties | **48/425 = 11.3%** (still valid — counted off the CSV) |
| ~~empty contribution to CER~~ | ~~+11.29 pts~~ |
| ~~CER if empties merely averaged~~ | ~~21.00%, beats zipformer's 24.85%~~ — **zipformer's 24.85% carries the same provenance risk and is being re-measured officially** |

Empty rate by etiology: ALS 24.5% · CP 10.5% · Down 7.5% · **Parkinson 0% · Stroke 0%**.
Empty slice shape: **22/48 are ≤3 words** (wake-words/commands), median duration 4.67 s, onset
(first 100 ms) median **−58.5 dBFS** on a loud body (full-utt RMS median −29.6 dBFS).

**Win condition:** severe CER **≤ 24%** while holding mean(TTFT,TTLT) **≤ 420 ms**.

> **ALREADY MET, and it was met before any of D1–D6 ran (2026-07-29).** Parakeet **Arm A**, the banked
> encoder-only model, on the official scorer: severe CER **18.69%** at mean(TTFT,TTLT) **356 ms**
> (TTFT p50 638 ms · TTLT p50 74 ms). It also beats the banked zipformer on both slices —
> severe 18.69% vs 22.48%, and Dev_clean2k (2000 utts / 122 speakers) **13.51% vs 18.19%**.
> We were spending GPU hours chasing a win condition a model on disk had already satisfied; we
> could not see it because the baseline we compared against was a proxy number 11 points too high.
> **The live question is no longer "how do we make parakeet good enough" — it is "package and ship
> Arm A", which is a Q decision, not a registry item.**

**The constraint that orders everything below:** the empty tail is an RNN-T confident-blank pathology
seated in the **joint network**, which Arm A left frozen. No data intervention can move a frozen
blank boundary. **D1 therefore gates D2–D6.**

> **D1 ANSWERED IT — the premise above is falsified (2026-07-29).** Unfreezing the joint for 4 epochs
> changed **45% of Dev_diag transcripts** yet moved official severe CER by **−0.05 pts the wrong way**
> (Arm A 18.69% → l0 18.74%) and moved empties **48 → 50**. A frozen blank boundary was not what was
> holding the empties: unfreezing it and training changed nothing that mattered. The confident-blank
> story does not survive its own control, so **D2–D5 no longer inherit it as a rationale** — they must
> justify themselves on generic robustness, at a much lower expected gain.

---

## Registry

Status vocabulary: `blocked` · `ready` (code exists, gate written, awaiting pod) · `running` ·
`done` · `killed`.

| ID | Name | Depends on | Cost | Status | Decision metric |
|---|---|---|---|---|---|
| **S0** | Official-scorer error decomposition | CPU pod (data lives there) | ~0, minutes | **ready** — `scripts/error_decomposition.py`, self-gated against the official metric classes | error mass in CER points by etiology / speaker / length / empty; char S/D/I |
| **S1** | Blank-penalty probe + sweep | S0 shares the session | ~4.5 h **CPU, no GPU** | **ready** — `investigations/step01_runbook.md` | probe GO/NO-GO first; then Dev_diag CER ≤ 18.43% ∧ Dev_clean2k ≤ +0.20 ∧ mean latency ≤ 420 ms |
| **D0** | Synthetic-data forensics | pod (data lives there) | ~0, CPU, minutes | **done** 2026-07-29 → `experiments/exp_d0_synth_forensics/NOTES.md` | kNN-VC: G-COVER PASS, G-PROV FAIL(100%) · G-EOS PASS · F5: UNMEASURED (no transcripts on pod) |
| **D1** | Arm B control — joint unfreeze + FastEmit | D0 not required | ~2.5 h GPU | **done 2026-07-29 — FALSIFIED** → `EXPERIMENT_LOG.md` `exp_armB_parakeet` | severe CER 18.69% → **18.74%** (worse), empties **48 → 50** |
| D6 | Short-command oversampling | folds into D1 | free | **deprioritized** — premise dented: ≤3-word utts are already 22.5% of train (74,606), not rare | A/B inside D1 |
| D2 | Arm B + kNN-VC subset | ~~D1 pass~~ | ~3 h GPU | **needs re-justification** — D1 did not pass; code ready (`scripts/run_d2_knnvc.sh`) but the rationale it inherited is void | severe CER, empty count |
| D3 | Arm B + F5-TTS subset | ~~D1 pass~~ | ~3 h GPU | **needs re-justification** + F5 text **not in repo or pod** (searched 2026-07-29) | severe CER, empty count |
| D4 | Sequential synth → real re-anneal | D2 or D3 partial | ~4 h GPU | **needs re-justification** | severe CER vs best of D2/D3 |
| D5 | TORGO + UASpeech targeted | ~~D1 pass~~ + license | ~3 h GPU | **needs re-justification** | severe CER, empty count |
| ~~D7~~ | F5 + kNN-VC mixed | — | — | **killed** | known EOS collapse; mechanism = our primary failure mode |

---

### S0 / S1 — measure with the real scorer, then pull the one lever that needs no GPU
**Added 2026-07-30, ahead of D2–D5.** Ranking is the CER × latency **Pareto frontier** (Q), and we
are already non-dominated on Test1 (19.01 / 416.9 ms) holding the low-latency corner. These two
target the CER axis without a GPU.

**S0.** Every error analysis we own came from proxy scripts; the load-bearing one was 11 CER points
wrong and justified D1–D6. `scripts/error_decomposition.py` reads the sclite SGML the official
scorer consumes and refuses to print unless it reproduces `CharErrorRateMinTwoRefs` /
`WordErrorRateMinTwoRefs` to 1e-6. Note official CER is **char-weighted micro**, so empties (median
≤3 words) cost ≈3.5–4.0 points, not the 11.3 the retracted table claimed.

**S1.** Official error profile is deletion-dominated (del:sub 4:1, **del:ins 18:1**) — the joint
over-emits blank. A blank-logit penalty was previously ruled out as blocked behind NeMo's
`greedy_batch` pinning; that stopped being true when we shipped ONNX, whose wrapper hand-rolls the
greedy loop. Ships `0.0` = byte-identical decode. It is the only lever that moves **both** Pareto
axes the same way (earlier emission also lowers TTFT).

**Probe gates the sweep.** `scripts/probe_blank_margin.py` measured blank margins of **6.7–15.9
(median 14.6)** on the shipped artifact — the originally planned 0–4 β grid was entirely dead and
would have produced a false negative. On-pod it also answers the go/no-go: if the empties' blank
margins are *higher* than ordinary blanks, no constant shift separates them and the decode-time line
closes in minutes rather than a session.

**Kill.** Probe NO-GO, or no β satisfies the pre-registered rule → keep `blank_penalty: 0.0`, and the
next spend is the GPU session (resume the un-converged Arm B curve × FastEmit λ sweep × top-k
checkpoint averaging × a speaker-disjoint val), not more decode-time tuning.

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

**Leakage filter, also not negotiable (D0, 2026-07-29).** Both corpora contain Dev-derived material —
kNN-VC 18,797 wavs / 41 Dev speakers, F5 11,931 wavs / 88 Dev-speaker buckets — plus a large `unknown`
class matching neither split (kNN-VC 40,339 · F5 23,373). Train/Dev speakers are disjoint, so filter
per file to **Train provenance only** and **exclude `unknown`** before any training use. Without this,
the Dev gate that authorizes shipping is contaminated by its own training data.
**How the cap is enforced (2026-07-29).** `nemo_finetune_v2.py` takes a single `--train-json`, and
`train_ds` shuffles, so the synthetic share of *steps* equals its share of *utterances* — not of hours.
`scripts/build_d2_mixed_manifest.py` therefore caps on utterance count and **hard-exits** if the cap is
exceeded, while separately reporting the duration share. Those two numbers diverge: kNN-VC's 5.44 s
median against SAP's shorter utterances means a 25% step share is ~33% duration share. Read both before
attributing any D2 delta to the data.

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
