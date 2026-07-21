# 42 — Fugu independent research review (2026-07-20)

> Independent audit by Fugu (autonomous research engineer). Starting point: `RESEARCH_SUMMARY_FOR_FABLE.md`.
> Every metric below was cross-checked against the primary docs (`research/09/33/40/41`,
> `investigations/*`, `experiments/summary.csv`, `track2_starting_kit/streaming_zipformer/{config.yaml,model.py}`).
> Follows the repo's faithful-harness discipline: no claim of a repro I did not run; this box is `darwin`
> (Linux-only wheels), so anything that "runs" is a gated pod session, not executed here.

---

## Executive Summary

The project is in a **healthy but lopsided** state. One deployable model — the SAP-finetuned streaming
**Zipformer with modified_beam_search paths=4** — is live and measured on Test1 (**CER 21.28%**, TTFT
1365 ms, TTLT 93 ms), and it Pareto-dominates the earlier greedy submission on both axes. That is a real,
verified #1. The competing bet, **Nemotron 0.6B (10× params)**, has been exhaustively and correctly
falsified as a rescue target: three independent adaptation arms failed the faithful gate, and the binding
cause is now understood (H4 domain shift + H7 offline→streaming-export collapse + poor Dev→Test transfer).
The decision to **stop Nemotron and bank the zipformer** is evidence-backed and I endorse it.

The lopsidedness: **essentially all training/adaptation effort went into the model that loses (Nemotron),
and almost none into the model that wins (zipformer).** The zipformer got beam-4 (a decode-time change) and
nothing else. The two highest-EV, evidence-backed, never-run levers — **RNN-LM shallow fusion** (research/09
"next") and a **severity-aware zipformer retrain** (research/33's recipe, but applied to the winner instead
of the loser) — both sit unexplored. Separately, the **latency axis is barely touched** despite being half
the scored objective and nearly free to sweep (`config.yaml` edit, no retrain). Given the win condition is a
**CER×latency Pareto frontier** (prize split among all frontier teams), the latency corner is cheap insurance
that the docs under-weight.

**One-line recommendation:** stop treating "beam-4 zipformer" as the finish line. It is the *baseline* for
three cheap, high-EV pushes on the model that actually transfers — LM fusion (CER), a latency Pareto sweep
(the other scored axis), and one severity-aware retrain (the severe tail) — in that order.

---

## Repository Validation

Claims in `RESEARCH_SUMMARY_FOR_FABLE.md` I checked against source, with verdicts:

| Summary claim | Source checked | Verdict |
|---|---|---|
| Beam-4 is shipped/live | `streaming_zipformer/config.yaml:40-41` (`modified_beam_search`, `num_active_paths: 4`) | **CONFIRMED** — it is the committed default, not just a plan |
| Beam-4 Test1 21.28% Pareto-dominates greedy 23.44% on both axes | `research/09`, summary L40-42 | **CONFIRMED** in-doc; the two Test1 points are real submissions (846354, 806053) |
| Nemotron lost (27.97% > 23.44%), all rescue arms failed | `investigations/nemotron_vs_zipformer.md` F10 scoreboard, `research/39/41` | **CONFIRMED** — Arm A 27.35%, PEFT armC 28.19%, v1 23.58% Dev_diag |
| H7 offline (18.46%) vs streaming (28.19%) collapse | `research/41` F8, `investigations/h7_fix_plan.md` | **CONFIRMED as documented**; note both numbers run at `[70,1]` (Arm-1 falsified the "context mismatch" story) — the gap is masked-single-pass vs chunked-with-caches |
| Parakeet does not beat baseline | `docs/results/parakeet_comparison.md` | **CONFIRMED directionally, with a caveat** (below) |
| threads=1 fixed the timeout | `research/LOG.md`, `experiments/summary.csv` | **CONFIRMED** — 20-worker topology, 0 hash diffs, RTF 0.017 |
| No official exp_000 accuracy row | `experiments/summary.csv:2` (`status=planned`, empty CER/WER) | **CONFIRMED** — the tracking scaffold's accuracy columns are empty |

**Discrepancies / things to flag:**
- **Zipformer "16 ep" fine-tune** (summary L21): `config.yaml:21` loads `epoch-30.pt`. The checkpoint filename
  (epoch 30) ≠ the "16 ep" training claim. Not load-bearing, but the exact training recipe of the *winning*
  model is not pinned in a doc I could find — a reproducibility gap worth closing before any retrain.
- **Parakeet comparison uses a proxy scorer** (`docs/results/parakeet_comparison.md:5`, "NOT the official
  sclite clip@1 pipeline") and has **no zipformer row on the exact same subset** (its own caveat L64).
  The "zipformer wins" conclusion compares zipformer *streaming* 17.25% vs Parakeet *offline* 22% — strong
  directionally (streaming beating an offline ceiling is decisive), but not an apples-to-apples official number.
- **`experiments/` scaffold is essentially unused for accuracy.** Only the three speed rows are filled; every
  accuracy result lives in prose under `research/`. State is reconstructable only by reading ~41 docs
  chronologically. This is the single biggest reproducibility/process debt.

Net: the summary is **accurate and appropriately hedged.** No conclusion in it is contradicted by source.

---

## Critical Assessment

**What is genuinely strong.** (1) The faithful-harness discipline (never sign off on a proxy) is correct and
was learned the hard way (Nemotron 51.5% post-mortem). (2) The Nemotron investigation is a model of
competing-hypothesis reasoning — H1/H2/H3 cleanly killed, H4 confirmed, H5 falsified by running the actual
arm, H7 isolated. (3) The beam-4 win generalized Dev→Test1 as predicted (21% forecast → 21.28% real), which is
real evidence the zipformer's offline gates are trustworthy.

**Weaknesses / challenges to current thinking:**

1. **The winning model has received almost no modeling attention.** Every retrain, augmentation, severity-
   sampling, and PEFT experiment targeted Nemotron. The zipformer got exactly one decode-time change (beam).
   `research/33`'s accuracy recipe (converged training + multi-lookahead + severity-aware oversampling +
   speed-perturb) was written *for Nemotron* and then abandoned when Nemotron was stopped — but the recipe's
   logic (attack the ALS/DS severe tail via train-time sampling) applies **just as well, and with far better
   Dev→Test odds, to the zipformer** (+1.8 slope vs +18). The project effectively optimized the loser and
   banked the winner untouched. That is the biggest strategic miss.

2. **The latency axis is under-invested relative to its scoring weight.** Pareto rank uses mean(TTFT, TTLT),
   i.e. latency is *half* the objective, yet there is no chunk_size / left_context sweep on record. `config.yaml`
   exposes `chunk_size: 16` and `left_context_frames: 128` as one-line knobs (`model.py:347-357` reads them),
   and `model.py:141-145` shows TTFT is gated by `is_ready` needing `chunk_size*2 + pad` frames — so chunk_size
   is a *direct* TTFT lever with zero retraining. `research/09` even lists "beam-8 latency" as a quick next step
   and it was never done, despite beam-8 giving **18.33% CER** (−0.68 vs beam-4) at unknown latency cost.

3. **Pareto strategy is treated as "lowest CER wins," but the prize is non-domination.** The win condition is
   sitting on the frontier; the prize splits among *all* frontier teams. That means a low-latency / slightly-
   higher-CER point can be just as prize-winning as a low-CER point, and — critically — a single submission can
   be **dominated on both axes** by a competitor and win nothing. The docs frame beam-4 as "the #1 submission"
   but never stress-test whether it is *non-dominated* or merely *good*. Cheap latency variants are frontier
   insurance the project has not bought.

4. **Dev→Test transfer is asserted from n≈1 anchor for the zipformer.** The "+1.8 slope" rests on essentially
   one Dev↔Test1 pair. It's plausible and the beam-4 forecast held, but "the zipformer transfers well" is a
   two-point line. Any new zipformer variant must re-verify transfer, not assume it.

5. **Severe tail (ALS/CP/DS ~34% CER, empties) is the accuracy floor and remains unsolved on either model.**
   Beam helps most here (research/09: CP −3.9, Stroke −4.9) but only decode-side. No train-side attack on the
   zipformer's severe tail has been attempted.

**A methodological caution I do *not* raise:** I am not recommending re-opening Nemotron. The stop is correct;
`h7_fix_plan.md` correctly estimates P(Nemotron ships AND beats zipformer) ≈ 10–15%. H7/Arm-2 is closure work,
not EV work. Do it only if someone wants a *proof* to retire it, not to win.

---

## Research Gaps (prioritized)

| # | Gap | Type | Impact | Effort | Confidence |
|---|---|---|---|---|---|
| G1 | No RNN-LM shallow fusion on the winning model | Missing experiment | High (5–15% rel CER, stacks on beam) | Low–Med (text-only LM train) | High |
| G2 | No latency Pareto sweep (chunk_size/left_context) on zipformer | Missing ablation on a scored axis | High (claims a frontier corner ~free) | Low (config edits, no retrain) | High |
| G3 | Zipformer never got severity-aware / converged retrain | Missing experiment on the winner | High (attacks the severe-tail floor) | High (one gated GPU session) | Medium |
| G4 | beam-8 latency never measured | Missing datapoint | Med (−0.68 CER if latency holds) | Very low (one pod pass) | High |
| G5 | No official exp_000 baseline row; accuracy results not in scaffold | Reproducibility/process | Med (blocks clean A/B, handoffs) | Low | High |
| G6 | No same-harness zipformer row vs Parakeet; proxy scorer | Evaluation weakness | Low–Med (tighten a closed claim) | Low | High |
| G7 | ILM / density-ratio blank-bias correction untried on zipformer | Missing method | Med (severe-tail deletions/empties) | Med | Medium |
| G8 | H7 masked-vs-cached diagnosis (Arm 2) | Open question / closure | Low (won't change ship decision) | Med–High | Medium |

---

## Recommended Experiments

Ordered by EV. Each names where it lands in the repo and its faithful-gate stop condition.

**R1 — RNN-LM shallow fusion on beam-4 zipformer (G1, the single highest-EV accuracy lever).**
The explicit `research/09` "next" that was never run. Train a small LSTM/transformer LM on SAPC2 + typical
text via icefall `rnn_lm`, add sherpa online `lm` + `lm_scale` on top of `modified_beam_search`. Text-only
training is cheap (small GPU, or even CPU). *Where:* new sibling dir `track2_starting_kit/streaming_zipformer_lm/`
(never edit the pristine baseline; repo house rule). *Gate:* real `local_decode.py` both passes → `evaluate.sh`
on Dev **and** Dev_diag; ship IFF faithful CER improves with credible CI **and** TTFT/TTLT stay on the frontier
(LM adds per-step compute → must re-check the latency budget, `model.py:368-377`). *Risk to watch:* on the
severe tail, an LM prior can *worsen* deletions/empties (same ILM dynamic that made blank-penalty produce
garbage) — measure per-etiology, and consider density-ratio/ILME (R5) if the tail regresses.

**R2 — Latency Pareto sweep on the zipformer (G2 + G4, the cheapest frontier move).**
No retraining. Sweep `chunk_size ∈ {8,16,32}` × `left_context_frames ∈ {64,128,256}` and beam ∈ {4,8}, each a
`config.yaml` clone. Measure the full CER×TTFT×TTLT surface on Dev_streaming via the faithful harness. Deliver
**two frontier points**: the current low-CER point (beam) and a new **low-latency** point (small chunk_size).
*Where:* config-only sibling dirs; the loop is `model.py` unchanged. *Gate:* plot the Pareto surface; keep any
point that is non-dominated on (mean(TTFT,TTLT), CER). *Why it matters:* claims a frontier corner competitors
chasing CER may leave open, and is insurance against being dominated on both axes. Fold beam-8's latency (G4)
into this single sweep.

**R3 — Severity-aware / converged zipformer retrain (G3, the severe-tail attack on the model that transfers).**
Port `research/33`'s recipe to the *zipformer*: severity-aware oversampling (proxy severity by etiology +
per-utt error/empty propensity), verify/keep speed-perturb + SpecAugment, mild gain aug (±6–10 dB, **not** the
−20/+10 that confounded Nemotron Arm A per `research/40`), and confirm convergence (don't under-train). The
zipformer's +1.8 Dev→Test slope makes recovered Dev gains far likelier to transfer than Nemotron's +18.
*Where:* icefall zipformer training on a gated pod; new dir `streaming_zipformer_ft2/`. *Gate:* Dev **and**
Dev_diag faithful CER + speaker-block bootstrap CI + a typical-speech forgetting probe; ship IFF it beats
beam-4's Test-projected point. *This is the one GPU bet I would actually fund*, and only after R1/R2 (which
are near-free) are banked.

**R4 — Close the process/repro gaps (G5, G6).** Record an official `exp_000` accuracy+latency row for the
pristine baseline through `evaluate.sh`; backfill the beam-4 and greedy rows into `experiments/summary.csv`;
add one same-harness (official sclite) zipformer row on the Parakeet subset to convert that directionally-strong
claim into an apples-to-apples one. Low effort, unblocks every future A/B and handoff.

**R5 — ILM / density-ratio blank-bias correction (G7), only if R1's tail regresses or empties persist.** A
principled decode-time fix for transducer over-blanking on the severe tail, unlike the crude blank-penalty that
failed. Med effort; conditional on R1 evidence.

**R6 — H7 Arm-2 audio.cpp diagnostic (G8), closure-only.** Per `h7_fix_plan.md`, run NeMo-native streaming @
`[70,1]` once to settle proxy-inflation vs export-bug. Do this *only* to retire Nemotron with proof; it will
not change the ship decision. Lowest priority.

---

## Engineering Improvements

- **Populate `experiments/summary.csv` as the single source of truth for accuracy** (currently speed-only).
  Every `research/NN` result should have a matching row + `experiments/exp_XXX/` snapshot (config + git hash +
  metrics json). This is a stated repo rule (`SAPC-template/CLAUDE.md`) that is being honored for speed and
  ignored for accuracy.
- **Consolidate the ~15 root handoff docs.** `RESEARCH_SUMMARY_FOR_FABLE.md` is now the best single entry point;
  `PLAN.md` is stale (session-1). Mark `PLAN.md` superseded and point it at the summary to stop future agents
  re-deriving state from 41 chronological docs.
- **Pin the winning model's exact training recipe** in one doc (the "16 ep vs epoch-30.pt" ambiguity). R3 needs
  it; a reproducible retrain requires knowing the baseline recipe precisely.
- **Parameterize the sweep.** A tiny wrapper that clones `streaming_zipformer/` with an overridden `config.yaml`
  and runs the faithful gate would make R2 (and future latency work) a one-command matrix instead of manual dir
  copies — reusing `local_decode.py` + `evaluate.sh` unchanged (wrap, don't modify).
- **Keep the pristine `streaming_zipformer/` untouched** as the always-working reference; all of R1–R3 land in
  sibling dirs. (Already the house rule; restating because R1–R3 all tempt in-place edits.)

---

## Prioritized Roadmap

### Immediate (next few hours — no GPU, mostly local + one cheap pod pass)
- **R4 process cleanup** (local now): backfill `experiments/summary.csv` accuracy rows, record the exp_000 gate
  intent, mark `PLAN.md` superseded, pin the zipformer recipe. Zero cost, unblocks everything.
- **Write R2's sweep configs + wrapper locally** (config clones + a run script). No pod yet — stage it fully so
  the pod session is just "run + copy + stop" per the cost gate.
- **Stage R1's LM-training + sherpa-fusion scripts locally** (icefall `rnn_lm` recipe, `lm_scale` wiring) so the
  only pod work is training + faithful eval.

### Short Term (next gated pod sessions, cheapest first)
1. **R2 latency Pareto sweep + beam-8** — one pod pass, no retrain, delivers a second frontier point and closes
   the beam-8 question. Highest EV-per-dollar.
2. **R1 RNN-LM shallow fusion** — train small LM, gate on Dev/Dev_diag, check latency budget. Highest EV-per-CER
   on the winner.

### Medium Term (one funded GPU bet, after R1/R2 bank)
3. **R3 severity-aware zipformer retrain** — the severe-tail attack on the model that transfers. Full cost-gate
   ritual + smoke→guardrail→full. Only after the near-free wins are in.
4. **R5 ILM/density-ratio** — conditional on R1's per-etiology evidence.

### Stretch Goals
- **A latency-first submission family**: deliberately push chunk_size down to own the low-latency corner of the
  frontier, treating the Pareto prize structure as a multi-point game rather than a single "best CER" race.
- **TTS-synthesized dysarthric augmentation for ALS/DS** (`research/33 §5`) if R3 leaves the severe tail binding
  — highest-ceiling data lever, heavy pipeline, defer until the cheaper severity-sampling run proves the tail is
  still the constraint.
- **FastEmit-style blank/latency regularization as a training loss** (not a decode hack) on the zipformer — the
  only lever that moves *both* CER-tail deletions and TTLT simultaneously; novel here, higher effort.

---

## Bottom line
The project correctly identified and banked its winner and correctly killed its losing bet. Its blind spot is
that the winner was never actually pushed: the two evidence-backed, never-run levers (**LM shallow fusion**,
**severity-aware retrain**) and the **entire latency axis** all sit on the model that transfers well. The next
phase should be short, cheap, and aimed squarely at the zipformer — LM fusion for CER, a config-only sweep for
the latency frontier, then one funded severity retrain — with Nemotron/H7 left as closure-only work.

---

## Appendix — Execution log (Fugu, 2026-07-20, no-GPU local session)

The three near-free "Immediate" items from the roadmap above were **executed locally**
this session (darwin box, no SAP data → no faithful-harness runs; these are code +
process artifacts staged for the next pod, per the AGENTS.md cost gate):

- **R4 backfill (DONE).** `experiments/summary.csv` now carries the verified accuracy
  rows it was missing: `exp_a1_greedy_test1` (23.44/31.51), `exp_a1_beam4_test1`
  (21.28/29.5, LIVE #1), `exp_nemotron_int8_test1` (27.97/37.71),
  `exp_v1_nemotron_devdiag` (23.58), `exp_armA_nemotron_devdiag` (27.35, failed),
  `exp_peft_armC_nemotron_devdiag` (28.19, failed). Each note cites its research/NN
  source. Verified parseable (10 rows × 14 fields).
- **Speaker-disjoint gate (NEW, DONE + self-tested).** `scripts/make_speaker_disjoint_dev.py`
  builds `Dev_heldout.csv` / `Dev_pool.csv` (+ id-matched `*_streaming.csv`) from the
  manifest schema in `utils/manifest.py`, holding out whole speakers stratified by
  etiology. Stdlib-only, `py_compile` clean; self-test on synthetic data proved zero
  pool/heldout speaker overlap, all 5 etiologies covered in heldout, and streaming-id
  parity. This operationalizes the anti-overfit control (Critical Assessment #4;
  AGENTS.md §5) that turns Dev into an honest Test proxy — the missing instrumentation
  for the Dev→Test risk.
- **R1+R2 pod staging (DONE).** `scripts/pod_run_lm_and_latency_sweep.sh` bundles the
  cheap wins into ONE cost-gated session: builds the speaker-disjoint gate, then runs
  the faithful harness (`local_decode.py` + `evaluate.sh`, unmodified) for exp_000
  (E-E), LM shallow fusion (R1/E-B), beam-8 (R2/G4), and the config-only latency sweep
  (R2/E-C), each writing an `experiments/exp_XXX/` snapshot; `bash -n` clean. It
  encodes the stop condition (scp back + stop pod once metrics are known).

**Not done (correctly gated):** any actual CER/latency number — requires SAP data + a
Linux/Docker CPU host + (for R1/R3) a GPU pod, all behind the explicit cost gate. The
next agent with a pod should: (1) prebuild the `streaming_zipformer_lm/`, `_beam8/`,
and latency sibling dirs, (2) run `scripts/pod_run_lm_and_latency_sweep.sh`, (3) pick
the point that is non-dominated on the speaker-disjoint gate, (4) submit + stop.
