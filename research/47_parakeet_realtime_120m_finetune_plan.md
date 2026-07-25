# 47 — Fine-tune `parakeet_realtime_120m` for a low-latency Pareto corner (plan)

> Expands the shelf-ready stub in `research/46` §7 into an executable, gated plan.
> **Status: SHELF-READY, not launched.** Needs explicit Q go + a fresh GPU pod.
> Companion: `TEAM_BRIEF_finetuning_techniques.md` (the Nemotron/Zipformer recipe we port from),
> memories `small-model-sweep-verdict`, `nemotron-vs-zipformer-roadblock`, `h7-fix-plan`,
> `validate-against-real-harness`.

---

## 0. The bet, stated honestly
`parakeet_realtime_eou_120m` is the **only** zero-shot model that hints at a viable *low-latency*
play: **31.8% CER @ 80 ms lookahead** on Dev_streaming (proxy scorer, EOU-stripped) — best zero-shot
CER *and* lowest latency of everything swept (research/46 §6). Fine-tuning closed **~19 pts** for the
zipformer (36→17 proxy / 12 official). If it closes a similar gap here, a FT parakeet lands **~14–17%
CER at 80 ms** — a corner our best zipformer (cs8: 13.85% @ **926 ms** TTFT; beam-8: 11.62% @ 1157 ms)
**cannot reach on latency**. That is the entire strategic reason to spend: not to beat the zipformer on
CER, but to **open a non-dominated corner it can't reach**, or to beat A1 outright.

**This is a bet, not a plan-to-ship.** It is the same NeMo cache-aware streaming-FT path the
Nemotron-0.6B lost on. What makes it *worth* the bet now (vs. the Nemotron post-mortem pessimism):

- **H7 is resolved (research/45): the streaming export is FAITHFUL.** Base `transcribe` 43.4% ≈
  NeMo-native streaming 43.8% ≈ our ONNX-export→`local_decode.py` 43.4% (0.4-pt spread). The dreaded
  offline→streaming "collapse" was **PEFT-adapter-specific** (adapters not surviving streaming
  cache-norm) or a confounded compare — **not** intrinsic to cache-aware streaming and **not** an
  export bug. → A **full or encoder-only** FT (no PEFT adapters) is expected to survive streaming.
  This is the single most important update; it removes the mechanism that sank the Nemotron rescue.
- The remaining Nemotron losses were **domain transfer** (Dev→Test +18 vs zipformer +1.8) and
  **severe-tail collapse / confident empties** — those are still live risks and the gate is built around them.

---

## 1. Direct answers to Q's two questions

### Q1: Should we use data augmentation? — **Yes, but selectively.**
| Augmentation | Verdict | Why |
|---|---|---|
| **Speed perturbation 0.9 / 1.0 / 1.1** | **YES — top priority** | The **high-value lever MISSING from v1 Nemotron** (brief §3.3, §6). Dysarthric speaking rate is highly variable; 3× speed copies directly target that. Cheapest real win available. |
| **SpecAugment, dysarthria-tuned** (more time masking, *less* freq masking) | **YES** | Standard robustness; the tuning (heavier time masking) matches the slow/variable-rate failure mode. Used in v1. |
| **Noise / RIR reverb (MUSAN-style)** | **MEASURE, don't assume** | SAP recordings aren't primarily a noise-robustness problem — the domain *is the disorder*. Over-augmenting risks **washing out the very articulation signal** the model must learn. Add only if a robustness/forgetting probe shows net benefit. Not in the v1 recipe. |
| **Pitch / tempo perturb (independent)** | **Skip initially** | Speed-perturb already couples rate+pitch; SpecAugment covers spectral masking. Low marginal value; revisit only if speed-perturb helps and we want more. |

**The augmentation caveat that matters most:** aggressive augmentation on *already-distorted* dysarthric
articulation can hurt more than it helps. Core = speed-perturb + tuned SpecAugment. Everything beyond
that is an ablation, measured on the faithful gate, not assumed.

### Q2: Other techniques that might help (ranked by expected ROI)
1. **Encoder-only FT first, then a low-LR joint/pred-net unfreeze arm.** Encoder-only *won* for
   Nemotron (less overfit: Dev_diag 23.6% vs 25.2% full-unfrozen) — start there as the safe anchor.
   BUT with the joint frozen, the **empty-output rate is capped** (brief §3.11), and under the scorer's
   per-utterance **100%-clip, every empty is pure 100% CER** — the single most expensive failure mode
   for a low-latency model. So the second arm is the **low-LR unfreeze of the joint/pred-net** — the
   one structural lever the Nemotron work identified but never ran. This directly attacks the empty
   floor + hardest speakers.
2. **Checkpoint averaging (SWA-style).** Measured-beneficial for Nemotron (11.83% avg vs 12.47% best
   single) *and* produced fewer empties. Nearly free at export time.
3. **Beam search decoding.** Free accuracy: Zipformer greedy 21.6→beam-4 19.0→beam-8 18.3, and
   **beam ≈ greedy on latency** because TTFT is chunk-accumulation-bound, not search-bound. Helps most
   on the hardest etiologies (CP/Stroke) — exactly parakeet's weak spot (CP 48.8 zero-shot). Confirm
   per-chunk decode still fits the 100 ms budget at 80 ms lookahead.
4. **int8 dynamic quantization of the encoder.** Near-lossless for Nemotron (FP32→int8 essentially
   free), and required to make CPU RTF + package size comfortable. Keep the small decoder FP32.
5. **LR-schedule fix.** v1's 15% warmup over a 4-epoch run under-trained (~60% of epoch-1 warming up).
   Use **5–10% warmup** for short runs. Base recipe: AdamW, lr 1e-4, wd 1e-3, betas (0.9,0.98), cosine,
   bf16, early-stop on speaker-disjoint internal val.
6. **Severity-aware sampling.** Payoff is split-dependent (brief §6) — measure, don't bake in.

---

## 2. The deployment-correctness landmines (wrapper = part of the model)
These are the bugs that make a *good* checkpoint look *broken*. Budget time for them BEFORE trusting any CER:
- **`<EOU>` stripping.** parakeet emits an end-of-utterance token; 100/123 hyps had it in the sweep.
  Strip standalone `<EOU>` in `model.py` (same class of fix as the `unk` strip — never touch the scorer).
- **RNN-T SOS / blank init.** Nemotron's one-line SOS-vs-blank bug made encoder-only look 11.7% when it
  was 1.3%. Verify parakeet's decoder priming token explicitly before believing any streaming number.
- **Config pinning:** `att_context_size = [70,1]` (80 ms) pinned **identical** across
  train = eval = export = deploy. The Nemotron `[70,1]`-train / `[70,6]`-deploy own-goal (research/19)
  is exactly what we must not repeat.
- **Mel frontend recompute.** The v1 wrapper recomputed the filterbank ~9× over the whole prefix per
  chunk — a real, fixable TTFT tax. Check parakeet's wrapper doesn't do the same.

---

## 3. The gate — write the success criterion BEFORE running (house rule)
**Ship IFF**, on the organizers' **exact** pipeline (real `local_decode.py` both passes → `evaluate.sh`
→ official sclite), the FT parakeet on Dev meets a **Test-predictive** bar:
- **(a) beats A1** (Test1 21.28% beam-4) on a Dev split we trust to project to Test, **OR**
- **(b) opens a non-dominated corner**: TTFT p50 materially **< 926 ms** (our cs8 corner) at a CER
  that keeps `(CER, mean(TTFT,TTLT))` **non-dominated** vs {beam-8, cs8, beam-4}.

Gate specifically on, all through the faithful harness:
- **Held-out SEVERE** (Dev_diag-425) — the transfer wall lives here; a Dev_streaming win alone is NOT a submit trigger.
- **Empty rate** (each empty = 100% CER; watch it explicitly, it's the low-latency killer).
- **Clean-speech forgetting probe** — did FT wreck the pretrained competence (a Nemotron symptom)?
- **Speaker-disjoint** internal val (hold out ~8% of speakers, deterministic seed) — never tune to Dev/Test text.

**Never submit to test a hypothesis.** Proxies (`score_predict.py`) are for *ranking/shape* only.
A Dev win is a projection, not a promise (Nemotron Dev→Test +18). See `validate-against-real-harness`.

---

## 4. Staged execution (cost gate — all code local first, stop at each decision metric)
Reuse the Nemotron FT scaffolding — **swap the base model, don't rewrite** (`scripts/nemo_finetune.py`,
`scripts/export_deploy.sh`); new sibling dir `track2_starting_kit/parakeet_realtime_ft/` (keep
`parakeet_realtime_120m` zero-shot wrapper pristine as reference).

- **Stage 0 (local, no spend):** finalize FT config + augmentation flags; write the deploy `model.py`
  with EOU-strip + verified SOS init + `[70,1]` pinning; `py_compile` / import-lint / synthetic
  `accept_chunk` smoke. Confirm exact pod file-upload list for Q approval.
- **Stage 1 — SMOKE (≤1 GPU-hr):** FT on a tiny slice, 20-utt faithful decode both passes. Gate:
  pipeline runs end-to-end, partials non-empty, EOU stripped, CER computes. (Correctness, not accuracy.)
- **Stage 2 — Arm A: encoder-only FT (full data).** Faithful gate on Dev_streaming + Dev_diag + empties
  + forgetting probe. Decision: does it clear §3 (a) or (b)? Record vs zero-shot 31.8 and vs A1.
- **Stage 3 — Arm B: low-LR joint/pred-net unfreeze** (only if Arm A promising but empty-floor-bound).
  Same gate. This is the empty-floor lever.
- **Ablations (only if a base arm clears):** ± speed-perturb, ± SpecAugment intensity, ± checkpoint-avg,
  beam width, int8. One variable at a time, one paired faithful-gate comparison each.

**Stop discipline:** minimum size first (smoke→guardrail ~200 speaker-disjoint→full); stop the moment
the stage's decision metric is known; copy artifacts back; **stop the pod immediately.** E2 is a
separate approved GPU bet — do not fold it into an E1/exploration pod.

---

## 5. Risks / kill-conditions
1. **Transfer wall (research/37 §6).** Dev win may not move Test (held-out speakers). Gate is held-out-severe; even a Dev win ≠ submit.
2. **Severe-tail collapse + confident empties** — the Nemotron symptom. If FT parakeet's Dev_diag empties don't drop with the joint-unfreeze arm, the low-latency thesis is weak (empties dominate CER under clipping).
3. **80 ms lookahead ≠ low CPU TTFT.** Algorithmic delay is low, but per-chunk compute could erase it. Measure actual TTFT on the faithful CPU harness; do not infer from lookahead.
4. **sherpa-onnx cache-aware export is WIP.** Prefer the NeMo-native streaming wrapper path (as the 114M bench used), not sherpa-onnx export, to sidestep the export bug.
5. **Kill-condition:** if Arm A + Arm B both fail to clear §3 (a)/(b) on the faithful severe gate → this line is closed; **default remains ship the gated beam-8/beam-4 zipformer.**

---

## 5b. Stage 0 — DONE (local, no spend) — 2026-07-24
Staged and locally verified; nothing touched in the organizer harness or the Nemotron scripts.
- **Created `track2_starting_kit/parakeet_realtime_ft/`** (Codabench-shaped sibling; pristine
  `fastconformer_medium32/` untouched): `model.py`, `config.yaml`, `setup.sh`, `requirements.txt`,
  `README.md`. Wrapper = medium32 template + `<EOU>`/special-token strip (`_clean`) + `[70,1]` pinning
  + generic `ASRModel.restore_from` (RNNT/TDT-agnostic) + NeMo-native decode (no hand-rolled SOS).
- **Created `scripts/smoke_parakeet_ft_wrapper.py`** — mock-model contract smoke (NeMo absent).
- **Local checks PASS:** `py_compile` (model + smoke), `bash -n setup.sh`, and the smoke's 18 checks
  (buffering cadence, sub-chunk buffering, callback/TTFT firing, `<EOU>` strip, reset, `_extract_text`
  quirk). Env: py3.13 / torch 2.6 / numpy 2.2 / omegaconf 2.3, **no NeMo** (correct scope).
- **Training reuse decided:** `scripts/nemo_finetune_v2.py` runs Arm A (`--freeze encoder_only`) and
  Arm B (`--freeze joint_unfreeze`) **unmodified** — it already has speed+gain aug (toggles), tuned
  SpecAugment, checkpoint-avg, 10% warmup, `[70,1]` pinning. **One GATE0 pod item:** line 48 restores
  via `EncDecRNNTBPEModel`; parakeet may need generic `ASRModel.restore_from` (1-line pod fix, NOT a
  local edit to the shared Nemotron script — second-order safety).
- **Correction logged:** there is **no committed zero-shot parakeet wrapper** (the 31.8% sweep ran on a
  now-stopped pod); Stage 0 built from the `fastconformer_medium32` template instead. The exact
  `from_pretrained` id remains **VERIFY-ON-POD** (`parakeet_realtime_eou_120m` is the sweep name, not a
  confirmed HF/NGC id) — not fabricated.

### Exact pod-upload list (for Q approval before any spend — cost gate)
If the pod has a fresh repo checkout/pull, **nothing new needs uploading** (all files are in-repo).
Otherwise the minimal set is:
1. `track2_starting_kit/parakeet_realtime_ft/{model.py,config.yaml,setup.sh,requirements.txt}`
2. `scripts/nemo_finetune_v2.py` (if not already synced)
3. `scripts/prep_nemo_manifest_v2.py` (SAP→NeMo manifest prep, if not already synced)
Already-on-pod (organizer harness, do NOT re-upload/modify): `track2_starting_kit/local_decode.py`,
`evaluate.sh`, `utils/`, `steps/`. No ONNX export step needed — parakeet deploys the `.nemo` directly
via the NeMo-native wrapper (simpler than the Nemotron ONNX path; `export_deploy.sh` not used).

## 6. What I need from Q to launch
1. **Go / no-go** on this as a funded GPU bet (it is not free; it is the queued §7 experiment).
2. **Fresh GPU pod + budget cap**, and who runs `runpodctl pod start`.
3. **Exact upload-file-list approval** for pushing local FT code/wrappers to the pod (sandbox policy).
4. **Data location** for SAP Train (+ Dev / Dev_streaming / Dev_diag manifests + WAVs) on the pod.
