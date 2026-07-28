# Investigation — How do we make parakeet WIN? (research framework)

**Date:** 2026-07-28 · **Author:** coding agent (for Q) · **Status:** framework, pre-execution
**Companion evidence:** `experiments/exp_parakeet_ft_empties/*` · `PLAN.md` parakeet blocks ·
`RESEARCH_NOTES.md` §7 · memories `parakeet-ft-wrapper-streaming`, `test1-standing-and-pareto`.

> Discipline note (global CLAUDE.md): FACTS = verified/observed; THEORY = plausible, untested.
> We hold ≥5 competing hypotheses and gate each with a cheap falsifier. One theory is
> confirmation bias with extra steps.

---

## 0. The win condition, stated precisely

We are scored on the **CER × latency Pareto frontier** (latency = mean(TTFT, TTLT)). A submission
"wins" a region of the frontier if no other submission dominates it on both axes.

- **Latency axis — already a win, already banked.** Parakeet Arm A streams at **mean ≈ 371 ms**
  (TTFT p50 669 / TTLT p50 73), non-dominated vs the whole Test1 board (board min 592 ms). *FACT*
  (proxy scorer, gate 2026-07-27). Nothing else we own reaches this corner; the zipformer sits at
  ~926 ms.
- **CER axis — the open problem.** Parakeet severe-slice CER = **29.93%** vs the banked
  zipformer's **~24.85%** severe. On the accuracy axis parakeet currently *loses*.

**So the win is single-sentence:** *drive parakeet severe CER below ~24.85% while holding the
371 ms corner and staying in the CPU/15000 s budget.* If we do that, parakeet is a strictly-added
frontier point the field cannot dominate — and it's ours alone at that latency.

---

## 1. Where the loss actually lives (CER budget accounting) — FACT

From `empties_analysis.json` (Dev_diag severe, n=425):

| Component | Value | Meaning |
|---|---|---|
| Total CER | **29.93%** | what we're trying to beat 24.85% with |
| Non-empty CER (377 utts) | **21.00%** | the "ordinary error" floor if empties vanished |
| Empties | **48/425 (11.3%)** | contribute **+11.29 CER points** (each clipped at 1.0) |
| CER if empties → non-empty avg | **21.00%** | −8.9 pts, **already beats the zipformer** |
| CER if empties → perfect | **18.63%** | −11.3 pts |

**Two independent levers, and the ranking is not close:**

- **Lever A — kill the empties.** Worth up to **−8.9 to −11.3 CER points**. Killing empties *alone*
  (even to the mediocre non-empty average) takes parakeet from 29.9% → 21.0%, i.e. **below the
  zipformer's 24.85%**. This one lever is *sufficient to flip the verdict.*
- **Lever B — squeeze the non-empty 21.0%.** Ordinary sub/del errors (del-rate 9.8%, del:sub ratio
  2.2 — a deletion-heavy profile, itself a mild blank-bias tell). Real, but smaller, and it's the
  same lever the whole field is pulling.

**Strategic consequence:** the empty tail is not a side-bug. It *is* the CER problem. The entire
research program should be organized around it. Everything else is a second-order squeeze.

---

## 2. The decisive evidence: the empties are RECOVERABLE — FACT

`zf_on_parakeet_empties.json` — run the fine-tuned **zipformer** on the exact 48 utterances
parakeet emits empty:

- Non-empty **58.3%** (28/48).
- **31.2%** scored **< 50% CER**; a cluster at **0.0 CER**: `alexa`, `dog`, `period`, `hey siri`,
  `football`, `throw`, `answer the call`, `what is on my shopping li[st]`.

**Interpretation (this is the crux of the whole framework):** the acoustic information needed to
transcribe these is *present in the signal* — a different model, fine-tuned on the same dysarthric
data, reads it fine. Therefore parakeet's empty is **not an information-theoretic floor** and **not
"the audio is too quiet to contain speech."** It is a **model-internal decoding pathology** unique
to *this* parakeet checkpoint. Pathologies are fixable; information floors are not. **This is why we
can win, not just place.**

(Note the earlier "quiet-onset root cause" from the causal-gain gate was correct about the *audio
statistics* — 40/48 have onset < −45 dBFS — but drew the wrong *conclusion*. Quiet onset is why the
**causal-inference** patch failed, not why the model must fail. The zipformer proves a model *can*
succeed on this same audio.)

---

## 3. What is wrong *inside* the model (root-cause theory)

### 3.1 The mechanism — THEORY (strongly supported)

Parakeet is an **RNN-T / TDT transducer**. At each acoustic frame `t` and label position `u`, the
**joint network** `J(enc_t, pred_u)` emits a softmax over `{vocabulary ∪ blank}`. Emitting *blank*
advances `t` without advancing `u`; emitting a token advances `u`. An **all-blank alignment over
the whole utterance yields the empty transcript.**

We fine-tuned **Arm A = encoder-only.** The **joint network and the prediction (decoder/LM) network
kept their pretrained weights** (LibriSpeech: clean, read, well-articulated, normal-energy audiobook
speech, and a vocabulary that contains no "alexa"/"cortana"/"hey siri"). So:

1. **The blank-vs-emit decision boundary was never retrained.** It is calibrated to a distribution
   where "quiet + breathy + short + atypical articulation" reads as *non-speech* → blank. Hypophonic
   dysarthric input lands on the blank side of a boundary nobody moved.
2. **Encoder-only FT can only fight this indirectly.** It can shift features, but to fix an empty it
   must shove these inputs across a *frozen* boundary into an already-"emit" region — a harder,
   more roundabout optimisation than simply *moving the boundary*. The gradient path to the blank
   logit is cut off.
3. **The deletion-heavy non-empty profile (del:sub = 2.2) is the same bias, sub-clinical.** The
   frozen joint over-emits blank everywhere; on strong audio that's a dropped phoneme, on weak audio
   it's a whole empty utterance. Same knob, two severities.

### 3.2 Literature corroboration — FACT (`[V-lit]`)

- `RESEARCH_NOTES.md` line 112: **"Full FT > freezing decoder/joint — confirmed, Takahashi'25
  Table 2."**
- The **SAPC1 first-place** system (Takahashi, Parakeet-TDT) did **full fine-tuning, no freezing.**
  We froze the two components that own the blank decision. *We have been fine-tuning the one part of
  the model that isn't the problem.*

### 3.3 What this predicts

If the theory is right, **unfreezing the joint (+prediction) network and continuing to train should
collapse the empty rate**, because the RNN-T loss on these utterances (whose refs are non-empty)
directly penalises the all-blank alignment and pushes P(blank) down where it currently saturates.
That is Arm B, and §5 gives it a falsifier.

---

## 4. What exact part of the *data* is tricky — FACT

The empties are not random; they are a tight cluster. Four correlated axes:

1. **Etiology / dysarthria type.** Empty-rate by disease: **ALS 24.5%** · Down 7.5% · CP 10.5% ·
   **Parkinson 0%** · **Stroke 0%**. ALS → flaccid/mixed dysarthria, weak respiratory support →
   **hypophonia (low volume) + breathiness**. Parkinson is hypokinetic (monotone) but retains onset
   energy → 0 empties. The gradient is essentially a **vocal-energy / SNR gradient**, not a "how
   dysarthric" gradient.
2. **Acoustics.** Empties have a **quiet onset** (first-100 ms median **−58.5 dBFS**) sitting on a
   **loud body** (full-utt RMS **−29.6 dBFS**). The energy is back-loaded.
3. **Length & lexicon.** Empties skew **short** (median 4.67 s, p25 2.82 s) and heavily
   **command/wake-word** (`alexa`, `cortana`, `hey siri`, `dog`, `football`). Short + quiet-onset =
   the transducer's cheapest escape is to blank the whole thing. And wake-words are **OOD lexicon**
   for a LibriSpeech prediction network → weak language prior to pull a token out.
4. **Speaker concentration.** **22 of 48 empties are one speaker** (`55c1784a`). Part of the failure
   is a single-speaker acoustic idiosyncrasy → a **personalization** surface (SAPC1 Team h).

**The tricky slice, in one line:** *short, quiet-onset, command-word utterances from
low-vocal-energy (ALS/Down) speakers.* Everything in §5 should be evaluated first on THIS slice, not
on aggregate Dev.

---

## 5. Hypothesis portfolio (competing, ranked by expected ROI × probability)

Each carries a **mechanism**, a **cheap falsifier**, and a **kill criterion**. We do not marry one.

### H1 — Joint+prediction unfreeze (Arm B). **PRIMARY.**
- **Mechanism (§3):** make the blank logit trainable; RNN-T loss pushes P(blank) down on dysarthric
  onsets. Directly targets the −8.9 pt lever.
- **Controlled recipe (this is how we avoid the Nemotron collapse scar):**
  - *Graded, not full.* Start from the Arm A encoder checkpoint. Unfreeze **joint + prediction
    only**; keep encoder at a **much lower LR** (e.g. 10–50× smaller) or frozen for the first phase.
    The joint is small → cheap, and less collapse-prone than a full unfreeze.
  - **Differential learning rates**, short schedule, **checkpoint averaging** (SAPC1 winner:
    4-ckpt avg), **early-stop on a speaker-disjoint dev empty-rate**, not on aggregate loss.
  - Optional **blank/emit loss shaping**: mild down-weight of the blank class or a small emit prior
    — only if plain unfreeze under-delivers (avoid over-emission → insertions).
- **Falsifier:** on the 48-empty slice, Arm B empty-rate does **not** drop below ~20% (from 100%).
- **Kill criterion:** if empties don't move AND non-empty CER rises (over-emission), Arm B is wrong
  about the mechanism → escalate to H3/H5 or close the parakeet CER line.
- **Cost:** one GPU pod, hours. **Risk:** overf/collapse (mitigated above). **This is the bet.**

### H2 — Failure-mode-targeted train-time augmentation. **PRIMARY, pairs with H1.**
- **Mechanism:** the inference-time gain patch is dead (causal blindness, §2). Its **train-time dual
  is exactly the fix**: we cannot normalize the quiet onset at test time, but we *can teach the model
  to emit under quiet/low-SNR onsets by training on them.* Augment the LOUD training utterances into
  the failure region so the joint sees low-energy *emit* targets:
  - random **gain reduction** (push RMS toward −40..−55 dBFS onsets),
  - **leading-silence / slow-onset** insertion,
  - **low-level additive noise** (lower SNR),
  - **tempo/speed perturb 0.9–1.1** (SAPC1 winner) + **SpecAugment** (already standard),
  - light **reverb**.
- **Falsifier:** Arm B **+ aug** beats Arm B **− aug** on the 48-empty slice. (A/B the augmentation.)
- **Kill criterion:** if aug hurts non-empty CER (distribution drift) more than it helps empties.
- **Cost:** free-ish (data pipeline), folds into the H1 pod run. **High ROI, low risk.**

### H3 — Severity/energy-weighted resampling. **CHEAP MULTIPLIER on H1/H2.**
- **Mechanism:** the tricky slice is a *minority* of train too; the joint's gradient is dominated by
  easy, loud, Parkinson/Stroke-like data. **Oversample low-energy / ALS / short-command utts** (or
  weight the loss by inverse energy / by empty-propensity) so the boundary move in H1 is driven by
  the cases that matter.
- **Falsifier:** weighted-sample Arm B beats uniform-sample Arm B on the empty slice.
- **Cost:** trivial (sampler weights). **Do it inside the H1 run as an arm.**

### H4 — Beam / mAES decoding. **SECONDARY, latency-gated, do AFTER H1.**
- **Mechanism:** reorders among *plausible* hypotheses → a CER squeeze on the non-empty 21%. **It
  does NOT manufacture tokens** (`RESEARCH_NOTES.md` §3.6): beam over an all-blank posterior is still
  all-blank. **Cannot fix empties.**
- **Hard constraints (already researched):** transducer beam is **5–10× slower on CPU** (the cheap-
  beam papers are GPU/CUDA-graph results that *do not transfer*); beam **reorders partials → TTFT/
  TTLT flicker** → can *lose* the latency corner; NeMo 2.7.3 cache-aware beam hits `partial_hypotheses`
  incompatibilities (integration task, not a flag). Treat as a **Pareto trade**, measure on the
  faithful streaming pass.
- **Kill criterion:** any TTFT/TTLT regression that dominates the CER gain → drop it. Corner > squeeze.

### H5 — Speaker personalization / adaptation. **EXPLORATORY.**
- **Mechanism:** 22/48 empties are one speaker → a speaker-embedding / i-vector / light per-speaker
  bias could recover a concentrated chunk (SAPC1 Team h). But **test speakers are unseen** → only
  helps if adaptation is *test-time unsupervised* (risky, latency cost) or if a speaker-conditioning
  input generalizes. Low near-term ROI; park it.

### H6 — LLM / rule-based post-ASR correction. **REJECTED for Track 2.**
- SAPC1 teams used it, but it **adds latency + CPU** → **kills TTLT**, our winning axis. `[V]` §7.4.
  Off the table while latency is the moat.

### H7 — Synthetic dysarthric speech (TTS / voice conversion). **DEFERRED.**
- Fabricating dysarthric audio (dysarthria-TTS, VC) is high **distribution-mismatch** risk, expensive,
  and unproven for the *specific* quiet-onset failure. H2 (augmenting *real* dysarthric audio into the
  failure region) gets 80% of the benefit at 5% of the risk. Revisit only if H1+H2+H3 plateau above
  target.

---

## 6. The framework (how the hypotheses compose into a plan)

**Organizing principle:** *the empty tail is the CER problem; attack it at its true seat (the frozen
joint) with the one intervention that owns the blank decision (H1), amplified by data that
over-represents the failure (H2, H3), and only then squeeze the remainder (H4) if latency permits.*

```
                 ┌─ hold latency corner (greedy, [70,1], CPU budget) ── invariant, never trade blindly
WIN = severe CER │
   < 24.85%      └─ drive CER down:
                     Stage 1  H1 (joint+pred unfreeze)      ← the −8.9pt lever, the whole ballgame
                       ⨯ H2 (failure-targeted augmentation) ← teach emit-under-quiet-onset
                       ⨯ H3 (severity/energy weighting)     ← aim the gradient at the tricky slice
                     Stage 2  H4 (mAES beam, latency-gated) ← squeeze the residual 21%, IFF Pareto-safe
                     Park:    H5 personalization · H7 synth-TTS · reject H6 post-LLM
```

### Staged execution with kill-gates

**GATE 0 — reproduce the control (no new science).** Restart the pod, load Arm A, run the REAL
`local_decode.py` (both passes) → `evaluate.sh` on Dev_diag severe. Must reproduce **29.96% / 48
empties** with gain OFF. *Purpose:* prove the harness + checkpoint are the ones the theory is about.
No claim ships on the proxy scorer (house rule `validate-against-real-harness`). **Kill:** if the
official gate disagrees with the proxy, stop and re-baseline before any training.

**GATE 1 — Arm B efficacy (H1 ⨯ H2 ⨯ H3), measured on the 48-empty slice FIRST.**
Train 3 arms in one pod session, cheapest signal first:
  - B0 = joint+pred unfreeze, uniform sample, no aug (isolate H1).
  - B1 = B0 + failure-targeted aug (adds H2).
  - B2 = B1 + severity/energy weighting (adds H3).
Decision metric = **empty-rate on the 48 slice** and **severe-slice CER via the official scorer**.
- **PASS:** any arm drives severe CER **< 24.85%** while holding the corner → that's the submission.
- **PARTIAL:** empties drop but CER stalls 25–29% → escalate loss-shaping / more aug, one more arm.
- **KILL:** empties don't move below ~20% AND non-empty CER worsens → §3 theory is wrong about the
  seat of the pathology; parakeet becomes latency-only, ship the zipformer, close the CER line.

**GATE 2 — latency + budget re-confirm (mandatory before submit).** Faithful streaming pass on the
winning arm: TTFT/TTLT must still ≈ 371 ms and the accuracy pass must fit 15000 s. Fix-2 (incremental
feature cache) is proven-safe insurance here. Only then, official `evaluate.sh`, then submit.

**Pre-registered success criterion (write it before the pod, per the house rule):**
> Submit parakeet Arm B IFF, on the REAL `local_decode.py`+`evaluate.sh` Dev_diag severe gate,
> severe CER ≤ 24.0% AND mean(TTFT,TTLT) ≤ 420 ms. Otherwise do not submit; ship the banked zipformer.

---

## 7. Direct answers to Q's questions

- **How do we improve the model?** Stop fine-tuning the encoder alone. The blank decision lives in
  the **joint + prediction network**, which we left frozen. Unfreeze it (H1) and feed it data that
  over-represents the failure (H2/H3). That's ~90% of the available CER.
- **More data?** No easy source of *real* new dysarthric audio (fixed challenge set). But we're
  under-using what we have: **re-weight toward the low-energy / ALS / short-command slice** (H3) —
  free, and it's exactly where the model fails.
- **Synthetic data?** **Yes, but the targeted kind:** augment *real* dysarthric utterances into the
  quiet-onset/low-SNR failure region (H2) — the train-time dual of the (dead) inference gain fix.
  **Not** dysarthria-TTS/voice-conversion yet (H7): high mismatch risk, defer until H1+H2+H3 plateau.
- **Why would joint unfreeze work?** Because the empties are **"confident blank"** from a joint
  network whose blank-vs-emit boundary was calibrated on clean LibriSpeech and never retrained.
  Unfreezing it makes the blank logit trainable so the RNN-T loss can push P(blank) down on
  dysarthric onsets — something encoder-only FT can only do indirectly, across a frozen boundary.
  Literature-confirmed (Takahashi Table 2; SAPC1 winner did full FT).
- **What exactly within the model is going wrong?** The **frozen joint network over-emits blank** on
  low-energy/short/atypical-articulation input → all-blank alignment → empty transcript. The same
  bias, sub-clinically, produces the deletion-heavy (del:sub = 2.2) non-empty errors.
- **What part of the data is tricky?** **Short, quiet-onset, command/wake-word utterances from
  low-vocal-energy speakers (ALS 24.5% / Down 7.5% empty), concentrated in one speaker (22/48).**
  Energy/SNR gradient, not a raw "severity" gradient (Parkinson is severe but 0% empty).

---

## 7a. Literature verification (2026-07-28) — what held, what I had to correct

Verified the load-bearing claims against primary sources rather than our own `[V-lit]` notes.
Verdicts: **CONFIRMED** / **REFINED** (true but I'd mis-stated it) / **UNVERIFIED** (couldn't
source; flag it).

1. **RNN-T "confident blank" / deletion bias is a real, named failure mode — CONFIRMED, and
   sharper than I had it.** It's not only an acoustic/joint effect: the **prediction network
   over-relies on consecutive-word dependencies from training, driving deletions on OOD / less-common
   phrases**, and the documented fix is to *suppress incorrect blanks and promote word tokens*
   (Ghodsi/Kim et al., "RNN-T Fails to Generalize to Out-Domain Audio", Interspeech 2021, arXiv
   2108.10752). Deletion/blank over-emission is important enough to warrant dedicated *loss-lattice*
   fixes (skip-frame transitions, arXiv 2504.06963). → **Strengthens H1's scope: unfreeze the
   prediction net, not just the joint** — our wake-words (`alexa`,`cortana`) are exactly the OOD
   phrases a LibriSpeech prediction net has no prior for.

2. **"Full FT strictly > freezing the joint" — REFINED (I overstated it).** The literature does NOT
   say "always full FT." Low-resource RNN-T work often **freezes the *encoder* and trains the
   decoder+joint**, and reports that beats full FT in some regimes (synthetic-data / masked-encoder
   settings). The robust, defensible claim is narrower: **the decoder/joint is the productive
   adaptation locus, and our Arm A — train encoder, *freeze* joint+prediction — is the configuration
   least supported by the field.** Consequence for the plan: **add an Arm B variant that freezes the
   encoder and trains only joint+prediction** (cheaper, more collapse-resistant) as a sibling to the
   graded full-unfreeze. Don't assume full FT is the answer; test both.

3. **Targeted augmentation helps dysarthric ASR — CONFIRMED with numbers.** Tempo perturbation
   **+4.24%** abs, speed perturbation **+2%** abs WER over healthy-only training; speed perturbation
   up to **2.92% abs / 9.3% relative** WER reduction on a 16-speaker dysarthric test. **Caveat
   (UNVERIFIED specifically):** speed/tempo/SpecAugment are attested; the *low-energy / quiet-onset
   gain-reduction* augmentation at H2's core is my **reasoned extension from our root cause**, not a
   directly-cited result. Treat it as a hypothesis to A/B, not a proven recipe.

4. **Voice-conversion augmentation > speed/tempo — CONFIRMED, upgrades H7.** VC with speaker+prosody
   conversion "significantly outperforms conventional speed/tempo perturbation" (arXiv 2505.14874;
   "Low-Burden Augmentation via Zero-Shot Voice Cloning", arXiv 2506.19823). → H7 is **evidence-backed**,
   not speculative; promote it from "deferred" to "the escalation if H2 plateaus."

5. **ALS → hypophonia / reduced vocal intensity — STRONGLY CONFIRMED, with one honest nuance.**
   Dysarthria in ~80% of ALS; decreased intensity / reduced volume / hypophonia are hallmark. Fits
   ALS = 24.5% empty. **Nuance I won't paper over:** Parkinson's is *also* clinically associated with
   hypophonia, yet 0% empty in our slice. So the driver is more precisely **onset energy / SNR of the
   specific utterances**, not a clean "disease → mean loudness" story. The audio statistics (§4) are
   the ground truth; the clinical label is corroborating colour, not the mechanism.

6. **SAPC1 winner recipe (Takahashi) — CONFIRMED.** Parakeet-TDT-1.1B, **#1 at 8.11 WER** (and top
   SemScore), **data augmentation for generalization**, **4-checkpoint weight averaging** (lowest-Dev-
   WER ckpts, ESPnet-style), **mAES** decoding (afforded reducing beam 30→10 at comparable accuracy).
   The specific *freeze-ablation table* our notes cite as "Table 2" I could **not** re-extract from the
   FlateDecode-corrupted PDF — so treat the exact ablation as **UNVERIFIED**; the recipe elements above
   are independently confirmed.

7. **NEW RISK found — TDT head stagnation on fine-tune (NeMo issue #14140).** Users report the
   **TDT/transducer head plateauing while the CTC/encoder path keeps improving**, driven by
   **tokenizer mismatch + insufficient LR/warmup** on the head. For us: we **keep the English
   tokenizer (no mismatch — favourable)**, but this confirms Arm B is *not a free win* — the head can
   fail to move without careful, higher LR + warmup on the joint/prediction. Mitigation folds into the
   H1 recipe (differential LR, warmup on the head, early-stop on dev empties).

**Net effect on the plan:** the core thesis survives — the empties are a decoder-locus pathology and
the fix is to train the decoder/joint(+prediction). Three concrete edits: (a) H1 scope explicitly
includes the **prediction network**; (b) add an **encoder-frozen / joint-only** Arm B variant; (c)
promote **VC augmentation (H7)** to a first-class escalation. Two honest gaps: the quiet-onset
augmentation and Takahashi's exact freeze-ablation are not directly sourced.

**Sources:** Takahashi'25 (isca-archive/interspeech_2025/takahashi25) · SAP Challenge overview (arXiv
2507.22047) · RNN-T OOD deletion/blank (arXiv 2108.10752) · RNN-T noisy-target losses (arXiv
2504.06963) · dysarthric speed/tempo aug (Springer s00034-022-02156-7; CUHK arXiv 2201.05845) · VC aug
(arXiv 2505.14874, 2506.19823) · ALS dysarthria (NCBI PMC12293193; UNM dys_ALS review) · TDT head
stagnation (NeMo GH #14140).

---

## 7b. Second research iteration (2026-07-28) — mechanism resolved, two of my own levers killed

I chased four new levers. Two died on contact with evidence (good — that's the point), the
mechanism resolved cleanly, and the real payoff was hardening the Arm B recipe.

**Mechanism RESOLVED — plain-RNNT confident-blank, not TDT, not EOU.**
- Our checkpoint `parakeet_realtime_eou_120m-v1` is **FastConformer-RNNT** (17 enc layers, att
  [70,1], RNNT decoder), **not** a Token-and-Duration Transducer (HF model card + aimodels.fyi). So
  my "TDT duration-head over-skips the quiet onset" idea (§ this iteration) **does not apply — killed.**
- It also does joint ASR + **`<EOU>`** detection. But the EOU-misfire hypothesis was already probed
  and returned **0 eou-only empties** (`offline_recovery.json`), so the empties are genuine blank, not
  EOU truncation. Confirmed dead.
- **Verdict:** the empties are the **textbook RNN-T "confident blank" / deletion failure on OOD
  low-energy input** — the joint over-emits blank and the prediction net has no prior to pull a token
  (arXiv 2108.10752). Our wake-words (`alexa`,`cortana`) are literally the OOD phrases that paper
  describes. The root cause is now well-identified and literature-matched, not speculative.

**The "cheap no-pod decode-time fix" — mostly ILLUSORY for our integration (honest correction to my
prior message).** I over-hoped here. From NeMo's decoding source:
- Core RNNT **greedy is pure `argmax` — no temperature, no blank penalty, no logit-bias hook.**
- We are **pinned to `greedy_batch`** by the NeMo 2.7.3 cache-aware streaming `merge_()` crash.
- `max_symbols_per_step` (default 10) bounds *emissions per frame* — irrelevant to empties (empties
  emit **zero**, not too many).
- → Blank-penalty / temperature (the documented empty-mitigation knob; Takahashi used temp ≈1.2–1.3)
  is a **real lever but an integration task behind the same greedy-pinning gate as beam**, NOT a free
  flag flip. A custom joint-logit hook in our `model.py` is *conceivable* (we own the file) but the
  cache-aware streaming API returns hypotheses, not per-step logits → it'd mean hand-rolling the
  streaming greedy loop = high-risk, not cheap. **Decode-time blank suppression does not displace Arm
  B; it's a secondary, integration-gated experiment.**

**CTC-head rescue (lever D) — KILLED for our checkpoint.** FastConformer-RNNT has **no CTC head**
(that's the separate hybrid `parakeet-tdt_ctc` family). A CTC-fallback-on-empty would require swapping
to a different base model = a distinct, larger bet. Off the table for the parakeet line as-is.

**THE REAL PAYOFF — Arm B recipe hardeners (NeMo fine-tuning best practices, docs.nvidia.com).**
These cost nothing to adopt and directly de-risk the pod:
1. **Override the Noam (warmup+decay) scheduler with constant or cosine-annealing.** NeMo explicitly
   warns the Noam warmup can **reset LR high and destabilize pretrained transducer weights** — this is
   a **plausible mechanism for our Nemotron collapse scar** (memory `h7-fix-plan`: "finetuned collapse
   was weights/training-specific"). *Check our old Nemotron train config for a Noam warmup; if present,
   the collapse may have been a scheduler artifact, not fundamental.* Treat as a hypothesis to verify
   against the old config, not a settled cause.
2. **LR 1e-4 → 1e-5, start low**; differential LR (head > encoder) still applies.
3. **`init_from_nemo_model_include` / `init_from_nemo_model_exclude`** — the exact NeMo mechanism to
   build the **encoder-frozen, train-joint+prediction** Arm B variant (§7a edit #2) cleanly.
4. **SpecAugment during FT + Lhotse dynamic batching** (both used by the SAPC1 winner).
5. **TDT-head-stagnation risk (NeMo #14140) is LOWER for us than feared** — that case was a *tokenizer
   mismatch* (new 1024-BPE for Persian) on a *TDT* head; we keep the English tokenizer and a plain
   RNNT head, so the stagnation driver largely doesn't apply. Still: warmup/LR care per (1)–(2).

**Net:** this iteration *subtracted* two speculative levers and one over-hoped "free fix," and
*added* a concrete, literature-grounded recipe that de-risks the one bet that matters (Arm B). The
plan is now leaner and the collapse risk has a named, testable mitigation.

**Sources (iteration 2):** FastConformer-RNNT arch (HF `parakeet_realtime_eou_120m-v1` card;
aimodels.fyi) · TDT mechanism (arXiv 2304.06795; Speechmatics TDT explainer) · multi-blank precursor
(arXiv 2211.03541) · RNNT blank penalty / spiky blanks (arXiv 2211.00896; icefall beam_search) · NeMo
greedy = pure argmax (NeMo `rnnt_greedy_decoding.py`) · NeMo fine-tuning best practices
(docs.nvidia.com/nemo .../asr/fine_tuning) · TDT-head stagnation (NeMo GH #14140).

---

## 7c. Third research iteration (2026-07-28) — the empty-slice levers, thread (b)

Thread (b) asked two coupled questions: *how to augment for the quiet-onset/empty slice*, and
*how to weight the loss toward emission over blank*. The loss half returned the strongest new
lever of the whole investigation; the augmentation half returned an honest **negative** that
reshapes H2.

### (b-1) LOSS — FastEmit is the on-target, NeMo-native, double-Pareto lever `[V]`

- **Mechanism (verified, arXiv 2010.11148 Eq. 11–12):** FastEmit scales the gradient to the
  **token** posterior by **(1+λ)** and leaves the **blank** gradient unchanged — "a higher learning
  rate for emissions." This is the *direct* train-time counter-pressure to the confident-blank
  pathology in §3 that produces our empties. It is a *sequence-level* regularizer (acts on the
  forward-backward), so unlike per-frame blank penalties it does not fight the transducer objective.
- **NeMo-native, zero custom code (verified):** `loss.warprnnt_numba_kwargs.fastemit_lambda`
  (default `0.0`). NeMo docs recommend `[1e-4, 1e-2]`, start `1e-3`. This is our *exact* toolchain —
  a training-config flag, not a fork.
- **Orthogonal to the streaming pin (KEY):** FastEmit is a **training-time** loss term. It is
  completely independent of the decode-time `greedy_batch` pinning that gated the §7b blank-penalty
  idea. So the lever §7b demoted as "not free" has a *training-side twin that IS free* — and it
  folds into the Arm B retrain at ~zero marginal cost (same pod, same run).
- **λ curve (verified, Table 2):** WER *improves* at λ ≈ 0.004–0.01 (LibriSpeech best-case
  4.4→3.1 WER, P90 latency 210→30 ms), starts regressing by λ = 0.02, back to baseline by 0.04.
  → **sweep {0.003, 0.005, 0.01}; never crank.** This regression curve is the empirical face of
  pre-registered **open-risk #2 (over-emission backfire)** — FastEmit is the *controlled* version
  of "push P(blank) down," with a known safe band.
- **Strategic uniqueness:** it is the **only lever in the portfolio that moves BOTH Pareto axes the
  right way at once** — fewer all-blank outputs (lower CER) *and* earlier first emission (lower
  TTFT). For a submission whose entire thesis is the ~371 ms latency corner, a CER fix that also
  *tightens* latency is maximally aligned. Everything else (beam, unfreeze) trades latency for CER.
- **Honest caveat `[U]`:** FastEmit was designed/validated for *emission delay* on clean speech
  where the token is eventually emitted; it is not literature-proven to recover *full* empties on
  OOD low-energy input. Mechanistically on-target (it up-weights exactly the emissions we're
  losing), but it is a **testable bet, not a guarantee**. Gate it on dev empty-count, not faith.
- **Synthesis with Arm B:** when Arm B unfreezes the joint+prediction (the network that *owns* the
  blank/token boundary), `fastemit_lambda ≈ 0.005` is the mechanistically-correct regularizer to
  shape that newly-trainable boundary toward emission. Arm B without FastEmit relearns the boundary
  blindly; Arm B *with* FastEmit relearns it with a prior against blank collapse. **They are a pair.**

### (b-2) AUGMENTATION — an honest negative that reframes H2 `[V]`

- **The dysarthric-ASR augmentation field does NOT address energy/gain.** Two independent surveys
  (arXiv 2401.00662; CUHK disordered-speech augmentation work) attack only **speaking-rate mismatch**
  (speed `{0.9,1.0,1.1}`; speaker-dependent factor = control_dur / dysarthric_dur) and **spectral
  distortion** (DCGAN / spectral-basis GAN / VC; best gains +2–3% abs, combined 16.5% WER on
  UASpeech). **Neither proposes volume/gain/energy/low-SNR perturbation.** This *confirms twice over*
  the H2 gap I flagged in §7a: my "quiet-onset / low-energy augmentation" is **my inference, not a
  dysarthric-literature-attested recipe.**
- **Reframe (this is the useful part):** our empties are an **energy + OOD** failure, and the
  dysarthric field under-serves exactly that (it chases *rate* and *spectral*, not *loudness*). So
  the energy angle must be borrowed from **general ASR robustness**, where random gain / additive
  noise (MUSAN) / SpecAugment are standard — while the **loss side (FastEmit) does the heavy lifting
  against blank-collapse.** Net: H2's *loss* half got much stronger; H2's *augmentation* half should
  be down-scoped to generic gain+noise robustness, and the empty fix's center of gravity moves from
  "bespoke augmentation" to "FastEmit + generic energy robustness."
- **Internal-LM deletion bias (verified, supports failure-targeted resampling):** RNN-T predictors
  over-rely on consecutive-word dependencies → **elevated deletion on rare/short/OOD phrases**
  (multiple sources; SegAug reduces it by lowering sentence-level semantics). Our empties are
  disproportionately **short wake-words/commands** (`alexa`, `cortana`, `hey siri`) — textbook cases
  of this bias. → **oversampling the short-command slice** (and keeping utterances short/low-context
  in a fraction of augmented data) is a literature-grounded way to weaken the internal LM's
  blank-on-rare-phrase reflex. This upgrades the "severity/energy-weighted resampling" idea from
  intuition to mechanism.

**Net (iteration 3):** thread (b) *added* one high-value, verified, toolchain-native lever
(FastEmit, λ≈0.005, folds into Arm B for free and helps both Pareto axes) and *corrected* H2 —
the augmentation half is not dysarthric-literature-backed for our energy failure, so its weight
shifts to generic gain/noise robustness plus short-command oversampling, with the loss doing the
real work. The Arm B run-book now has a concrete regularizer, not just an unfreeze.

**Sources (iteration 3):** FastEmit mechanism + λ curve (arXiv 2010.11148, Eq. 11–12 & Table 2) ·
NeMo `fastemit_lambda` config (docs.nvidia.com/nemo .../asr/configs; conformer_transducer_bpe.yaml) ·
dysarthric augmentation recipes/gaps (arXiv 2401.00662; CUHK 2201.05845) · RNN-T internal-LM /
deletion on OOD short phrases + SegAug (WebSearch hits; arXiv 2504.06963 Star-Transducer;
token-weighted RNN-T 2406.18108).

---

## 8. Open risks / what could make this fail

1. **Nemotron collapse scar (FACT):** aggressive unfreeze on a small in-domain set has collapsed a
   model for us before. → graded unfreeze, differential LR, ckpt-averaging, early-stop on dev empties.
2. **Over-emission backfire (THEORY):** pushing P(blank) down too hard trades empties for insertions
   → non-empty CER up. → monitor ins-rate; loss-shaping only if plain unfreeze under-delivers.
3. **Proxy/real divergence (SCAR, house rule):** every number above except the official-scorer ones is
   proxy. GATE 0 and the pre-registered criterion exist precisely to stop a proxy-driven false submit.
4. **Latency erosion:** any decode change (H4) can flicker partials and lose the 371 ms corner — the
   corner is the *reason parakeet exists*. Never trade it for a CER squeeze without measuring both.
5. **Fixed compute/budget:** GPU spend needs explicit Q go (cost gate). Everything here is staged to
   one pod session with kill-gates so we stop the moment the decision metric is known.
```
