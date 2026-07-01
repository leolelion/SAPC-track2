# 37 — Test1 gap diagnosis: severity-fragility + quiet-audio empties; cheap fixes ruled out (2026-06-29)

> Triggered by the int8 Test1 result (CER 27.97% / WER 37.71%), which is WORSE than the A1 zipformer
> baseline (23.44%). This note is the post-Test1 big-picture review + the cheap-lever experiments that
> gate the v2 GPU decision. All numbers via the faithful `local_decode.py` + official/bootstrap scorers
> on the restarted pod (Xeon 8568Y+, the Codabench-comparable CPU). Pod stopped after. Artifacts in
> `~/Downloads/dev2k_int8_eval_results/` (+ `severe_empties_audit/`).

## 1. The fact that reframes the line of work
Nemotron-0.6B encoder-only int8 LOSES to our own A1 baseline on the only metric that counts:

| system | Test1 CER | Dev→Test |
|---|---:|---|
| A1 zipformer-66M | **23.44%** | +1.8 (21.62%→23.44%) — healthy |
| Nemotron enc-only int8 | **27.97%** | +18 (9.91% balanced-2k → 27.97%) — broken |

Doc 29's big-picture audit projected Test1 at "low-to-mid teens" from Dev; it was off by ~15 pts.
The faithful-harness work validated the **pipeline**, not the **generalization**.

## 2. Disambiguation: within-etiology SEVERITY, not etiology mix (dev_eval_2k)
Decoded the etiology-balanced 2k Dev ruler (400 each ALS/CP/DS/PD/Stroke), official scorer:
**CER 9.91%** (per-eti: PD 3.65, ALS 8.87, Stroke 12.07, DS 14.17, CP 19.60). Balancing etiologies
barely moved it from the easy Dev-500 (8.76%) → the Dev→Test collapse is **not** etiology mix. It is
**within-etiology severity**: severe speakers (Dev_diag ALS/DS ~34%) are where the model is no better
than A1, and Test1 (27.97%) sits next to Dev_diag (24.85%) ⇒ Test1 is severe-speaker-heavy. Nemotron's
big Dev lead is entirely on easy/moderate speakers and evaporates on severe ones. Mechanism: a large
pretrained encoder, encoder-only finetuned on 683 h of a narrow distribution, overfits its
representations to the train/dev speaker pool (doc 14 §0/§5 predicted exactly this).

## 3. Empties decomposition on Dev_diag (severe set, int8)
Re-decoded Dev_diag faithfully; utt-averaged CER so each empty hyp = exactly 1.0:

| | CER | empties | non-empty CER |
|---|---:|---:|---:|
| overall | 23.58% | 31/425 (7.3%) | 17.57% |
| **ALS** | 33.52% | **22/110 (20%)** | 16.89% |
| Down Syndrome | 33.39% | 2/67 (3%) | **31.34%** |
| Cerebral Palsy | 21.55% | 7/152 (4.6%) | 17.76% |
| Stroke | 25.09% | 0/19 | 25.09% |
| Parkinson's | 4.49% | 0/77 | 4.49% |

Two distinct failure modes: **ALS = empties** (model emits blank at every frame → 100% CER each; ~7 pts
of the severe CER, concentrated in ALS); **DS = wrong words** (3% empty, 31% non-empty CER).

## 4. Empties are a QUIET-AUDIO artifact (root cause found)
- `localmel.py` uses `normalize=NA` — **no per-feature or gain normalization** → absolute input energy
  flows into the log-mel magnitude.
- The 31 empty utts are systematically quiet: RMS median **0.0246** vs non-empty **0.0594** (~2.4×),
  with a tail of extremely quiet ones (RMS 0.0004–0.006). ALS speech is low-energy/breathy.
- ⇒ quiet audio → weak log-mel → encoder under-activates → frozen base-Nemotron joint emits blank.

## 5. Cheap inference fixes — BOTH ruled out
- **Decode-side blank-penalty fallback** (re-decode empties with blank suppressed): eliminates empties
  but output is **garbage** — every utt yields the SAME input-independent string ("< >What< >Explain's
  one of your steps your steps…"). When blank is suppressed the decode is dominated by the frozen
  decoder/joint **LM prior**, not the acoustics. CER stays 100%. DEAD.
- **Input gain normalization** (real mechanism): on the 31 empties, peak-norm recovered 8/31 and some
  PERFECTLY (ALS "do i have anything scheduled for friday" → exact). But the **net** on full Dev_diag
  with conditional norm (boost only RMS<0.05) is **23.58% → 23.35% (~0.2 pt)** and it **trades
  etiologies** (ALS −1.4, but DS +1.7). Not a meaningful win; very-quiet utts stay empty.

Neither closes the 4.5-pt gap to A1. **The severe tail requires a training fix, not an inference trick.**

## 6. Literature caveat that now binds
Severity-finetuning gains in the lit (Sapkota ~32% rel; Geng 56–61% rel) are on TORGO/UASpeech —
**closed-speaker / speaker-matched** corpora. SAPC2 Test1 is **held-out speakers**, exactly the
generalization that just failed. Severity-oversampling can lift Dev_diag while NOT moving Test1 — the
same Dev-overfit trap one level down. So severity sampling must not be the lead lever and must be gated
on held-out-severe, not in-distribution-severe.

## 7. Conclusion → v2 design (if Q approves the GPU spend)
The diagnosis points to ONE structural intervention, not the encoder-only/severity-only plan:
1. **Unfreeze the joint (± prednet) at low LR** — the frozen joint's blank propensity is the binding
   constraint on the ALS empties (cheap fixes can't touch it; §5). This is the highest-EV lever.
2. **Input gain augmentation / normalization in TRAINING** (novel, motivated by §4) — train the encoder
   to be robust across the energy range of ALS speech, fixing empties at the root instead of inference.
3. **Severity-aware sampling as a guarded ride-along** (not lead; §6), with a PD/mild forgetting probe
   and a DS-regression guard (DS is wrong-words, the hardest residual).
- Gate on Dev_diag severe tail AND empty-rate AND a clean-speech forgetting probe; faithful harness only.
- Eyes open: the bar is **beating A1's 23.44% on Test1**, and Dev just proved it doesn't predict Test —
  so even a Dev_diag win may not transfer. A1 remains the safe Pareto entry.

## 9. CORRECTION (2026-06-30, independent review → research/40)
The §7 conclusion ("unfreeze the joint — the frozen joint's blank propensity is THE binding constraint") is
DOWNGRADED. Our own §5 recovered empties (8/31, some perfectly) with the joint STILL FROZEN, via input gain
alone → the empties are (at least co-)caused by **encoder energy-sensitivity**, not solely a joint cap. And
§2's overfitting diagnosis means joint-unfreeze ADDS trainable capacity in the wrong direction. The empties
attribution is now ≥3 live hypotheses (encoder energy robustness / decode-time ILM blank-bias / joint
adaptation), and the literature-correct adaptation for our small atypical corpus is **PEFT**, not unfreeze.
See research/40 for the corrected, EV-ranked plan.

## 8. Strategic
We are currently behind our own baseline on Test1. Doc 14 §10 called the zipformer track the higher-EV
near-term move and Nemotron the higher-ceiling gamble; Test1 says the gamble hasn't paid yet. The v2
structural finetune is a real, uncertain GPU bet — to be decided explicitly, not assumed.
