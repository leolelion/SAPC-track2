# 15 — exp A (finetuning rescue test) + beam-4 gate — RESULTS (2026-06-25)

Run via the faithful harness on the 425-utt severe-enriched diag set + representative Dev_rand300.
Answers the gating question from [[14_nemotron_finetune_plan]]: **will finetuning help the failures Nemotron
empties on?** And gates the zipformer beam-4 ship.

## exp A — does a SAP-finetuned model rescue zero-shot Nemotron's failures? YES.
Same 425 utts (enriched with the 7 worst speakers + duration bins), official normalizer, two-ref CER:
| model | meanCER | empty | <20% | >80% near-fail |
|---|---|---|---|---|
| Nemotron (zero-shot) | **47.5%** | 25% (107) | 164 | 144 |
| Zipformer A1 (SAP-finetuned), greedy | **31.3%** | 6% (26) | 197 | 44 |
| Zipformer A1, beam-4 | **29.1%** | 5% (23) | 215 | 42 |

**Head-to-head on the SAME utterances** (Nemo catastrophic failures = >0.8 CER or empty = **140/425**):
- **74 (53%) rescued to <0.5 CER by the finetuned model → DOMAIN failure that adaptation fixes.**
- ~33 partially improved (0.5–0.8).
- **33 (24%) still fail for both → genuinely hard audio (the severe-tail ceiling the literature warned of).**

=> Finetuning is empirically validated: a SAP-adapted model **halves the catastrophic-failure set, cuts mean
CER 47.5→29–31%, and drops empties 25%→5–6%** on the exact distribution that breaks zero-shot Nemotron. The
failures are predominantly domain, not irreducible. The ~24% both-fail tail is the real floor.

## Complementarity (interesting, for later)
By duration, the models trade places: on **>30 s** clips Nemotron is *better* (17.0% vs zipformer 24.5%) — long
context suits its clean-English prior — while on **short severe** clips the zipformer crushes it. Suggests
ensemble/routing value, but hard to exploit under streaming; note and move on.

## beam-4 gate — GREEN (ship it)
Through the faithful harness:
- **Dev_rand300 (representative):** greedy p50 CER 15.2 → **beam-4 13.6**; near-fails(>80%) 9 → **3**.
- **diag-425 (hard):** greedy 31.26% → **beam-4 29.10%** (−2.16 CER), fewer empties.

Beam-4 beats greedy on both the representative and hard sets, reduces catastrophic fails, same weights, same
streaming. **Validated through the real harness → cleared to upload** (gate per [[validate-against-real-harness]]).

## Decisions this unlocks
1. **Ship zipformer beam-4** — bankable, gated green. (`/Users/o/Downloads/submission_a1_beam4.zip`, SHA
   `f8ae6d87…`.) Needs your upload.
2. **Finetuning Nemotron is justified** — the rescue test proves adaptation recovers most failures, and
   Nemotron's higher mild-speech ceiling (median ~7%) means a finetuned Nemotron could beat the zipformer's
   ~23% Test1. But the zipformer is *already* a strong adapted baseline (29–31% on this hard set), so the bar
   to clear is real. Highest-EV sequence: ship beam-4 + add LM fusion to the zipformer now (cheap, proven),
   AND launch the Nemotron finetune (plan v2 ready, de-risked) for the higher ceiling.
3. The ~24% both-fail tail won't be solved by finetuning alone → that's where severity-aware sampling + TTS
   augmentation (plan §3–4) and possibly the long-context complementarity matter.

Artifacts: pod `/workspace/finetune/eval/expA_beam4/` (diag_nemo, diag_zf_greedy, diag_zf_beam4,
rand300_zf_*). Tooling: `scripts/run_expA_beam4.sh`.
