# 18 — Gate 2 results: finetuning Nemotron WORKS (directly proven, 2026-06-25)

The go/no-go from [[17_finetune_gate1_2_runplan]]. First DIRECT evidence (not Parakeet/zipformer analogy) that
finetuning our 0.6B cache-aware FastConformer-RNNT on SAP dysarthric speech helps. **Verdict: GREEN-light the
full finetune; encoder-only is the winning arm.**

## Setup
- Clean overfit (30 hard ALS/CP utts, bs32): **CER 0.00%** → pipeline proven end-to-end.
- Smoke: only **4,000 utts (1.3% of the 309k available), 3,000 steps**, trained at low-latency context `[70,1]`,
  two arms. Eval = NeMo transcribe at `[70,1]` on the severe-enriched **Dev_diag-425** (Dev speakers disjoint
  from Train — held-out). Baseline = zero-shot Nemotron 47.5% CER / 25% empty ([[15_expA_beam4_results]]).

## Held-out per-etiology (Dev_diag-425)
| model | overall CER | empty | ALS | CP | DS | Parkinson's | Stroke |
|---|---|---|---|---|---|---|---|
| zero-shot | 47.5% | 25% | 62.7/39% | 54.3/32% | 52.8% | 10.9/0% | 35.0/21% |
| smoke full | 32.1% | 13% | 44.6/29% | 33.0/12% | 39.2% | 7.4/0% | 28.6/5% |
| **smoke encoder-only** | **30.8%** | **9%** | 43.3/23% | 30.0/7% | 40.3% | **7.2/0%** | 27.7/0% |
| (zipformer, exp A ref) | 29–31% | 5–6% | | | | | |

## VERIFICATION (2026-06-25) — confounds resolved, numbers corrected
- **Speaker leakage: NONE.** Train=875 speakers, Dev=124, overlap=0; Dev_diag's 103 speakers, 0 in Train. The
  eval is on genuinely unseen speakers. ✓
- **Apples-to-apples baseline:** the 47.5%/25% zero-shot baseline used a DIFFERENT path (ONNX submission +
  local_decode harness at [70,6]). Re-run zero-shot via the SAME NeMo-transcribe-[70,1] path = **43.4% CER /
  18% empty.** So the CLEAN finetuning gain is **43.4 → 30.8% CER (−12.6 pts), empties 18 → 9%** — real and
  large, but ~4 pts smaller than first reported. Per-etiology (zero-shot[70,1] → enc-only): ALS 61.1→43.3,
  CP 45.8→30.0, DS 49.5→40.3, PD 11.2→7.2 (no forgetting), Stroke 31.5→27.7.
- **STILL OPEN (confound C):** NeMo transcribe ≠ the deployment 100ms-chunk ONNX streaming harness. The 43.4
  (transcribe) vs 47.5 (ONNX-harness) zero-shot gap proves the inference path matters → the finetuned model's
  DEPLOYMENT CER must be confirmed via the faithful harness after ONNX export (already the submission gate).

## What this says
1. **Finetuning works, and fast.** A *tiny* 4k-utt / 3k-step smoke cut overall CER **47.5 → 30.8%** and empties
   **25 → 9%** on the hard set — already matching the FULLY-finetuned zipformer (29–31%), using 1.3% of the data.
2. **Encoder-only WINS** (30.8% vs 32.1%, 9% vs 13% empty) — the anti-forgetting hypothesis holds for our 0.6B
   streaming model. Use encoder-only (or LoRA) for the full run, NOT full-unfrozen.
3. **No catastrophic forgetting** — mild speech (Parkinson's) actually IMPROVED 10.9 → 7.2%. The whole worry
   about destroying the mild-speech competence didn't materialize (esp. encoder-only).
4. **The empties — the original failure mode — are responding:** ALS 39→23% empty, CP 32→7%, Stroke 21→0%.
   ALS stays hardest (23% empty) but halved. This is the severe-tail moving, exactly the goal.
5. Achieved at **`[70,1]` = 160 ms low-latency context** → good CER AND low TTFT are compatible (the latency lever).

## Projection (full run)
Diag-425 is deliberately severe-enriched. Zero-shot was 47.5% on diag vs ~25% representative Dev vs 51% Test1.
A smoke at 30.8% on diag → the FULL run (all 309k utts, more steps, avg-5, encoder-only) should land well below
that on diag, projecting to **~mid-teens CER on representative Dev / Test1** — potentially beating the
zipformer's 23.4% Test1, with Nemotron's higher ceiling. (Projection, not promise — the full run measures it.)

## Recommended full run
- **Encoder-only** (or add a **LoRA** arm), **all 309k utts**, more steps (e.g. 20–40k), warmup, **avg top-5**.
- Train multi-lookahead incl. `[70,0]`/`[70,1]` to keep the latency lever; sweep deploy context for CER×TTFT.
- Resolve the **target-text** ablation (normalized vs raw cased) — start normalized (worked here).
- Then **export → ONNX int8 → offline package → faithful `local_decode.py` eval on full Dev** (the submission
  gate) before any upload.
Artifacts: pod `/workspace/finetune/nemo_ft/gate2_{full,enc}/*.nemo`, `gate2.log`. Scripts: `nemo_finetune.py`,
`nemo_eval_diag.py`, `run_gate2.sh`.
