# Axis 6 — Fine-tuning Recipe: literature best-practice vs. what we did

Goal: maximize dysarthric streaming CER via fine-tuning, generalizing to **unseen, speaker-disjoint,
"unshared"-text** test speakers. Sources: SAPC1 winner (Takahashi'25), icefall finetune docs, dysarthric
aug literature (research/01). Tags: `[V-lit]`/`[V]`/`[K]`/`[?]`.

## What the prior run did (verified) vs. what the literature says
| Lever | Prior run `[V]` | Literature best practice | Gap |
|---|---|---|---|
| Epochs | ~2 (loss→0.035, text-overfit) | SAPC1 winner 10–20; icefall finetune ~10–20 | **Under-trained** |
| LR | 0.0005 | icefall finetune ≈ 0.0045 (≈1/10 base 0.045) | likely too low *and* too few epochs |
| Speed perturb | none seen | 3-way 0.9/1.0/1.1 (winner + dysarthric lit: speaking-rate varies a lot) | **Missing — high value** |
| SpecAugment | (icefall default?) | yes (winner); robustness/anti-overfit | confirm on |
| MUSAN noise | none seen | optional; modest gain | optional |
| Anti-forget | none | icefall `--use-mux` (mix base LibriSpeech) — matters since test text is "unshared" | **Missing** |
| Long-audio | none seen | forced-align **sentence segmentation** (winner) — SAP has long full-text recordings | **Missing** |
| Ckpt averaging | unclear | avg best 4–10 (winner, ESPnet-style) | confirm |
| Model size | zipformer-M 66M | M proven; L higher ceiling, still CPU-OK on 24 cores | try L later |

## The recipe to run (evidence-derived, port the SAPC1 winner to streaming)
1. **Full FT** (no frozen modules — winner showed freezing hurt). Start from LibriSpeech epoch-30
   (`epoch-0.pt` here). Keep streaming/causal config identical → latency profile unchanged.
2. **Augmentation**: 3-way speed perturbation (0.9/1.0/1.1) + SpecAugment on. (MUSAN optional.)
3. **`--use-mux`**: interleave original LibriSpeech cuts to prevent catastrophic forgetting / text-overfit
   (critical because test is "unshared" — don't memorize SAP transcripts).
4. **Long-audio handling**: forced-align segmentation of >~30–45 s recordings into sentence cuts.
5. **LR/epochs**: base-lr ≈ 0.0045, 15–20 epochs, lr-epochs ~6, warmup; **early-stop / select on a
   speaker-disjoint internal val** (NOT on Dev text).
6. **Checkpoint averaging**: average the best 4–10 by valid loss before export.
7. **Decode**: greedy for streaming (fast TTLT); modified_beam_search where budget allows; temp ~1.2 (winner).

## Continuous-improvement loop (how we "keep improving")
Each finetune = one experiment in the ledger with: config snapshot, git hash, **CER (official min-two-refs)
+ TTFT/TTLT** on a fixed speaker-disjoint Dev subset, RTF. Compare to prior best; keep/reject; one next
question. Ordered ablations (cheap→expensive):
- A0: **proper eval of the CURRENT finetuned model** (true baseline CER — the phase4b number is bug-contaminated).
- A1: + augmentation (speed-perturb + SpecAugment) + more epochs (15–20) + use-mux.  ← biggest expected gain
- A2: + forced-align long-audio segmentation.
- A3: + checkpoint averaging; LR sweep.
- A4: zipformer-L (ceiling) if CPU/RTF still comfortable.
- A5: streaming-boundary fix (warmup/tail) — separate from FT, helps the truncation seen in testing.
- (later) per-etiology error analysis → targeted aug/synthetic only where it helps (CP/DS worst).

## Open
- Exact aug flags in the prior run (speed/spec/musan) — `[?]` confirm from finetune.py args if needed.
- Best start checkpoint (epoch-0 LibriSpeech vs a stronger multi-domain zipformer e.g. Kroko; check license).
- Whether to retrain BPE on SAP text (CER is char-level → low stakes; likely keep BPE-500).
