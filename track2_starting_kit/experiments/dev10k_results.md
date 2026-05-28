# Dev_10k validation — Nemotron int8-static vs Zipformer kroko

**Date:** 2026-05-28
**Dataset:** Dev_10k (10,521 utts, 23.6 hours of dysarthric speech)
**Scoring:** sclite + SAPC2 HF-leaderboard text normalization
  (`utils/compute_metrics.py`, `bash evaluate.sh --start_stage 2`)
**Hardware (decode only):** RunPod GPU pod, Xeon Platinum 8568Y+ (4-thread
  for Nemotron compute-RTF measurement, 16-thread for batch decode wall
  time). Zipformer kroko predictions are from an earlier run; only re-scored
  here.

## Headline

**Nemotron wins.** Lead on Dev_10k matches the Dev_streaming trend:

| Model | Encoder bundle | CER% | WER% | Empty rate |
|---|---|---|---|---|
| **nemotron_streaming / int8-static (Model class)** | **876 MB** | **21.59** | **27.46** | 13.9% |
| sherpa_zipformer / kroko | ~70 MB | 23.92 | 33.57 | 4.6% |
| Δ (Zipformer − Nemotron) | — | **+2.33** | **+6.11** | −9.3% |

Nemotron is **2.33 CER points** and **6.11 WER points** better than Zipformer
kroko on 10,521 utts. Nemotron silently drops more utterances entirely
(13.9% vs 4.6% empty), but when it does produce text the predictions are
substantially more accurate, and the net effect dominates the empty
penalty.

Sample size 10,521 → binomial CI on CER ≈ ±0.79 pp (worst case at p=0.5,
tighter at 22%). The 2.33-point gap is **~3× the noise floor** — safely
statistically significant.

## Per-etiology breakdown

(jiwer min-of-two-refs scoring per-utterance; absolute values 1–2 pp
higher than the sclite aggregate above, but the **comparison between
models is apples-to-apples** since both go through the same scorer.)

| Etiology | N | Nemotron CER% | Nemotron WER% | Nemotron empty | Zipformer CER% | Zipformer WER% | Zipformer empty | Nemotron ΔCER vs Zip |
|---|---|---|---|---|---|---|---|---|
| Parkinson's Disease | 3187 | **12.28** | 19.86 | 1.7% | 14.94 | 24.45 | 0.7% | **−2.66** |
| ALS | 2204 | **23.52** | 30.79 | 14.3% | 24.86 | 36.35 | 4.5% | −1.34 |
| Stroke | 889 | **28.33** | 38.50 | 14.6% | 32.91 | 46.50 | 7.4% | **−4.58** |
| Down Syndrome | 1711 | **37.83** | 48.56 | 12.7% | 39.75 | 55.55 | 3.2% | −1.92 |
| Cerebral Palsy | 2530 | 42.74 | 51.13 | 29.4% | **42.44** | 56.54 | 9.6% | +0.30 |
| **TOTAL (jiwer)** | 10521 | **24.39** | 32.55 | 13.9% | 26.20 | 37.98 | 4.6% | −1.81 |

**Findings**
1. Nemotron wins on **every etiology except Cerebral Palsy**, where the
   two models are essentially tied (42.74 vs 42.44 CER, within rounding /
   per-etiology noise).
2. Nemotron's largest lead is on **Stroke** (−4.58 CER) and **Parkinson's**
   (−2.66 CER) — the etiologies that are individually most intelligible.
3. **Cerebral Palsy** is the hard tier for both models (CER ~42%, WER
   ~51–57%) and the only place Zipformer is competitive. CP has 29.4%
   empty rate from Nemotron — the model bails on the hardest utterances.
4. CP + Down Syndrome together (4,241 utts, 40% of Dev_10k) carry most
   of the residual error budget. They're the right targets for any
   finetune.

## RTF measurement (Nemotron)

Batch decode (no real-time pacing), 16 threads, Xeon Platinum 8568Y+:

```
n_utt              10,521
total_compute_sec   9,894     (2.75 hours)
total_audio_sec    84,972    (23.6 hours)
aggregate_rtf       0.116
rtf_p50             0.120
rtf_p90             0.144
rtf_max             0.302
wall_sec           10,177    (2.83 hours)
```

The 16-thread RTF (0.116) is lower than the Phase 3 4-thread streaming
measurement (0.243). Linear scaling would predict 0.061 at 16 threads —
we observe 0.116, so ORT MatMul scaling is sub-linear past 4 threads,
which is expected. The eval VM (likely fewer/slower cores) will see a
proportional uplift; the Phase 5 CPU diagnostic line will tell us by
how much.

## What this changes for submission

The 123-utt Dev_streaming numbers (CER 22.31 / WER 28.08 for Nemotron;
the user's earlier 24.9 WER for Zipformer kroko on Dev_streaming was
estimated, not measured here) suggested a Nemotron lead but couldn't
distinguish it from sampling noise on 123 utts. **Dev_10k confirms the
lead is real, not a 123-utt fluke**, with high statistical confidence.

**Recommendation: proceed with Phase 5 submission** of `nemotron_streaming /
int8-static` (the kit in `track2_starting_kit/nemotron_streaming.zip`).

## Files

- `track2_starting_kit/experiments/results.jsonl` — two new entries:
  `2026-05-28-nemotron-modelclass-int8static-Dev_10k`
  `2026-05-28-zipformer-kroko-Dev_10k-rescored`
- `/workspace/phase4b/dev_10k_nemotron.csv` (pod) — predictions
- `/workspace/phase4b/dev_10k_nemotron.timing.json` (pod) — per-utt RTF
- `/workspace/phase4b/dev_10k_nemotron.etiology.json` (pod)
- `/workspace/phase4b/dev_10k_zipformer_kroko.csv` (pod, copied from
  `/workspace/SAPC-track2/track2_starting_kit/sherpa_zipformer/Dev10k.kroko.predict.csv`)
- `/workspace/phase4b/dev_10k_zipformer_kroko.etiology.json` (pod)
