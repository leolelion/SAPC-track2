# 07 — Streaming ASR techniques deep dive (literature tracker)

Deep literature survey (2026-06-21) of ASR / streaming-ASR / dysarthric-ASR techniques, mapped to **SAPC2
Track 2 constraints**: streaming · CPU real-time · ~150 MB submission(?) · latency-scored (CER×latency
Pareto, CER primary) · dysarthric. Status tags: ✅ done · ⏳ untapped/high-value · ⚠️ heavier/contingent ·
❌ ruled out by constraints.

## Current system (baseline to beat = ours)
Finetuned streaming **zipformer-M (66M)**, avg-5, greedy, **no LM**. Test1: **CER 23.44% / WER 31.51%**,
TTFT 1437 / TTLT 95 ms. #1 on leaderboard (field at 34.59%). 69 MB zip, offline-validated.

## Key precedent data points
- **SAP Challenge 2025** (Interspeech) — top WER 8.11% (offline, Whisper/Parakeet-large). Winning recipe:
  foundation model + **finetune** + **checkpoint averaging** (winner) + audio segmentation + (LoRA, GER,
  personalization). https://arxiv.org/html/2507.22047v1
- **Fine-tuning Parakeet-TDT for SAP dysarthric** — 36.3% → **23.7% WER** with finetuning.
  https://www.isca-archive.org/interspeech_2025/takahashi25_interspeech.pdf
  ⇒ Our finetuned 66M zipformer (~27% WER Test1) is in the SAME ballpark as finetuned Parakeet-TDT.
  The small-model gap to big models shrinks dramatically once both are dysarthria-adapted.

## 1. Adaptation (dominant lever for dysarthric)
- ✅ **Full finetuning** — done (36→21.6% CER dev). The #1 lever, confirmed by all SAP teams.
- ⚠️ **Speaker adaptation / personalization** (VR-SBE, f-LHUC, adapters, x-vector, meta-learning) — up to
  ~18% rel CER for dysarthric, but needs test-time speaker handling (awkward for one-shot streaming submit).
  https://arxiv.org/html/2407.06310v1 ; meta-learning https://arxiv.org/html/2509.15516v2
- ⚠️ LoRA/AdaLoRA — parameter-efficient FT (SAP teams C/D). We do full FT (fine).

## 2. Decoding + LM  ← OUR BIGGEST UNTAPPED LEVER (we are greedy, no LM)
- ⏳ **Beam search + n-gram (KenLM) shallow fusion** — classic, ~MBs, **sherpa-onnx supports natively**.
  Helps atypical speech (AM uncertain → LM disambiguates).
- ⏳ **Internal-LM subtraction: ILME / Density Ratio / HAT** — subtract the transducer's implicit LM before
  fusing external LM → **+14–15% rel over plain shallow fusion**. Lightweight at inference.
  https://ar5iv.labs.arxiv.org/html/2011.01991 ; https://ar5iv.labs.arxiv.org/html/2203.16776
- ⏳ **Two-pass neural-LM rescoring / deliberation** — rescore n-best with small neural LM → 6–19% rel WER,
  minimal added latency. https://ar5iv.labs.arxiv.org/html/2101.11577 ; RescoreBERT.

## 3. Latency (our weak axis — worst TTFT on board)
- ⏳ **FastEmit** — sequence-level emission regularization; emit tokens earlier → lower TTFT (next finetune).
  https://ar5iv.labs.arxiv.org/html/2010.11148
- ⚠️ **CUSIDE / CUSIDE-T** — simulate future context → low latency w/o accuracy loss (streaming SOTA);
  needs retrain w/ simulation module. https://arxiv.org/html/2407.10255v1
- ⚠️ **TDT (token-and-duration transducer)** — 2.8× faster decode, ~same acc (Parakeet-TDT). RTF/latency lever.

## 4. Data augmentation (we did only basic speed-perturb + SpecAugment)
- ⚠️ **Synthetic dysarthric via VC / TTS** (severity/pause/rhythm control — DARS, GAN) — BEATS speed/tempo
  perturb; 9–13%+ rel gains. Heavier (build VC/TTS pipeline). https://arxiv.org/pdf/2603.01369 ;
  https://arxiv.org/html/2505.14874v5 ; Dysarthric-SpecAugment two-stage https://www.sciencedirect.com/science/article/pii/S0010482525003051
- ⚠️ **Pseudo-labeling / self-training** (SAP Team B; dysarthric Whisper self-training
  https://arxiv.org/html/2506.22810v1) — less critical (we have 336k labeled).

## 5. "Foundation quality in a small box"
- ⚠️ **Distillation Whisper/Parakeet → small streaming student** via pseudo-labels — principled path to
  big-model quality within the size budget. https://arxiv.org/html/2409.13499v1 ;
  non-streaming→streaming encoder KD https://arxiv.org/pdf/2308.16415
- ⚠️ **Larger-but-fitting zipformer-L (~150M ≈ ~150MB)** — capacity, gated on size limit.

## ❌ Ruled out by constraints
- Big foundation models / **Nemotron 0.6B** — 821MB int8 (likely > size limit; UNVERIFIED — TEST by uploading).
- **LLM Generative Error Correction (GER)** — size + latency; AND prior team measured GEC HURTING this data.
- ROVER / multi-model ensembles — size.

## Recommended order (EV-ranked, fits budget)
1. **Decoding: beam search + KenLM shallow fusion → ILME subtraction** (cheapest, size-free, upgrades live #1).
2. **Second-pass n-best neural-LM rescoring** (stacks 6–19% rel).
3. **Next finetune: FastEmit (fix TTFT) + synthetic dysarthric augmentation.**
4. **Capacity (zipformer-L / distilled student), gated on size limit.**
5. **Nemotron-int4 / personalization / CUSIDE-T** — contingent / higher-effort.

OPEN: actual SAPC2 submission size limit unknown → empirically test by uploading the validated Nemotron zip.
See memory [[nemotron-higher-ceiling]], [[submission-offline-packaging]].
