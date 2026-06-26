# 20 — Full finetune results: encoder-only wins, big gain (2026-06-26)

Two-arm full finetune (all 309k utts / 683h, cased, pinned [70,1], dysarthria-SpecAug, LR 1e-4/15% warmup,
4 epochs + early-stop + top-5 ckpt avg). Held-out Dev_diag-425 (severe-enriched, Dev speakers disjoint from
Train, verified overlap=0), NeMo transcribe at [70,1]. Baseline = zero-shot via SAME path = 43.4% CER / 18% empty.

## Results
| model | Dev_diag CER | empty | internal-dev CER | ALS | CP | DS | PD | Stroke |
|---|---|---|---|---|---|---|---|---|
| zero-shot [70,1] | 43.4% | 18% | — | 61.1 | 45.8 | 49.5 | 11.2 | 31.5 |
| smoke enc-only (4k) | 30.8% | 9% | 10.4 | | | | | |
| **FULL encoder-only** | **23.6%** | **8%** | **8.01** | 32.5 | 22.4 | 34.1 | **4.8** | 20.7 |
| FULL full-unfrozen | 25.2% | 6% | 11.06 | 32.9 | 24.6 | 35.1 | 7.4 | 23.2 |

## Findings
- **Finetuning gain (apples-to-apples, unseen speakers): 43.4 → 23.6% CER (−19.8), empties 18 → 8%.** Full data
  bought another −7 over the smoke (30.8→23.6).
- **ENCODER-ONLY is the winner** (23.6 < 25.2 overall; 8.01 < 11.06 internal-dev). The smoke-scale advantage
  HELD at 309k utts — contra the prediction it might reverse. Full-unfrozen has marginally fewer empties (6 vs 8%)
  but worse CER and worse internal-dev (overfit signal). **Pick encoder-only.**
- **No catastrophic forgetting** — Parkinson's (mild) 11.2 → 4.8%. Empties on severe etiologies slashed
  (ALS 61→32%, CP 46→22%, Stroke 31→21%).

## CAVEATS (do not over-read)
1. **This is NeMo transcribe() CER, NOT deployment streaming-CPU CER** (confound C still open). The real Test1
   number needs ONNX → local_decode.py faithful harness. 23.6% is the model-quality signal, not the submit number.
2. **Dev_diag is severe-ENRICHED** (deliberately the 7 worst speakers + duration bins). Zero-shot was 43.4% here
   vs ~20-25% on representative Dev. So full enc-only 23.6% on the hard set projects to **~mid-teens CER on
   representative Dev/Test1** — which would beat the zipformer's 23.44% Test1. PROJECTION, not promise.
3. **Cost/time:** the run took ~13h wall-clock (dataloader-bound at ~53% GPU + heavy full-unfrozen arm), not the
   4-8h estimated → ~$57. For future runs: set `pretokenize=False` / more dataloader workers; encoder-only alone
   is ~half the time (skip full-unfrozen now that it lost).

## NEXT — the remaining critical path (deployment)
The model is good; shipping is the open work. Build/validate the **ONNX cache-aware streaming `model.py`**
(adapt our existing Nemotron submission model.py to NeMo's exported encoder-model.onnx + decoder_joint-model.onnx
I/O) → run the FULL encoder-only checkpoint through `local_decode.py` on Dev → the TRUE streaming CPU CER +
latency (TTFT/TTLT) → int8 quantize + offline package → faithful Dev gate → submit. Validate the harness on the
zero-shot export FIRST. Artifacts: pod `/workspace/finetune/nemo_ft/full_enc/ft_smoke_encoder_only.nemo` (the
winner) + `full_unfrozen/` + `export_enc/` (earlier ONNX export).
