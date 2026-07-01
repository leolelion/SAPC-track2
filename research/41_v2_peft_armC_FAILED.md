# 41 — v2 PEFT arm (encoder adapters): FAILED the faithful gate, WORSE than v1 (2026-06-30)

> The research/40 independent-review pivot said: drop Arm B (joint-unfreeze, adds capacity to an
> overfitting problem); the literature-correct lever is PEFT (fewer trainable params). Q chose "PEFT
> arm directly", then "faster variant". This is the result. **PEFT also FAILED** — worse than v1 on the
> faithful harness, like Arm A. Pod 3dwiczo41jeg1y stopped (EXITED) the moment Dev_diag was known.
> Artifacts: the Dev_diag bootstrap/empties JSON were read live (below); the post-stop scp missed the dir.

## Config (faster variant)
Per-layer NeMo LinearAdapter (dim 64, norm_position=pre) on all 24 ConformerLayers; **base fully frozen**
(3.195M trainable / 618M total — verified at Gate 0). Full 308k cased train, 2 epochs, lr 3e-4, AdamW
cosine (warmup 3805), **gain-only aug (-8/+8 dB), NO speed, NO severity**. CER-selection. att_context
pinned [70,1]. Adapters added per-layer because this build's ConformerEncoder lacks AdapterModuleMixin
(the layers have it). Export verified to carry adapters (96 keys loaded, 24 layers, non-zero — no-op trap
avoided at Gate 0).

## Result: WORSE than v1 (faithful Dev_diag, the decisive metric)
| set | v2 PEFT | v1 baseline | delta |
|---|---:|---:|---:|
| **Dev_diag CER** | **28.19%** | 23.58% | **+4.6 (worse)** |
| Dev_diag empties | **40/425 (9.41%)** | 7.3% | **WORSE (more empties)** |
| ALS | 39.92% (empty 25/110=22.7%) | 33.52% (20%) | worse |
| Cerebral Palsy | 28.18% | 21.55% | worse |
| Down Syndrome | 36.64% | 33.39% | worse |
| Parkinson's | 5.24% | 4.49% | ~same (mild forgetting) |
| Stroke | 23.53% | 25.09% | slightly better |
| nonempty CER | 20.73% | 17.57% | worse |

Pre-registered gate: Dev_diag CER<22% ✗, empties<5% ✗ → **STOP** (decisive; did not wait for Dev-500/PD —
stopped the pod to halt billing once the gate metric was known).

## What this tells us (the important part)
1. **PEFT did NOT beat v1 — it was the WORST of the three encoder finetunes.** Faithful Dev_diag:
   v1 (full encoder FT, no aug) **23.58%** < Arm A (enc-only + aug + severity, 50k) 27.35% < **PEFT 28.19%**.
   The reviewer's core prediction (research/40: "PEFT > full-FT for this small overfitting corpus") is
   **FALSIFIED on the faithful harness.** Fewer trainable params did not help; it hurt.
2. **Empties went UP (9.41% vs 7.3%), not down.** The research/40 thesis that empties are encoder-energy
   -sensitive and gain-aug would fix them is also **not supported** — mild gain aug at the encoder made
   empties WORSE. (Arm A's aggressive gain left them ~unchanged; mild gain increased them.)
3. **The recurring villain is the train→streaming-export gap, not capacity.** In-script NeMo transcribe
   proxy = 18.46% (looked great); faithful streaming export = 28.19% (bad). Same internal-vs-faithful
   divergence research/39 flagged for Arm A. The adapter IS applied in the ONNX (28.19% ≠ base no-op of
   ~23.58%) — it actively HURTS in the cache-aware chunked-streaming regime despite helping offline. NB:
   `attention_adapter_mixin: "No adapter compatible... Skipping adapter forward pass"` fired at export
   (expected for a Linear-not-attention adapter), but the faithful number proves the adapter changed the
   output, just for the worse. Possible mechanism: the adapter (pre-LN) interacts with the streaming cache
   normalization differently than the full-sequence training forward → offline gains don't survive streaming.
4. **Confound / caveat:** 2 epochs (vs v1's 4) and the streaming-export adapter behavior are not perfectly
   clean. But internal val was already ~flat (ep0 0.2160 → ep1 0.2139), and chasing the adapter-streaming
   interaction is more GPU on a method that must survive the export — high cost, low EV.

## Scoreboard — every Nemotron severe-tail rescue has now FAILED the faithful gate vs v1
- Cheap inference fixes (blank-penalty, gain-norm): failed (research/37).
- Arm A (encoder-only + aug + severity): failed, 27.35% (research/39).
- **PEFT adapters (this doc): failed, 28.19%, empties up.**
- v1 (full encoder FT, no aug) remains the BEST Nemotron artifact at 23.58% faithful Dev_diag.

## Recommendation → STOP the Nemotron encoder-rescue (research/40 option 3, now evidence-backed)
Three independent approaches — the original team's (Arm A/B), the reviewer's (PEFT), and cheap fixes — have
all failed to beat v1 on the faithful harness. The binding problem is NOT overfitting-fixable-by-PEFT; it is
a train→streaming-export mismatch that encoder finetuning does not improve (and adapters worsen). Banking
**A1 (Test1 23.44%)** as the Pareto entry and/or shipping **v1 int8** is the defensible call; research/14 §10's
"zipformer track higher-EV" judgment is further reinforced. Any further Nemotron GPU should target a
*different* lever (the streaming-export gap itself, or a joint/decoder-side change), not more encoder
adaptation — and only as a fresh, explicitly-approved bet. Spend so far this session: ~3h train + ~0.5h
gate/eval H200 ≈ ~$15-16.
