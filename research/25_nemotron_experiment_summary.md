# 25 - Nemotron experiment summary and current deployment plan (2026-06-27)

## Big picture

Nemotron is no longer just an interesting high-ceiling model. The full-data finetune and faithful
deployment tests now support it as the next submission candidate, specifically the encoder-only
finetune with a one-line SOS fix in the streaming wrapper.

The project arc was:

1. Zero-shot Nemotron showed a strong clean-speech prior but failed badly on severe dysarthria.
2. Full-data finetuning cut severe diagnostic CER from 43.4% to 23.6%.
3. A deployment artifact made encoder-only look broken under streaming.
4. The artifact was traced to wrong RNNT SOS initialization (`0` instead of blank id `1024`).
5. With SOS fixed, encoder-only is again the best deploy model.

## Results table

| stage | model/path | set | CER | notes |
|---|---|---|---:|---|
| zero-shot model-quality baseline | NeMo transcribe `[70,1]` | `Dev_diag` | 43.4% | 18% empty |
| full finetune | encoder-only, NeMo transcribe `[70,1]` | `Dev_diag` | 23.6% | best offline/model-quality arm |
| full finetune | full-unfrozen, NeMo transcribe `[70,1]` | `Dev_diag` | 25.2% | fewer empties, worse CER |
| broken deploy wrapper | encoder-only, `_last_token=0` | Dev_100 | 11.7% | `"Wh "` on 80/100 |
| SOS-fixed deploy wrapper | encoder-only, `_last_token=BLANK_ID` | Dev_100 | 1.3% | `"Wh "` on 0/100 |
| SOS-fixed deploy wrapper | full-unfrozen, `_last_token=BLANK_ID` | Dev_100 | 5.9% | clean but worse |
| SOS-fixed deploy wrapper | encoder-only | `Dev_diag` | 24.49% | speaker CI 16.06-33.16%; 30 empties |
| SOS-fixed deploy wrapper | encoder-only | representative Dev-500 | 10.49% | speaker CI 7.92-13.55%; 6 empties |
| extracted FP32 package zip | encoder-only | `Dev_diag` | 24.49% | matches deploy dir; 0 `"Wh "` |
| extracted FP32 package zip | encoder-only | representative Dev-500 | 10.49% | matches deploy dir; 0 `"Wh "` |

## Scientific interpretation

The encoder-only finetune is the best current model. It adapts acoustic representations to
dysarthric speech while preserving the pretrained decoder/joint behavior. Full-unfrozen likely
overfits or perturbs the transducer head enough to hurt generalization.

The `"Wh "` issue was not evidence that encoder-only was intrinsically incompatible with the
frozen joint. It was a wrapper bug: the predictor was primed with BPE token 0 instead of RNNT
blank/SOS. This is exactly why deployment-faithful CPU evaluation must remain the final arbiter.

The representative Dev-500 CER of 10.49% is the strongest submission-relevance signal so far and
is well below the current Zipformer hidden-Test1 reference of 23.44%. The severe diagnostic set
is still difficult, especially ALS and Down Syndrome, and should drive v2 training work.

The built FP32 submission zip is a valid correctness artifact: when extracted and evaluated, it
reproduces the deploy-dir severe and representative CER exactly. It is not yet the most polished
artifact because it is 2.2G. The old local Nemotron zip around 820M was smaller because it used a
dynamic-int8 ONNX export; the new zip is larger because it packages raw FP32 finetuned ONNX weights
with many external initializer files. Finetuning itself does not make the architecture three times
larger.

## Why not GPU batch CER + CPU latency?

GPU NeMo transcribe is useful for training and model-selection proxies, but it is not a valid final
deployment CER claim. The official accuracy pass instantiates the submitted CPU/offline `model.py`.
Accuracy bugs can live in the wrapper, setup, ONNX export, mel implementation, thread behavior, or
token initialization. The SOS bug is the proof: GPU transcribe said encoder-only was good, while
the initial CPU wrapper made it look broken.

Use GPU batch for cheap model-quality signals. Use CPU `local_decode.py` for every deploy decision
and every number trusted for submission.

## Current next tasks

1. Quantize encoder int8 only and rerun the same package-level CER gates.
2. Run an offline/no-network package gate on the extracted zip if a clean container is available.
3. Check Codabench upload practicality for the 2.2G FP32 package or submit an int8 successor instead.
4. Investigate TTFT, targeting the gap between p50 1.15 s and the expected ~0.56 s class.
