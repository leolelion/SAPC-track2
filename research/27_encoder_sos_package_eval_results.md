# 27 - Encoder-only SOS extracted-package gates (2026-06-27)

Ran the actual built submission zip, after extraction, through the same severe-enriched and
representative CER gates used for the deploy directory. This verifies the submission artifact
itself, not only the pod-side working directory.

## Artifact under test

- Zip: `/workspace/finetune/nemo_ft/artifacts/nemotron_encoder_sos_submission.zip`
- SHA-256: `0d20d3501fca443bd9da7ae423c2112c075517539242c87adc327a61dbf96f32`
- Size: 2.2G
- Extracted to: `/dev/shm/nemotron_encoder_sos_eval_extract`

Local pulled evidence:
- `/Users/o/Downloads/package_encoder_sos_eval.log`
- `/Users/o/Downloads/pkg_enc_sos_devdiag_cer_bootstrap.json`
- `/Users/o/Downloads/pkg_enc_sos_devrep500_cer_bootstrap.json`

Remote script:
- `scripts/run_package_encoder_sos_eval.sh`
- `scripts/autorun_package_encoder_sos_eval.sh`

Pod status after run: `EXITED` verified with `runpodctl get pod 3dwiczo41jeg1y`.

## Results

| set | n | speakers | mean CER | speaker-block 95% CI | empty | `"Wh "` |
|---|---:|---:|---:|---:|---:|---:|
| `Dev_diag.csv` severe-enriched | 425 | 103 | 24.49% | 16.06-33.16% | 30 | 0 |
| representative Dev-500 seed 23 | 500 | 119 | 10.49% | 7.92-13.55% | 6 | 0 |

These match the deploy-directory results in `research/24_encoder_sos_eval_results.md`.

## Etiology breakdown

### `Dev_diag.csv`

| etiology | n | CER |
|---|---:|---:|
| ALS | 110 | 33.95% |
| Cerebral Palsy | 152 | 22.49% |
| Down Syndrome | 67 | 33.85% |
| Parkinson's Disease | 77 | 6.20% |
| Stroke | 19 | 26.75% |

### Representative Dev-500 seed 23

| etiology | n | CER |
|---|---:|---:|
| ALS | 103 | 4.60% |
| Cerebral Palsy | 123 | 19.11% |
| Down Syndrome | 82 | 17.24% |
| Parkinson's Disease | 156 | 3.72% |
| Stroke | 36 | 11.80% |

## Interpretation

The zip is package-correct: extraction layout, root files, SOS initialization, ONNX weights, tokens,
and local-mel runtime all reproduce the validated deploy-path behavior. The old leading `"Wh "`
artifact is absent on both package gates.

The 2.2G size is plausible for this FP32 export but not ideal. The previous local Nemotron package
was about 820M on disk / 859,883,202 compressed bytes by `zipinfo`, and the handoff identifies it
as a dynamic-int8 ONNX export. Therefore the size difference is a precision/export-format issue,
not a baseline-versus-finetuned issue. Finetuning changes weights, not the model's parameter count.

## Decision

Keep this FP32 zip as the correctness fallback. The next submission-quality artifact should be an
encoder-int8 package if it preserves the representative and severe CER gates within tolerance.

## Next steps

1. Quantize the encoder to int8, keep decoder/joint FP32, rebuild the package, and rerun these two
   package-level gates.
2. Run a clean offline/no-network package gate if available.
3. Check Codabench upload constraints for the 2.2G FP32 fallback while developing the int8 successor.
4. Revisit TTFT after accuracy/package stability; p50 1.15 s is good but likely not optimal.
