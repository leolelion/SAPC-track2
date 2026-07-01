# 26 - Encoder-only SOS offline package build and smoke (2026-06-27)

Built the actual encoder-only + SOS-fixed offline submission zip on the RunPod volume and smoke-tested
the extracted package through the real `local_decode.py` path.

## Artifact

- Zip: `/workspace/finetune/nemo_ft/artifacts/nemotron_encoder_sos_submission.zip`
- SHA-256: `0d20d3501fca443bd9da7ae423c2112c075517539242c87adc327a61dbf96f32`
- Size: 2.2G
- Zip structure: 310 entries, 300 weight files, no nested root, required root files present.
- Local pulled checksum: `/Users/o/Downloads/nemotron_encoder_sos_submission.zip.sha256`
- Local log: `/Users/o/Downloads/package_encoder_sos.log`
- Local smoke JSON: `/Users/o/Downloads/pkg_smoke_cer.json`

## Build recipe

Script: `scripts/run_package_encoder_sos.sh`

It builds from:
- base offline submission dir: `/workspace/finetune/nemo_submission`
- encoder-only ONNX export: `/workspace/finetune/nemo_ft/export_full70_1`

Applied deploy changes:
- rename exported ONNX roots to `weights/encoder_model.onnx` and `weights/decoder_model.onnx`
- keep the NeMo-exported external initializer files in `weights/`
- set `CHUNK_NEW=16`
- set `_last_token=[[BLANK_ID]]`

The image lacks `zip` and `unzip`, so the script uses Python `zipfile` for both packaging and extraction.

## Extracted package smoke

Smoke settings:
- extracted zip to `/dev/shm/nemotron_encoder_sos_extract`
- ran `setup.sh`
- ran `local_decode.py` on a 20-row Dev smoke manifest
- used one-row streaming smoke with `--streaming-interval 0`

Result:

| set | CER | empty | `"Wh "` |
|---|---:|---:|---:|
| Dev smoke 20 | 0.99% | 0 | 0 |

The package loads the ONNX files from the extracted zip tree and decodes successfully.

## Package-level gates

After this smoke, the extracted zip was also run through the severe-enriched and representative
package gates. It reproduced the deploy-dir CER numbers exactly:

| set | n | CER | empty | `"Wh "` |
|---|---:|---:|---:|---:|
| `Dev_diag.csv` severe-enriched | 425 | 24.49% | 30 | 0 |
| representative Dev-500 seed 23 | 500 | 10.49% | 6 | 0 |

See `research/27_encoder_sos_package_eval_results.md`.

## Caveats

- The zip is large (2.2G) because this package uses the raw FP32 finetuned ONNX export with many
  external initializer files. The earlier local Nemotron zip was 820M because it used the dynamic-int8
  ONNX export, not because it was an unfinetuned baseline.
- Size was not a correctness blocker, but upload practicality should be checked before Codabench.
- The smoke did not enforce a network-disabled container, though `setup.sh` uses the offline bundled
  wheel path in the existing submission recipe.

## Next steps

1. Quantize encoder int8 before final submission if the CER delta is acceptable.
2. Run a stricter offline gate in a clean/no-network container if available.
3. Prepare Codabench upload of this FP32 fallback or its int8 successor.
