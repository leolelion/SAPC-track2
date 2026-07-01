# 23 - SOS-token fix confirms encoder-only is the deploy model (2026-06-27)

Ran the discriminating deploy-harness test from `scripts/run_sos_fix.sh` on the RunPod H200.
The test patched the streaming submission `model.py` from `_last_token=[[0]]` to
`_last_token=[[BLANK_ID]]` (`BLANK_ID=1024`), then ran the real organizers' `local_decode.py`
on both finetuned arms.

Log:
- local: `/Users/o/Downloads/sos_fix.log`
- pod: `/workspace/finetune/nemo_ft/sos_fix.log`
- pod status after run: stopped / `EXITED`

## Result

| model @ `[70,1]` streaming + SOS fix | Dev_100 mean CER | `"Wh "` prefixes | TTFT p50 | TTFT p90 | TTLT p50 | TTLT p90 |
|---|---:|---:|---:|---:|---:|---:|
| **FT encoder-only** | **1.3%** | **0/100** | **1.15 s** | **2.36 s** | 0.31 s | 0.35 s |
| FT full-unfrozen | 5.9% | 0/100 | 1.20 s | 2.41 s | **0.28 s** | **0.32 s** |

## Conclusion

The SOS-token root cause is empirically confirmed. The earlier encoder-only streaming failure was
not a model-quality failure; it was a deployment harness bug that primed the RNN-T prediction net
with BPE token 0 instead of the blank/SOS token. With the correct SOS token, the stream-start
`"Wh "` artifact vanishes (`0/100`) and encoder-only becomes the best deploy candidate again.

**Decision: ship the encoder-only finetune with the SOS fix** (`_last_token=[[BLANK_ID]]`).
This supersedes `research/22_deployment_results.md`, which chose full-unfrozen before the SOS fix
was tested.

## What this means

- Encoder-only now wins both model-quality evidence streams:
  - offline NeMo transcribe on severe-enriched `Dev_diag`: 23.6% CER vs full-unfrozen 25.2%
    (`research/20_fullrun_results.md`)
  - faithful streaming harness on clean `Dev_100`: 1.3% CER vs full-unfrozen 5.9%
- The old `11.7% BROKEN` encoder-only streaming result is invalid as a model comparison because it
  used the wrong SOS token.
- Full-unfrozen remains clean with the SOS fix but is now clearly the fallback, not the deploy pick.

## Next steps

1. Run encoder-only + SOS fix on severe/representative streaming sets:
   - `Dev_diag.csv` severe-enriched 425
   - representative Dev sample / full feasible Dev
   - speaker-block bootstrap CI
2. Patch/package the actual offline submission directory with `_last_token=[[BLANK_ID]]`.
3. Investigate TTFT: encoder-only p50 is 1.15 s, still above the published ONNX-CPU algorithmic
   delay target (~0.56 s).
4. Quantize encoder-only encoder int8, keep decoder/joint FP32, then measure dysarthric CER delta.
5. Run a clean offline packaging gate before Codabench upload.
