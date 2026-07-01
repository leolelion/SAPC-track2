# 22 — Deployment results: full-unfrozen is the deploy model (2026-06-26)

**Superseded by `research/23_sos_fix_results.md`:** the later SOS-token test confirmed that the
encoder-only streaming failure was a harness bug (`_last_token=0` instead of `BLANK_ID=1024`).
With the SOS fix, encoder-only streams cleanly and is again the deploy model.

Built the [70,1] streaming submission (existing model.py + NeMo export; only CHUNK_NEW 56->16) and ran the REAL
local_decode.py. KEY: the deployment harness works, and the streaming metric FLIPS the arm choice.

## Streaming CER + latency on Dev_100 (clean subset, harness validation)
| model @ [70,1] streaming | CER | 'Wh' artifact | TTFT p50 | TTLT p50 |
|---|---|---|---|---|
| zero-shot base | 5.2% | 0 | (n/a) | |
| FT encoder-only ("transcribe winner") | 11.7% | 80/100 | -0.90s* | 0.32s |
| **FT full-unfrozen** | **1.2%** | **0/100** | 1.20s | 0.29s |
*enc-only negative TTFT = artifact of emitting premature garbage 'Wh'.

## The 'Wh' artifact — root cause
Finetuned ENCODER produces a first-frame representation the encoder-only model's FROZEN joint decodes as 'Wh'.
Full-unfrozen adapted its joint to its encoder -> decodes cleanly. (Warmup-frame-drop and skip-first-chunk both
FAILED to fix enc-only -> confirms it's the encoder rep + frozen joint, not timing.)

## DECISION: deploy FULL-UNFROZEN.
On offline transcribe, enc-only won (23.6 vs 25.2 Dev_diag). On the REAL streaming harness, full-unfrozen wins
(clean 1.2% vs broken 11.7% Dev_100). Vindicates the agent's "decide on the real harness, not transcribe" +
the two-arm run. Shipping the enc-only "winner" would have shipped a broken model.

## Latency (full-unfrozen): TTFT 1.20s / TTLT 0.29s -- both better than old Nemotron sub (2.27 / 0.29), honest.

## Remaining to a submission
1. Full-unfrozen STREAMING CER on a representative/severe set (Dev_diag-425 + full Dev) -> the real deploy CER
   (transcribe said 25.2% on hard Dev_diag; streaming likely similar since clean). vs zipformer 23.44% Test1.
2. int8 quantize encoder (keep decoder/joint fp32) + confirm CER delta.
3. Offline package (bundle ONNX + ORT wheel + local-mel model.py, no NeMo/network) -> Codabench zip.
4. Faithful Dev gate -> submit (should beat zipformer on CER AND latency).
Artifacts: pod export_unfrozen70_1/, deploy_unfrozen/ (working streaming submission dir).
