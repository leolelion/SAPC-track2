# 31 — unk fix + final validation gates (2026-06-28)

> This note records the unk-token fix applied to the submission model.py files,
> the re-built zips, and the status of the final validation gates launched on
> RunPod pod `3dwiczo41jeg1y`.

## Summary of what was done

### Root cause (diagnosed in research/30)

The official `evaluate.sh` Dev-500 gate failed with:
```
ERROR: preds from sgml-ref1 and sgml-ref2 are not identical!
```
Cause: the model emits a literal `unk` token in hypotheses (utt000491 and 18 others).
sclite aligns `unk` as S (substitution) against ref1 but I (insertion) against ref2 →
`process_unk` drops it from `pred_ref2` but keeps it in `pred_ref1` → assertion fires.

Confirmed: `mismatches=19 only_unk_token_delta=19` — ALL 19 mismatches are unk-only.
Confirmed: pod scorer is stale (predates c48d945 tolerance patch) → must fix in model.py.

### Fix applied

Added `_strip_unk` helper and wrapped both `accept_chunk` and `input_finished` returns in
four files:

1. `/workspace/finetune/nemo_submission/model.py` — deploy dir (patched directly)
2. `/workspace/finetune/nemo_ft/submission_encoder_sos/model.py` — FP32 package source
3. `/workspace/finetune/nemo_ft/submission_encoder_sos_int8/model.py` — int8 package source
4. Zips rebuilt from (2) and (3):
   - `nemotron_encoder_sos_submission.zip` (FP32, ~2.2G)
   - `nemotron_encoder_sos_int8_submission.zip` (int8, ~839M)

Patch inserted between Section 5 and Section 6 of model.py:
```python
import re as _re
_UNK_RE = _re.compile(r"\bunk\b")

def _strip_unk(text: str) -> str:
    return _re.sub(r"\s+", " ", _UNK_RE.sub(" ", text)).strip()
```
Both return paths wrapped: `return _strip_unk(self._last_emitted)`.

Verified in both rebuilt zips: `_strip_unk=True`, `bare_return=False`.

SOS init (`np.array([[BLANK_ID]], dtype=np.int32)`) unchanged.
CHUNK_NEW=16 unchanged.

### Why suppressing hyp unk is correct (not a scorer hack)

From research/30: in `parse_sgml_csdi`, the reconstructed `pred` is hypothesis tokens in
alignment order. sclite never drops a hypothesis word (C/S/I all retain the hyp token).
The ONLY rule that drops is `I and hyp == unk -> ""`. Therefore: if hypothesis contains
NO literal `unk`, every hyp token is retained identically in both alignments, so
`pred_ref1 == pred_ref2` exactly. The assert can never fire.

Bonus: a stray `unk` in the hypothesis scores as a garbage word → suppressing it can
only improve or maintain CER, never hurt it.

## Validation gates launched (pod-side, PID 1473)

Script: `/workspace/run_unk_fix_gates.sh`
Log: `/workspace/finetune/nemo_ft/artifacts/unk_fix_gates.log`
Started: 2026-06-28, ~09:07 UTC

### Gate 0 (complete): patch verification

Both zips confirmed: `_strip_unk` present, no bare `return self._last_emitted`.

### Gate 1 (running): official evaluate.sh fp32 + int8

Steps:
1. Delete stale `/dev/shm/fp32_devrep500.csv` and `/dev/shm/int8_devrep500.csv`
2. Delete stale extract dirs in `/dev/shm/`
3. Run `diagnose_official_eval.sh` — re-extracts from patched zips, decodes 500 utts each,
   runs official `evaluate.sh`, checks `pred_ref1 == pred_ref2` assertion

EXPECT: `fp32 rc=0 AND int8 rc=0`.

IF NOT: assertion still fires from a non-unk source → STOP, do not submit, investigate.

### Gate 2 (pending): throughput sweep

20 workers, threads=1 and threads=4, 120 rows from representative Dev-500.

EXPECT: `aggregate_rtf_wall` well below 15000 / (10521 * avg_audio_sec).
At old zero-shot int8 RTF ~0.029 → Test1 budget ~1500 wall-sec (<<15000 s limit).
Finetuned int8 might be 2-3x slower → still ~3000-4500 s, safely under budget.

### Gate 3 (pending): Dev_streaming latency

Real streaming pass (100 ms pacing), 123 utterances in Dev_streaming.csv.
Reports TTFT p50/p90 and TTLT p50/p90 for the int8 package.

CAUTION: pod has 192 vCPUs (H200). Codabench likely has ~32 vCPUs. Latency on pod is
an optimistic lower bound. Frontier membership is still expected via CER leadership
(10.6% vs A1 baseline 23.44%), not latency.

### Gate 4 (pending): artifact pull + pod stop

After all gates:
1. Pull: gate log, diagnose log+tgz, summary JSON, throughput JSONL/logs, latency JSON,
   int8 zip.
2. Stop pod `3dwiczo41jeg1y`.
3. Record final numbers in research/32.

## What is NOT redone

- CER gates (smoke, Dev-500, Dev_diag) — already completed and documented in research/28.
  The unk fix can only improve CER (fewer garbage words). No regression expected.
- Offline/no-network test — the pod has no Docker, `unshare -rn` is blocked. The ONNX
  Runtime wheel is bundled inside the zip; model.py only imports stdlib + numpy + onnxruntime.
  As per research/25 and memory `submission-offline-packaging`, this is the validated offline
  recipe. A Docker test on Mac/arm would not give representative latency.

## Decision criterion for submission

Submit the int8 zip IFF:
1. ✅ (after Gate 1) Official evaluate.sh Dev-500 `rc=0` for int8 arm.
2. ✅ (already done) Paired CER delta within tolerance: +0.12% Dev-500, +0.36% Dev_diag.
3. ✅ (after Gate 2) Test1 time budget: predicted wall time < 12000 s with 20 workers.
4. ✅ (after Gate 3) Latency recorded and documented (no pass/fail criterion, informational).

If Gate 1 fails for a non-unk reason, stop immediately — do not submit.

## Backups

- FP32 zip still exists and is also patched — use as fallback if int8 fails Gate 1.
- A1 Zipformer is the always-valid fallback public LB entry.
