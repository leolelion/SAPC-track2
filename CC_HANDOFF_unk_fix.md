# Claude Code task brief — fix the official-eval `unk` assert (SAPC2 Track 2)

You have Mac terminal + pod access (the other agent does not). Run the commands below,
report outputs back. Full analysis: `research/30_official_eval_unk_assert.md`.

## Pod
```
ssh -p 30637 -i /Users/o/.runpod/ssh/RunPod-Key-Go \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@38.80.152.249
```
Deploy submission dir: `/workspace/finetune/nemo_submission`
int8 package: `/workspace/finetune/nemo_ft/artifacts/nemotron_encoder_sos_int8_submission.zip`
Diag artifacts: `/workspace/finetune/nemo_ft/artifacts/official_eval_diag/`
Repo on pod: `/workspace/sapc-nemotron`  •  NeMo venv: `/workspace/nemoenv`

## What's wrong (already diagnosed)
`evaluate.sh` Dev-500 exits 1: "preds from sgml-ref1 and sgml-ref2 are not identical".
Control run shows **fp32 rc=1 AND int8 rc=1** — NOT int8-specific. Root cause: the model
emits a literal `unk` token in the hypothesis (e.g. utt000491 "...see new `unk` meet..."),
which aligns as a substitution vs ref1 but an insertion vs ref2, so the reconstructed hyps
diverge. Verified against the real scorer: stripping `unk` from the hyp makes pred_ref1 ==
pred_ref2, so the assert can't fire on any scorer version. This matches CLAUDE.md contract #3.

## The fix (edit submission `model.py` ONLY — never the scorer)
Strip standalone `unk` from what `accept_chunk` and `input_finished` return. Helper:
```python
import re
_UNK_RE = re.compile(r"\bunk\b")
def _strip_unk(text):
    return re.sub(r"\s+", " ", _UNK_RE.sub(" ", text)).strip()
```
Wrap both return paths so the returned string never contains a standalone `unk`.
Keep the SOS/blank init unchanged: `np.array([[1024]], dtype=np.int32)`.

## Step 1 — confirm two facts, paste outputs back
```
# (a) are ALL mismatches unk-only? want mismatches == only_unk_token_delta
grep SGML_PRED_MISMATCH \
  /workspace/finetune/nemo_ft/artifacts/official_eval_diag/int8/sgml_pred_mismatch.txt
# (b) is the pod scorer stale (predates upstream commit c48d945)?
git -C /workspace/sapc-nemotron log --oneline -1 -- utils/compute_metrics.py
# (c) dump the two output methods so the patch can be exact
sed -n '1,40p' /workspace/finetune/nemo_submission/model.py
awk '/def accept_chunk/,/def input_finished/' /workspace/finetune/nemo_submission/model.py
awk '/def input_finished/{f=1} f{print} /def input_finished/,0' /workspace/finetune/nemo_submission/model.py | head -40
```
If any mismatch is NOT unk-only (mismatches > only_unk_token_delta), STOP and report —
that's a separate non-unk bug to chase before the fix is trusted.

## Step 2 — apply the patch
Edit `/workspace/finetune/nemo_submission/model.py`: add `_strip_unk` and wrap the
`accept_chunk` / `input_finished` returns. (Paste the methods from step 1(c) to the other
agent for an exact diff if unsure.)

## Step 3 — rebuild the int8 package
Use the existing packaging script (already in repo `scripts/`):
`scripts/run_encoder_sos_int8.sh` (rebuilds + gates) or the package helper it calls.
Do not hand-roll the zip if a script exists.

## Step 4 — REAL validation gate (no proxy; house rule)
```
cd /workspace && bash diagnose_official_eval.sh   # expect fp32 rc=0 AND int8 rc=0 now
```
Accept the int8 package ONLY if: official evaluate.sh Dev-500 rc=0, paired CER/empty deltas
vs FP32 still acceptable, and the 20-row + Dev_diag gates still pass. NEVER submit to test a
hypothesis.

## Step 5 — when done
Pull final artifacts to the Mac, then STOP pod `3dwiczo41jeg1y` (house rule: always stop when done).

## Hard constraints
- Never edit scorer semantics: `local_decode.py`, `evaluate.sh`, `utils/compute_metrics.py`,
  `utils/compute_latency.py`, `steps/eval/*`. Wrap, don't modify.
- Submission must be offline (no NeMo/HF/network at submission time).
- `git add .` forbidden — stage individually. Push only to `fork`, never `origin`.
