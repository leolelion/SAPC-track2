# 30 — Official `evaluate.sh` Dev-500 assert: literal `unk` in hypotheses

> 2026-06-28. Diagnosed from the pod-side `diagnose_official_eval.sh` run
> (`official_eval_diag.log` + `official_eval_diag/{fp32,int8}/sgml_pred_mismatch.{txt,json}`).
> This note is the root-cause record and the fix spec. It changes **no** scorer semantics.

## Symptom

The organizers' `evaluate.sh` on the representative Dev-500 hyp exits 1 with:

```text
ERROR: preds from sgml-ref1 and sgml-ref2 are not identical!
len(preds_ref1)=500, len(preds_ref2)=500
```

## Key diagnostic result

`diagnose_official_eval.sh` ran the official path for **both** packages as a control:

```text
fp32 rc=1
int8 rc=1
```

**Both fail identically.** => The assert is NOT int8-specific. int8 quantization is exonerated;
this is a pre-existing hypothesis-content problem affecting the FP32 and int8 packages equally.

## Root cause — model emits a literal `unk` token

Representative offending utterance (`only_unk_delta=True`):

- utt000491, speaker `4ede54a7-...`, Parkinson's Disease.
- `official_hyp_trn`: `... getting to see new `**`unk`**` meet new people ...`
- ref1 alignment: `S,"peop","unk"` — the `unk` aligns as a **substitution** => kept in `pred_ref1`.
- ref2 alignment: `I,,"unk"` — the same `unk` aligns as an **insertion** => dropped from `pred_ref2`
  by the `process_unk` rule (`I and hyp == unk -> drop`).

So `pred_ref1` and `pred_ref2` differ by exactly that one `unk` token. This is the documented
pitfall in `CLAUDE.md` contract #3: *"Do not emit literal `unk` in hypotheses expecting it to be free."*

The literal `unk` is the model's OOV/blank-ish token detokenizing to the string `unk`. The
reference-side `unk` (from `{w:N}` / `{u:...}` uncertainty markup, see
`utils/normalizer/text_normalizer_hf.py:671-677`) is legitimate; the scorer's `process_unk`
handling is designed for *references*, not for a model that emits `unk` in its output.

## Why suppressing hyp `unk` is a provably complete fix

In `parse_sgml_csdi`, the reconstructed `pred` is simply the hypothesis tokens in alignment
order. sclite never drops a hypothesis word — C/S/I all retain the hyp token. The **only** rule
that drops a hyp token is `I and hyp == unk -> ""`. Therefore:

- If the hypothesis contains **no** literal `unk`, every hyp token is a real word retained in
  both alignments, so `pred_ref1 == pred_ref2` exactly and the assert can never fire.
- This holds regardless of the scorer/grader version Codabench runs (robust to a stale grader
  that lacks the `_is_unk_only_mismatch` tolerance added in upstream commit `c48d945`).

Bonus: a stray `unk` in the hypothesis is also a literal garbage word that scores as an
insertion error against any reference without an uncertainty marker at that position, so
suppressing it should not hurt — and likely slightly helps — CER/WER.

## The fix (submission `model.py` only — scorer untouched)

Strip standalone `unk` tokens from everything the streaming wrapper returns, in **both**
`accept_chunk` (partials) and `input_finished` (final). Apply to the deploy Nemotron
`model.py` on the pod / inside the package zip (not present in this repo).

```python
import re

_UNK_RE = re.compile(r"\bunk\b")

def _strip_unk(text: str) -> str:
    # remove standalone 'unk' word tokens, collapse the whitespace it leaves
    return re.sub(r"\s+", " ", _UNK_RE.sub(" ", text)).strip()
```

Then wrap the return values:

```python
def accept_chunk(self, buf):
    text = self._decode_partial(buf)   # existing logic
    return _strip_unk(text)

def input_finished(self):
    text = self._finalize()            # existing logic
    return _strip_unk(text)
```

Match the actual variable names in the deploy `model.py`; the only requirement is that the
*returned* string never contains a standalone `unk`. Keep the existing SOS/blank init
(`np.array([[1024]], dtype=np.int32)`) unchanged.

## Validation gate (real harness — no proxy, per house rules)

On the pod, rebuild the package with the patched `model.py`, then run the organizers' path on
Dev-500 and require a clean exit:

```bash
# decode with the patched package, then official scoring
bash diagnose_official_eval.sh            # fp32+int8 control; expect both rc=0 now
# or the targeted gate inside run_encoder_sos_int8.sh
```

Accept only if: official `evaluate.sh` Dev-500 returns `rc=0` for the int8 package, paired
CER/empty deltas vs FP32 remain acceptable, and the 20-row + Dev_diag gates still pass.
Do **not** submit to test the hypothesis.

## Open confirmation

Need the header line `SGML_PRED_MISMATCH ... mismatches=N only_unk_token_delta=M` from
`sgml_pred_mismatch.txt`. If `N == M`, suppressing `unk` clears the assert entirely. If `N > M`,
there is a residual non-`unk` divergence to chase separately before sign-off.
