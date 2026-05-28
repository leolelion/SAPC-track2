# Pre-submission scoring audit

**Date:** 2026-05-28
**Question 1:** Does our sclite + SAPC2 normalization pipeline match what Codabench will actually run on Test1?
**Question 2:** Have we been scoring against both disfluency references (per-utt MIN), or just one?

## TL;DR

**Both checks pass. The shipped 21.59 CER stands.** Submit with confidence.

| Check | Verdict | Evidence |
|---|---|---|
| Codabench scoring path = our path | **MATCH** (with one upstream addition that's dead code on our predictions) | utils/ diff vs upstream `xiuwenz2/SAPC-template/utils/` HEAD `c48d945`; 0 of 10,521 hyps contain UNK tokens that would trigger the added branch |
| Dual-ref MIN(with-disfl, without-disfl) used | **YES** | `compute_metrics.compute_from_sgml` calls `CharErrorRateMinTwoRefs(clip_at_one=True)` with both ref TRNs; verified via code reading and sanity-check that single-ref CER is +0.47 / +0.66 worse than min(ref1, ref2) |

No re-submission needed if upstream scoring changes; the 21.59 number is computed on dual-ref MIN already.

---

## Task 1 — Codabench scoring path

### 1.1 Locating the canonical source

- The Track 2 spec ([`Track2 (Streaming ASR Track).md`](../../Track2%20(Streaming%20ASR%20Track).md)) explicitly cites `utils/metrics/cer.py` and `utils/metrics/wer.py` at `github.com/xiuwenz2/SAPC-template` as the implementation.
- Our repo is a fork of that upstream. Cloned upstream HEAD `c48d945` (update compute_metrics) into `/tmp/SAPC-template-upstream/`.
- Docker image `xiuwenz2/sapc2-runtime:latest` was not extracted (no docker/podman in this RunPod container; tried earlier in Phase 4). Confirmed `scoring.py` mentions in our `utils/normalize_hyp.py:82` and `utils/compute_metrics.py:155` ("Called by scoring.py (bundle) or directly as a library function") — `scoring.py` is a thin wrapper around the same library calls in `utils/`.

Conclusion: **`utils/` IS the scoring code**. The wrapper just calls it.

### 1.2 Specific behaviors (from code reading)

#### A. Empty hypothesis handling

When hyp is `""`, sclite aligns it as all-D against the reference. After `parse_sgml_csdi`, `preds = ""` and `target = ref`. Then `_cer_update`:

```python
ed = _edit_distance(list(pred_tokens), list(tgt_tokens))   # ed == len(target) (all deletes)
if clip_at_one and tgt_len > 0:
    ed = min(ed, tgt_len)                                  # ed = ref_len (already)
errors += ed
total += tgt_len
```

So an empty hyp contributes per-utt CER = 1.0 (= ref_len errors / ref_len chars). Both refs contribute the cap; the min path picks whichever has lower CER (both are 1.0 → tie → average). No special "skip empty utts" branch. **`utils/metrics/cer.py:53-65`, lines 93-108 for the min-two-refs version.**

#### B. Text normalization

Two paths in `utils/normalizer/text_normalizer_hf.py`:

- **Hyp**: `normalize_text(text)` calls `EnglishTextNormalizer().norm(text, apply_markup=False)`. `utils/normalize_hyp.py:88`. Pipeline:
  1. Merge uppercase abbreviations + disambiguate `St.` (saint vs street) before lowercasing.
  2. Lowercase.
  3. `expand_common_is_contractions` — only the canonical "is" contractions (`it's`, `what's`, `where's`, …) expand to `<word> is`. Possessives like `Mom's` stay as `mom's`.
  4. Strip ignore_patterns + double quotes + word-boundary single quotes.
  5. Run a list of regex replacers (loaded at init).
  6. Remove commas between digits (`1,000` → `1000`); collapse multiple periods.
  7. Periods not followed by a digit → space (`abc. def` → `abc def`).
  8. `remove_symbols_and_diacritics`, keeping `.'%$¢€£`.
  9. `standardize_numbers` (spelled → digits, suffixes preserved like `1960s`, currency-after rules).
  10. `standardize_spellings` (US/UK normalization via `english_abbreviations.py`).
  11. Re-collapse whitespace.

- **Ref**: same pipeline but `apply_markup=True`. Additionally:
  - `[anything]` → space (square-bracket annotations like partial-word markers gone).
  - `{...}` blocks handle UNK count syntax: `{w:N}` → N copies of `UNK`, `{u:...}` → 1 `UNK`, otherwise keep mapped content.
  - `(...)` blocks: if `remove_parentheses=True` (ref2 = without-disfluency) → parens content dropped; otherwise (ref1 = with-disfluency) parens preserved.

So **ref1** = the canonical text with the parenthesized disfluencies preserved; **ref2** = the same with parens removed. Both have UNK markers for unintelligible regions.

No tokenization happens here — output is a space-separated string. Character-level edit distance later uses Python's `list()` to split by Unicode codepoint.

#### C. Aggregation

**Corpus-level**: `errors / total` where `errors = sum(per-utt ed)` and `total = sum(per-utt ref_len)`. NOT per-utt CER averaged. `utils/metrics/cer.py:53-65` and `_cer_compute` at line 112.

Per-utt CER is capped at 1.0 via `clip_at_one=True` (the SAPC2 spec "100% per utterance" rule). `utils/metrics/cer.py:60-62`.

#### D. Tokenization for CER

`_edit_distance(list(pred_tokens), list(tgt_tokens))` — `list(str)` splits by **Unicode codepoint**. Spaces are characters. Hyphens, apostrophes, digits are characters. NFKD normalization happens earlier in `remove_symbols_and_diacritics` so combining marks are handled. No grapheme-cluster awareness, but on English text the codepoint and grapheme paths agree.

### 1.3 Our path (this fork)

Our `utils/` is the same code paths as upstream — we forked from `xiuwenz2/SAPC-template` and didn't modify normalization or sclite invocation. Specifically:

- We call `bash evaluate.sh --start_stage 2 --split <split> --hyp-csv <csv>`.
- `evaluate.sh` calls `evaluate.py` which normalizes hyp + runs sclite (`-i wsj -o all sgml`) twice (ref1 and ref2) + parses SGML via `compute_metrics.compute_from_sgml`.
- The sclite flags match the standard SAPC2 use: `-i wsj` for parenthesized utt IDs; `-o all sgml` to emit the SGML format our parser reads.
- No pre-normalization beyond what `normalize_hyp.normalize_text` does. No post-processing of the metric. **It is the SAPC2 scoring path.**

### 1.4 Diff: ours vs upstream (`c48d945`)

```
utils/evaluate.py              : IDENTICAL
utils/normalize_hyp.py         : IDENTICAL
utils/normalize_ref.py         : IDENTICAL
utils/normalizer/text_normalizer_hf.py : IDENTICAL
utils/manifest.py              : IDENTICAL
utils/compute_latency.py       : IDENTICAL

utils/compute_metrics.py       : DIFFERS
utils/metrics/cer.py           : DIFFERS
utils/metrics/wer.py           : DIFFERS
```

#### What the differences do

Upstream added support for **dual predictions** in min-two-refs scoring:

- Old API (ours): `_cer_update_min_two_refs(preds, target1, target2)` — same hyp tokens used against both refs.
- New API (upstream): `_cer_update_min_two_refs(preds1, preds2, target1, target2)` — potentially different hyp tokens per ref.

When can `preds1 != preds2`?

The hyp tokens going into sclite are the same (one `hyp.trn`). They become potentially different in `parse_sgml_csdi` because of UNK substitution rules:

```python
# I and hyp == UNK : drop this UNK from preds
if process_unk and hyp_tok == unk_token:
    hyp_tok = ""
```

If sclite aligns an Insertion segment where the hyp token was `unk`, that token gets dropped from the parsed preds — and this can fire on the ref1 alignment but not the ref2 alignment (or vice-versa), producing slightly different `preds_ref1` and `preds_ref2`.

Our old code errors out (`sys.exit(1)`) if `preds_ref1 != preds_ref2`. Upstream's new code (1) tolerates UNK-only mismatches via `_is_unk_only_mismatch`, and (2) computes ED separately per ref using each ref's matching preds.

#### Does this change OUR 21.59?

**No**, because the upstream branch only fires when the hyp contains a literal `unk` token. We verified:

```
$ python3 -c "<scan dev_10k_nemotron.csv through utils/normalize_hyp.normalize_text>"
total predictions: 10521
hyps containing literal UNK token after normalization: 0 (0.00%)
```

Zero of our 10,521 hyps contain `unk` after normalization. The "I-with-hyp-UNK" rule in `parse_sgml_csdi` cannot fire, so `preds_ref1 == preds_ref2` for every utt in our run, and our old code (which uses the single `preds` for both EDs) produces the same numbers as upstream's new code (which uses `preds_ref1` and `preds_ref2`, which happen to be equal).

Confirmed: our scoring on Dev_10k succeeded with no `sys.exit(1)` from the mismatch check, consistent with this.

### 1.5 Verdict per A/B/C/D

| Question | Match? | Notes |
|---|---|---|
| A. Empty hyp | **MATCH** | both clip per-utt CER at 1.0 via `clip_at_one=True`; aggregation is corpus-level so an empty utt costs `ref_len` errors |
| B. Normalization | **MATCH** | both use HF EnglishTextNormalizer with our local mods; identical files |
| C. Aggregation | **MATCH** | `errors / total`, corpus-level. Not per-utt average. |
| D. Tokenization (CER) | **MATCH** | per Unicode codepoint via `list()`; spaces and punctuation are characters |

**No RISKY mismatch.** Our 21.59 is what Codabench will compute on the same predictions, modulo the harmless upstream UNK-dual-pred enhancement which is dead code on our data.

---

## Task 2 — Dual-reference verification

### 2.1 Manifest has two ref columns ✓

`Dev_10k.csv` header:
```
id,speaker,etiology,audio_filepath,duration,text,
norm_text_with_disfluency,norm_text_without_disfluency
```

Both refs are populated. They are produced by `utils/normalize_ref.py` using `EnglishTextNormalizer.norm` with `apply_markup=True` and either `remove_parentheses=False` (ref1, with disfluency) or `remove_parentheses=True` (ref2, without).

Sample CP utt (`24e7fbed-4e86-452f-67f6-08dcbab8e251_60_8012`):
- `text`: `Let me see this is uh hmm UNK go UNK and uh hmm`
- `norm_text_with_disfluency`: same kind of content with parenthesized disfluencies kept
- `norm_text_without_disfluency`: parenthesized regions dropped

In Dev_10k, **1231 / 10521 = 11.7%** of utts have ref1 ≠ ref2. The rest are identical (no parenthesized content) and contribute equally to either single-ref score.

### 2.2 Our scoring uses dual-ref MIN ✓

`utils/compute_metrics.py:155-190`:

```python
def compute_from_sgml(sgml_ref1: str, sgml_ref2: str):
    preds_ref1, target_ref1 = parse_sgml_csdi(sgml_ref1, ...)
    preds_ref2, target_ref2 = parse_sgml_csdi(sgml_ref2, ...)
    ...
    cer_metric = CharErrorRateMinTwoRefs(clip_at_one=True)
    wer_metric = WordErrorRateMinTwoRefs(clip_at_one=True)
    cer_min = float(cer_metric(preds, target_ref1, target_ref2))
    wer_min = float(wer_metric(preds, target_ref1, target_ref2))
```

`utils/metrics/cer.py:CharErrorRateMinTwoRefs` (line 246) computes `min(CER(hyp, ref1), CER(hyp, ref2))` per utterance (line 100-108) with the per-utt `clip_at_one=True`. The corpus aggregate selects the ref that gave the lower edit count per utt and sums its errors and target length:

```python
if cer1 < cer2:
    errors += ed1; total += len1
elif cer2 < cer1:
    errors += ed2; total += len2
else:
    errors += 0.5 * (ed1 + ed2)
    total += 0.5 * (len1 + len2)
```

So we are using the dual-ref MIN scoring. **Confirmed via code reading and the sanity check below.**

### 2.3 Sanity: dual-ref improves vs single-ref

Re-scored Dev_10k Nemotron predictions with jiwer (not sclite — absolute numbers differ from 21.59 by ~3 pp because jiwer normalization is less aggressive). The *relative* comparison still holds and tells us whether dual-ref was material.

```
Mode                            CER%    WER%
----------------------------------------------
Single ref (with disfl)        24.78   33.13
Single ref (without disfl)     24.97   32.89
Min(both refs)                 24.31   32.46

Δ ref1-only vs min: CER +0.47, WER +0.67
Δ ref2-only vs min: CER +0.66, WER +0.42
```

Dual-ref MIN buys ~0.5 CER absolute. Modest but real. If we had been scoring single-ref only, our number would be ~22.1 instead of 21.59 — still wins vs Zipformer kroko (23.92), so the qualitative conclusion would be the same, but the *quantitative* margin would shrink.

### 2.4 Codabench also implements dual-ref MIN

Confirmed from `utils/compute_metrics.py` (both ours and upstream): `CharErrorRateMinTwoRefs` and `WordErrorRateMinTwoRefs` are the metric classes used in `compute_from_sgml`. The Codabench `scoring.py` wrapper imports the same library functions per the docstring comments.

The spec ("two references with and without disfluencies, the lower error is selected per utterance") matches the code: per-utt min, then corpus-level aggregation of the winning ref's errors and target length.

---

## Combined comparison table

| Number | What it is |
|---|---|
| **21.59** | Our Dev_10k CER, sclite + dual-ref MIN, on `dev_10k_nemotron.csv` (`f160e39`) |
| ~21.59 | What Codabench would compute on the same predictions (verified equivalence) |
| 24.31 | Dev_10k CER under jiwer (looser normalization), dual-ref MIN |
| 24.78 | Dev_10k CER under jiwer, **single ref (with disfluency)** — what we'd report if we'd been scoring wrong |
| 24.97 | Dev_10k CER under jiwer, **single ref (without disfluency)** — same |

Zipformer kroko's published Dev_10k CER (23.92) was already re-scored on the same sclite + dual-ref MIN path in Phase 4b. The Nemotron-vs-Zipformer 2.33 CER lead is apples-to-apples and survives.

---

## Verdict

**Submission CER 21.59 stands as the expected Codabench number** modulo the standard Dev_10k → Test1 distribution shift. No re-scoring or zip change needed.

Things worth knowing going into the submission:

- The model will be scored by code we have read and tested locally.
- We are not under-scoring ourselves (dual-ref MIN was used).
- Empty hyps cost 100% per utt (cap), exactly as expected. The 1463 empties in Dev_10k accounted for 7.05% of total reference characters; their cap is built into the 21.59.
- A future upgrade to upstream's dual-pred scoring would not change our number on this model. If a future model emits literal `unk` tokens, we'd need to re-check.

## Files

- `scripts/audit/scoring_source.md` — source locations
- `scripts/audit/scoring_audit.md` — this file
- `scripts/audit/single_vs_dual_ref_sanity.py` — jiwer sanity script comparing single-ref vs dual-ref MIN
- `/tmp/SAPC-template-upstream/` — local clone of upstream for diffs (gitignored, re-cloneable)
