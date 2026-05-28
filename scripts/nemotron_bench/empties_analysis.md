# Empty-prediction triage — Nemotron Dev_10k

**Date:** 2026-05-28
**Source data:** `dev10k_data/dev_10k_nemotron.csv` (Phase 4b run via the Phase 3
Model class wrapping the danielbodart int8-static ONNX, [70,6]). 1463 / 10521
utts (13.91%) produced empty output — matches the ledger figure exactly, so
the extraction is reliable.

## TL;DR

**H1 (short-utt starvation) is REJECTED. H2 (RNN-T blank collapse on hard
acoustics) is CONFIRMED.** No structural fix to ship. Recommendation:
**ship the existing 21.59 CER submission as-is**; empties are the model
correctly declining to guess on a known-hard subset (CP-heavy, short
references, high-severity speakers). Forcing output would produce garbage
and hurt CER.

Document the breakdown as a finetune-planning input. The CP / high-severity
empty cluster is the right target for SAP-specific finetuning.

## Step 3 — `model.py` structural inspection (code reading)

### H4 — VAD / energy gating: ABSENT ✓

- `accept_chunk()` (model.py:202–212) unconditionally appends every chunk
  to `self._raw_chunks`. No silence check, no energy gate, no VAD.
- `_ensure_features()` (model.py:316–327) runs the NeMo preprocessor with
  `dither=0.0, pad_to=0` over all accumulated audio. No trimming.
- No call to `webrtcvad`, `torchaudio.functional.vad`, or anything similar.

→ Not a contributor.

### H3 — `input_finished()` tail-flush: implemented correctly ✓

`input_finished()` (model.py:214–221) calls `_run_steps(is_final=True)` then
`_run_drain()`.

`_run_drain()` (model.py:379–393) builds `[last 9 real mel | 56 zero mel]`,
`length = 65`, runs one extra encoder step with the rolled-forward caches.
The 56 trailing-zero mel frames give the encoder the [70,6] right-context
lookahead it needs to commit the final real frames.

**Gate:** drain returns early if `step_num == 0`. This subset is fully
contained in H1a below (T == 0 utts).

### H1 — short-utterance starvation: small corner-case, but Step 2 data rules it out

- **H1a** (T == 0, i.e. `<25 ms` of audio): `_run_steps` returns at line 339,
  drain is gated off at line 381, output stays `""`. **Step 2 data shows 0
  utts in this bucket.** Not happening.
- **H1b** (T < 56, ~25–560 ms): partial-chunk path runs one truncated
  encoder step with `chunk_length = 9 + T`. Effective real input is
  `7 + T` frames; encoded output `≈ ceil((7+T)/8)` frames. Decoder has
  little evidence, *might* emit only blanks. **Step 2 data shows 0 utts
  in this bucket.** Not happening either.
- **H1c** (drain skipped on `step_num == 0`): subsumed by H1a; same null
  set.

→ H1 is not the explanation for any of the 1463 empties. The code is
robust to short utts here — they just don't exist in Dev_10k.

## Step 1 — Empty-set extraction

```
total predictions: 10521
manifest utts:     10521
empties:            1463 (13.91%)   ← matches ledger
```

Per-utt projection saved to `scripts/nemotron_bench/empties_dev10k.csv`
(manifest fields + reference_word_count + timing data). Used by Step 2.

Note: Dev_10k manifest doesn't have `mfa_speech_start` / `mfa_speech_end`
columns — those exist only on Dev_streaming. So Step 2 distribution #2
(speech-active duration) is N/A for Dev_10k. The five other distributions
are sufficient to discriminate H1 vs H2.

## Step 2 — Six characterization distributions

### 2.1 Audio-duration buckets, empty rate per bucket

| Bucket | N total | Empty | Empty rate |
|---|---:|---:|---:|
| <0.5s | 0 | 0 | — |
| 0.5–1s | 0 | 0 | — |
| 1–1.5s | 1 | 0 | 0.00% |
| 1.5–2s | 147 | 48 | **32.65%** |
| 2–3s | 923 | 165 | 17.88% |
| 3–5s | 3221 | 463 | 14.37% |
| >5s | 6229 | 787 | 12.63% |

**Read:** Empty rate is *higher* on shorter utterances (32.65% at 1.5–2 s
vs 12.63% at >5 s) but **no utterances are short enough to starve the
encoder**. The shortest Dev_10k utt is 1.05 s = ~1.9× one full chunk. The
duration→empty gradient is consistent with "shorter utts give the model
less evidence to compose a guess, so it bails more often" — H2 behavior,
not H1.

### 2.2 Speech-active duration

N/A — Dev_10k manifest doesn't carry MFA columns. Not blocking; #6 already
proves audio duration is well above the starvation threshold.

### 2.3 Reference word count

| Set | n | min | p25 | p50 | mean | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Empty | 1463 | 1 | 3 | 4 | **4.44** | 6 | 8 | 20 |
| Non-empty | 9058 | 1 | 4 | 6 | **9.91** | 9 | 18 | 228 |

71.5% of empty references have ≤6 words. The empty set is biased toward
shorter references (typically short commands / single-line answers in
SAPC2). Hard-utt × short-utt is exactly where dysarthria worst-case lives.

### 2.4 Per-etiology empty rate

| Etiology | N | Empty | Empty rate |
|---|---:|---:|---:|
| Parkinson's Disease | 3187 | 55 | **1.73%** |
| Down Syndrome | 1711 | 218 | 12.74% |
| ALS | 2204 | 316 | 14.34% |
| Stroke | 889 | 130 | 14.62% |
| Cerebral Palsy | 2530 | 744 | **29.41%** |

**17× ratio** between Parkinson's (cleanest tier) and Cerebral Palsy
(hardest tier). Empty rate tracks etiology severity precisely. This is a
direct signal of H2: encoder representations for high-severity dysarthria
are far enough OOD that the RNN-T decoder collapses to blank.

### 2.5 Top-10 speakers by empty count

| Speaker (prefix) | Etiology | N utts | Empty | Rate |
|---|---|---:|---:|---:|
| `55c1784a-ece…` | ALS | 104 | 90 | **86.54%** |
| `6b942f5f-0f1…` | Stroke | 95 | 80 | 84.21% |
| `24e7fbed-4e8…` | Cerebral Palsy | 109 | 90 | 82.57% |
| `d41daa38-6d4…` | Cerebral Palsy | 95 | 78 | 82.11% |
| `54618732-a2c…` | Cerebral Palsy | 96 | 75 | 78.12% |
| `5801631a-2f0…` | Cerebral Palsy | 92 | 70 | 76.09% |
| `fb9c683c-41a…` | Cerebral Palsy | 89 | 66 | 74.16% |
| `b2f16a07-844…` | Cerebral Palsy | 99 | 67 | 67.68% |
| `031e84ad-54f…` | ALS | 99 | 63 | 63.64% |
| `4a9f71ab-f3a…` | Down Syndrome | 88 | 51 | 57.95% |

Top 10 speakers account for **730 / 1463 = 50%** of all empties despite
holding ~9% of the manifest. Six of the ten are CP. These are the
highest-severity speakers and the model produces empty on 58–87% of their
utterances. The empty problem is concentrated, not spread.

### 2.6 Encoder-step count per empty utt — the decisive distribution

`n_steps ≈ floor(audio_duration / 0.56)` at [70,6].

| Steps available | Empty count | Empty share |
|---|---:|---:|
| `T==0` (<25 ms) | **0** | 0.00% |
| `<1` full chunk (25–560 ms) | **0** | 0.00% |
| 2 chunk(s) | 22 | 1.50% |
| 3 chunk(s) | 55 | 3.76% |
| 4 chunk(s) | 91 | 6.22% |
| 5–9 chunks | 620 | 42.38% |
| 10+ chunks | 675 | 46.14% |

**Zero empties were duration-starved.** Every empty utt had at least 2 full
encoder chunks (1.12+ s of audio). 88.5% had 5+ chunks. The model had
plenty of evidence and chose to emit blank on every encoder frame anyway.

**This conclusively rejects H1.** The structural empty path in model.py
exists (H1a) but is hit by zero utts in Dev_10k. Padding `input_finished()`
or guaranteeing a minimum chunk count would change nothing — there's no
short-utt subset to rescue.

## Hypothesis verdict

| H | Verdict | Evidence |
|---|---|---|
| H1a (T==0 bypass) | **REJECTED** | 0 utts in Dev_10k have <25 ms audio |
| H1b (T<56 truncated step) | **REJECTED** | 0 utts have <560 ms audio |
| H1c (drain skipped on step_num==0) | **REJECTED** | subsumes H1a; same null set |
| H3 (tail-flush bug) | **NOT PRESENT** | drain correctly appends 56 zero mel frames as right-context |
| H4 (VAD over-trims) | **NOT PRESENT** | no VAD anywhere in pipeline |
| H2 (RNN-T blank collapse on OOD acoustics) | **CONFIRMED** | empty rate tracks etiology severity 17×; top-10 high-severity speakers carry 50% of empties; shorter refs (less context for composition) bias toward empty |

## Step 4 decision — DO NOT SHIP A FIX

The plan's hard rule applies: "if the fix turns empties into garbage, do
NOT ship it. A correct empty often scores better than a confident wrong
guess under CER. We'd rather submit 21.59 honestly than 22.5 from
forced-wrong-output."

There is **no structural fix to test** — H1 is rejected and the code is
already correct. Forcing output on H2 cases would require either:
- Reducing the blank probability threshold (push the model toward
  emitting non-blank) — high risk of garbage on CP.
- Beam search with length penalty — possible but adds compute and is a
  research detour vs. the planned finetune.
- Quantization-aware finetune of the encoder on SAP — the proper fix and
  the next planned phase anyway.

None of these belong in the first submission. The 21.59 CER number
represents Nemotron's honest performance with the model declining to
guess where it doesn't have a confident path. Ship as-is.

## What this changes for the finetune phase

The empty-set characterization is the highest-EV input for finetune data
mixing:

- **Oversample CP and the top-10 high-empty speakers.** Their utterances
  carry the bulk of the residual error budget and the model is currently
  silent on them.
- **Bias toward shorter references** (≤6 words) in the augmentation set —
  the model's empty bias correlates with short references, suggesting
  the encoder doesn't have time to settle into a confident representation
  on short clips of hard speech.
- Track per-speaker empty rate as a metric. The 10 speakers above are the
  scoreboard for any future finetune.

## Files

- `scripts/nemotron_bench/empties_dev10k.csv` — 1463 rows, per-utt manifest fields
- `scripts/nemotron_bench/empties_extract.py` — Step 1 extraction
- `scripts/nemotron_bench/empties_characterize.py` — Step 2 distributions
- `scripts/nemotron_bench/empties_step2.md` — raw Step 2 markdown
  (subset of this file)
- `scripts/nemotron_bench/dev10k_data/` — local copy of Dev_10k.csv,
  `dev_10k_nemotron.csv`, `dev_10k_nemotron.timing.json` (gitignore-able
  if size matters; ~6 MB combined)
