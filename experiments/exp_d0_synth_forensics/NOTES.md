# exp_d0_synth_forensics — D0 run, session 2026-07-28 (overnight, pod 38.80.152.148:30836)

Registry row: `experiments/PLANNED.md` D0. Runbook: `investigations/d0_runbook.md`.

## What actually happened vs the runbook

The runbook assumed synthetic transcripts live under the corpus roots. **They do not.**
`--mode discover` found 5 "candidate transcript files" per corpus, but inspection showed they are
**tar-packaging manifests** (`tar_name`, `file_count`, `tar_sha256`, `s3_tar_key`) — no `text` field.
A repo-wide search for a `slot`→text table hit only NeMo library files. Rule 0 applied: no guessing.

Recovered what was recoverable, reported the rest as N/A:

| corpus | transcripts | how |
|---|---|---|
| **kNN-VC** | **RECOVERED** (144,291 / 203,427 = 70.9%) | filenames are `{SAP_id}__v-{voice}.wav`; kNN-VC is voice conversion of REAL SAP audio, so the source utterance's transcript applies. Joined to `Train.csv` on id (with a `_16kHz` suffix variant). Script: `scripts/build_knnvc_transcripts.py`. |
| **F5-TTS** | **NOT AVAILABLE on this pod** | filenames are `{severity}__slot{NNNNNN}__llm.wav`; the LLM-generated text and its slot table are not on disk. Passed an explicit empty file so the tar manifests could not masquerade as transcripts. F5's text half of G-COVER and G-PROV = **N/A, not guessed.** |

## UNPLANNED FINDING (binding on D2–D4): both synthetic corpora contain Dev-derived material

`scripts/d0_leak_check.py` (written this session), full walk, not sampled:

| corpus | Train-provenance | **Dev-provenance** | unknown |
|---|---|---|---|
| kNN-VC (by source utterance id) | 144,291 wavs / 307 spk | **18,797 wavs / 41 spk (9.2%)** | 40,339 / 72 spk |
| F5-TTS (by speaker-UUID bucket) | 104,113 wavs / 727 buckets | **11,931 wavs / 88 buckets** | 23,373 / 193 buckets |

SAP `Train`/`Dev` speakers are disjoint (overlap = 0), so provenance is decidable per file.

**Consequence.** Training on either corpus unfiltered puts Dev-derived audio (kNN-VC: Dev audio +
Dev transcripts) into training, which would contaminate the Dev gate — our **only** ship authority
(`validate-against-real-harness`). Any D2/D3 run **must** filter to Train-provenance only. This is
cheap (id / speaker-UUID filter) but it is not optional.

**Open question, not resolved:** the `unknown` class (40,339 kNN-VC + 23,373 F5 wavs) matches neither
`Train.csv` nor `Dev.csv` ids/speakers. No Test manifest exists on this pod to check against. Do not
assume it is safe — treat `unknown` as excluded until its provenance is established.

## G-PROV, answerable early for kNN-VC without the audit

kNN-VC is voice conversion of SAP audio, so its text **is** SAP text by construction — the join
succeeding at 70.9% is itself the evidence. G-PROV therefore **FAILS** for kNN-VC (≥80% SAP text):
it carries **no new lexical information**, only new acoustics. Per the runbook's consequence table:
treat purely as augmentation, cap mix fraction hard, expect small gains on the empty lever.

## The pre-registered gates — results

Run: `--n-per-bucket 40` (seeded). The runbook's 400 was restarted down to 40: with 1008 F5 +
419 kNN-VC buckets, 400/bucket is effectively the whole 343k-file corpus over a network FS
(35 min with no output). Percentiles do not need it; 40/bucket = 40,320 F5 + 16,702 kNN-VC
sampled files, 0 unreadable.

| Gate | Verdict | Numbers |
|---|---|---|
| **G-COVER[kNN-VC]** | **PASS** | 21.1% of utts ≤3 words · onset dBFS p25 = **−56.3** (62.4% of files ≤ −45 dBFS) |
| **G-COVER[F5]** | **UNMEASURED — not FAIL** (see below) | text N/A · onset is digital silence |
| **G-EOS** | **PASS** | trailing silence median: F5 320 ms vs kNN-VC 340 ms → **gap 20 ms** (< 300 ms) |
| **G-PROV[kNN-VC]** | **FAIL** | exact SAP-Train text match = **100.0%** |
| **G-PROV[F5]** | **N/A** | transcripts not on this pod |

### Correction to the script's printed verdict (do not read the log literally)
The log prints `FAIL G-COVER[f5]` with `onset_dbfs_p25 = None` and `onset_le_neg45_pct = 100.0`.
Both are artifacts, and the FAIL is not a real measurement:
- F5 onsets are **exact digital zero** — verified directly on 10 sampled files: `nonzero=0/2400`
  samples in the first 100 ms, all at 24 kHz. `_dbfs` returns `-inf`, which `dist()` filters out
  (hence `n=0` / `p25=None`), while the `≤ −45 dBFS` count includes every `-inf` (hence a vacuous 100%).
- **Interpretation that matters:** F5's silent onset is **TTS padding**, a different phenomenon from
  our failure region. The empties are quiet-*but-live* onsets (median −58.5 dBFS **with a loud body**,
  full-utt RMS −29.6 dBFS). Digital silence does not teach the model anything about that.
  F5's coverage of the failure region is therefore **unproven**, and its text is unknown → **D3 stays
  blocked on evidence, not demoted on a false FAIL.**

### Other measured facts worth keeping
- **Sample rates differ:** F5 = **24 kHz**, kNN-VC = 16 kHz. F5 needs resampling before any training use.
- Durations: F5 median 5.23 s (mean 5.52) · kNN-VC median 5.44 s (mean 6.88).
- **Corpus size estimates disagree with the registry.** From the sample: F5 ≈ 214 h (registry said 217.4 h,
  close) but kNN-VC ≈ **389 h** (registry said 150 h). The ~2.6× discrepancy is unexplained — re-derive
  before using kNN-VC hours in any mix-fraction calculation.
- Both corpora carry **leading silence** (F5 lead-sil median 0.46 s; kNN-VC 0.46 s). Training on them
  teaches "nothing happens early", which is the opposite of what the low-latency corner needs. Flagged.

## What this changes in the plan
1. **kNN-VC (D2)** covers the failure region acoustically (PASS G-COVER) but carries **zero new lexical
   information** (G-PROV 100%). It is an **acoustic augmentation only** → cap hard at ≤25% of steps,
   finish on real, expect small gains on the 11.3-pt empty lever.
2. **F5 (D3)** is not demoted and not promoted: unproven. To resolve it, the F5 slot→text table must be
   located off-pod. Until then D3 has no measurable coverage claim.
3. **Both are gated by the leakage filter above** regardless of D0 outcome.
4. G-EOS PASS removes trailing-silence mismatch as the mechanical explanation for the D7 EOS collapse.
   D7's ban therefore stands on its original grounds — the mechanism is not the one D0 could test.

## Files
- `d0_leak_check.json` / `.log` — provenance audit (this session's addition)
- `d0_forensics.json` / `.log` — the three pre-registered gates (audio half + kNN-VC text half)
- scripts: `scripts/build_knnvc_transcripts.py`, `scripts/d0_leak_check.py` (both new this session)
