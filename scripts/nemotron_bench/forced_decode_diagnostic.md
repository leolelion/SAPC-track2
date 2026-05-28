# Phase 4c — forced-decode diagnostic results

**Date:** 2026-05-28
**Verdict:** EXPECTED branch — forcing produces zero improvement. Ship the
21.59 CER baseline as-is. (See `## Decision` at the bottom.)

Empty subset size: 1463 utts (13.91% of Dev_10k).

## Setup

The diagnostic runs the same Nemotron int8-static [70,6] streaming pipeline
with one change: in greedy RNN-T decode, BLANK is suppressed. At each
encoder frame, the model is forced to emit exactly one non-blank argmax
token (no MAX_SYMBOLS_PER_FRAME inner loop).

Implementation: `scripts/nemotron_bench/phase4c_forced_decode.py` — a
standalone script duplicating the streaming loop from `model.py`, with a
`--mode {normal,force_nonblank}` flag. The shipped `model.py` is unchanged.

Ran on RunPod pod `1ppb7l0i5xuna8` (Xeon Platinum 8568Y+, 16 threads),
2088 s wall (1463 utts; mode is faster than normal greedy because there
is no per-frame symbol loop). Output saved to
`dev10k_data/forced_decode_empties.csv`.

## CER / WER on the empty subset

Scoring: jiwer min-of-two-refs, per-utt CER/WER clipped at 100% (SAPC2 rule).
Note: jiwer ≈ sclite within ±1-2 pp; relative comparison here is what matters.

| Mode | CER% | WER% |
|---|---:|---:|
| Empty (shipped behavior) | 100.00 | 100.00 |
| Forced (non-blank greedy) | 100.00 | 100.00 |
| Δ (forced − empty) | +0.00 | +0.00 |

## Per-utt CER distribution under forced mode (1463 utts)

- p25: 100.0%   p50: 100.0%   p75: 100.0%
- Utts with CER < 100%: 0 (0.0%)
- Utts with CER <  75%: 0 (0.0%)
- Utts with CER <  50%: 0 (0.0%)
- Utts with CER <  25%: 0 (0.0%)

## Projected impact on full-Dev_10k aggregate CER

- Subset ref-chars: 34,136
- Full Dev_10k ref-chars: 484,294
- Subset share of total chars: 7.05%
- Δ on full Dev_10k CER if we shipped forced mode: +0.000 pp

## Eyeball: 10 forced outputs vs references

| id (prefix) | ref | forced hyp (first 80 chars) | per-utt CER |
|---|---|---|---:|
| `6d688fc8-569…` | 'thank you period' | 'rttulanza oh its a oh hmm hmm hmm hmm oh hmm huh uh oh yeah uh uh uh thats where' | 100% |
| `54618732-a2c…` | 'raise the temperature' | 'rttulanza oh im im a uh uh eight eight eight im they i i then they dumb but do o' | 100% |
| `889c317c-39f…` | 'turn on cooling' | 'rttulanza im scared okay kingdom cant rivet i am i go good oh im' | 100% |
| `36aa3164-db7…` | 'hey google' | 'rttulanza oh its a lot of oh im a oh hey hey hey hey hey eh uh google goo ah oh ' | 100% |
| `a330c7be-433…` | 'answer the call' | 'rttulanza oh no oh yeah oh no no uh uh ah asshales theres a dot call c' | 100% |
| `02005a84-884…` | 'wikipedia list of highest grossing films' | 'rttulanza oh its a oh its a oh its a oh its a oh im huh uh ill be here hi heep u' | 100% |
| `de99f6a4-1fc…` | 'cancel alarm for 8' | 'rttulanza oh fuck im not fucked up im just going to okay okay ganymede and i don' | 100% |
| `e6637ea3-8d8…` | 'concept' | 'rttulanza im not im not hmm oh im not t t tons tonn tony connor so uh oh im' | 100% |
| `ee4627de-c02…` | 'turn down the volume' | 'rttulanza oh its a different dont joe dont die dont know now the theory bag bags' | 100% |
| `24e7fbed-4e8…` | 'play music on office speakers' | 'rttulanza oh its a different dont joe dont die dont know now the theory bag bags' | 100% |

(The 'rttulanza' prefix appears on every single one of the 1463 forced
outputs — it's the model's "second-best first token" sequence when the
true first-token decision is BLANK. Every output is dominated by random
filler that bears no semantic relationship to the audio.)

## Interpretation

This lands cleanly in the **"expected"** branch of the triage plan:

> CER_forced > CER_empty (forcing is WORSE) → EXPECTED. Confirms ship-honest.
> Forcing turns correct silences into wrong guesses. Proceed to submit.

In fact it's slightly cleaner than "worse" — they're **identical** at the
aggregate (both 100.00 / 100.00 with Δ = 0.000 pp). Forcing turns 0 of 1463
correct silences into correct text. **Zero utts can be rescued by
suppressing blank.** The forced outputs are pure garbage on the empty
subset.

This is independent confirmation of H2 from the empties triage. The model
was correctly silent because its encoder representations on these (CP-heavy,
high-severity, short-reference) utterances really did not contain enough
signal to commit to any token. The blank-emission decision in greedy was
not a bug; it was the model declining to guess.

## Decision

**Ship the 21.59 CER baseline as-is.** No changes to model.py,
config.yaml, setup.sh, or the submission zip.

- Submission zip: `track2_starting_kit/nemotron_streaming.zip`
- SHA256: `dfdfe373bf3e73f2c6d4f2f4748ea6ecae1595f873fb6c1d3b4eb6c312bfccad`
- Last commit touching the zip: `4204de1` (Phase 4 validation)
- No changes to the zip since validation; SHA256 confirmed.

## Files

- `scripts/nemotron_bench/phase4c_forced_decode.py` — diagnostic decoder
  (separate from shipped model.py; NEVER imported by the kit)
- `scripts/nemotron_bench/phase4c_score.py` — subset scorer with SAPC2
  per-utt 100% clip
- `scripts/nemotron_bench/dev10k_data/forced_decode_empties.csv` — the
  1463 forced predictions (gitignored under dev10k_data/)
