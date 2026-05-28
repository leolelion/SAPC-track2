# SAPC2 Track 2 — Submissions log

## v1 — `nemotron-int8static-7010-7-baseline-v1`

**Status:** Ready to upload, awaiting user action (the Codabench web UI
flow can't be done from here).

**Date prepared:** 2026-05-28
**Zip:** `track2_starting_kit/nemotron_streaming.zip`
**Zip SHA256:** `dfdfe373bf3e73f2c6d4f2f4748ea6ecae1595f873fb6c1d3b4eb6c312bfccad`
**Zip size:** 9,727 bytes (model.py + config.yaml + setup.sh + README.md)
**Last commit touching the zip:** `4204de1` (Phase 4 validation)
**Git state at submission time:** see `git log -1` in the repo

### What it is

The conservative baseline:

- Wrapping danielbodart's prebuilt **int8-static** ONNX encoder of
  `nvidia/nemotron-speech-streaming-en-0.6b` + the FP32 decoder, at
  `att_context_size=[70, 6]` (560 ms model chunks).
- Streaming via our 5-method SAPC2 `Model` class (`model.py`).
- Greedy RNN-T decode; no beam, no LM, no pass-1/pass-2 differentiation,
  no finetune.
- `setup.sh` downloads weights from danielbodart's HF repo at a pinned
  revision so the zip itself stays small.
- `model.py.__init__` prints a `[CPU_DIAGNOSTIC]` line to stdout and
  writes `/tmp/cpu_diagnostic.json` with lscpu + cpuinfo head + meminfo +
  loadavg + a 1024² × 20-iter matmul benchmark. **This is the only way
  we'll learn the eval-VM CPU spec.** Save the captured stdout.

### Validated numbers

| Set | Scorer | CER% | WER% | TTFT P50 (ms) | TTLT P50 (ms) | compute-RTF P50 | Source |
|---|---|---:|---:|---:|---:|---:|---|
| Dev_streaming (123 utts) | sclite | 22.31 | 28.08 | 1592 | 270† | 0.243 (4 threads) | Phase 3 |
| Dev_10k (10,521 utts) | sclite | **21.59** | **27.46** | — | — | 0.120 (16 threads, batch) | Phase 4b |
| Dev_10k (re-validated from clean setup.sh, 4 threads) | sclite | 22.31 | 28.08 | 1590 | 242† | 0.218 | Phase 4 |

† TTLT measured on a contended shared host (load avg 15-31 during run).
Inflated ~3× vs the same script's April measurement on the same pod.
Submission TTLT will reveal the eval-VM true value.

### Apples-to-apples vs Zipformer kroko (the existing dev100 incumbent)

| Model | Dev_10k CER% | Dev_10k WER% | Bundle | Notes |
|---|---:|---:|---|---|
| **nemotron int8-static** | **21.59** | **27.46** | 876 MB | this submission |
| sherpa_zipformer kroko | 23.92 | 33.57 | ~70 MB | re-scored same pipeline |

Nemotron wins by 2.33 CER / 6.11 WER on 10,521 utts. Lead is ~3× the
binomial noise floor.

### Pre-submission gate status

- [x] Dev_streaming validation (Phase 3): CER 22.31, RTF gate PASS
- [x] Clean setup.sh validation (Phase 4): CER 22.31 reproduces byte-for-byte
- [x] Dev_10k validation (Phase 4b): CER 21.59, beats Zipformer by 2.33
- [x] Empty-prediction triage (Phase 4b post): H1 rejected, H2 confirmed,
      no structural fix to ship
- [x] Forced-decode diagnostic (Phase 4c): forcing makes 0 of 1463 empties
      better, confirms ship-honest decision
- [x] Zip SHA256 captured, unchanged since validation
- [ ] **User uploads zip to Codabench** ← only remaining step

### Expectations on Test1

- Estimate: **20–25 CER**, given distribution shift vs the 34.59 CER
  official baseline (sherpa_zipformer standard).
  - In-range = success.
  - <20 = upside (model + scoring conventions align unexpectedly well).
  - >25 = investigate distribution shift (e.g. Test1 has more CP than Dev_10k).
- compute-RTF: TBD per the [CPU_DIAGNOSTIC] line. On our host with 4
  threads we measured 0.218. The eval VM is almost certainly slower per
  core; estimate 0.4–0.7 reasonable, anything <1.0 passes the gate.

### Codabench upload checklist (for the user)

1. Confirm SHA256 matches: `shasum -a 256 track2_starting_kit/nemotron_streaming.zip`
   should print
   `dfdfe373bf3e73f2c6d4f2f4748ea6ecae1595f873fb6c1d3b4eb6c312bfccad`.
2. Submission name on Codabench: `nemotron-int8static-7010-7-baseline-v1`.
3. **Screenshot the submission form before clicking submit**.
4. After it scores (~3 days per the spec for Test1):
   - Screenshot the leaderboard row.
   - Save the eval-VM stdout — search for the line beginning
     `[CPU_DIAGNOSTIC]`. That's the CPU spec we've been guessing at all
     week.
   - Save the official scoring report if Codabench provides one.
5. Commit both screenshots + the CPU diagnostic JSON to
   `submissions/v1/` and update this row with:
   - Codabench submission ID
   - Test1 CER / WER
   - eval-VM CPU model and matmul_1024_20iter_ms
   - Actual measured RTF

### Deliberately NOT in this submission

- pass-1 / pass-2 differentiation trick
- int4 / k-quant alternates
- alternate att_context configs ([70,1] / [70,13])
- forced-decode (diagnostic only — confirms shipping empty was correct)
- finetuned encoder

These are all candidates for v2 once we have a real Test1 anchor and the
eval-VM CPU spec.
