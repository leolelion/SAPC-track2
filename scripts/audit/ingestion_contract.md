# Codabench ingestion contract audit — SAPC2 Track 2

**Status:** STOPPED at Step 1. Per the audit's hard rule
("If you can't find the ingestion code in the Docker image, STOP and
report. Do not invent the contract from spec docs alone — the spec
describes intent; the code is what runs"), this report documents the
negative result, the methodology, the indirect evidence we *can*
verify, and what remains unverifiable without organizer-supplied
artifacts.

---

## Step 1 — Locate the ingestion program

### Method

Extracted the full `xiuwenz2/sapc2-runtime:latest` filesystem
(10.7 GB) to `/tmp/sapc2_runtime_fs/`:

```
docker create --name ingest_audit xiuwenz2/sapc2-runtime:latest
docker cp ingest_audit:/ /tmp/sapc2_runtime_fs/
docker rm ingest_audit
```

Then searched for any of the ingestion contract's distinctive markers.

### Searches performed

| Pattern | Excluded | Hits in image |
|---|---|---|
| `find -name "ingestion*.py"` | `site-packages` | **0** |
| `find -name "ingest*.py"` | `site-packages` | **0** |
| `find -name "scoring*.py"` | `site-packages` | **0** |
| `grep -rl "accept_chunk"` | `site-packages`, `*.pyc` | **0** |
| `grep -rl "input_finished"` | `site-packages`, `*.pyc` | **0** |
| `grep -rl "set_partial_callback"` | `site-packages`, `*.pyc` | **0** |
| `grep -rl "predict.csv"` | `site-packages`, `*.pyc` | **0** |
| `grep -rl "raw_hypos"` | `site-packages`, `*.pyc` | **0** |
| `grep -rl "Detected splits"` | (none) | **0** |
| `find -path "*/program/*" -name "*.py"` | `site-packages` | **0** |

Also checked common Codabench bundle locations: `/app`, `/scoring`,
`/program`, `/solution`, `/workspace`, `/opt/codabench`, `/eval`. The
image has `/workspace` but it is **empty** (`ls -la
/tmp/sapc2_runtime_fs/workspace` shows only `.` and `..`).

### Cross-check: upstream repos

| Source | Has ingestion or scoring.py? |
|---|---|
| `xiuwenz2/SAPC-template` (upstream/main) — root tree | Only `track1_starting_kit`, `track2_starting_kit`, `utils/`, `steps/`, `evaluate.sh`, `preprocess.sh`. **No** `scoring_program/`, `ingestion_program/`, `codabench_bundles/`, or anything named ingest/score. |
| `track1_starting_kit/` | Only baseline `model.py` examples + `local_decode.py`. |
| `track2_starting_kit/` | Same shape: baselines + `local_decode.py`. |
| `utils/evaluate.py` | Local scoring helper. Docstring says: *"ASR Evaluation – same pipeline as scoring.py, without Codabench API."* This is a **local equivalent**, not the Codabench `scoring.py` itself. |

### Conclusion of Step 1

**The Codabench ingestion and scoring programs are not public
artifacts.** They are mounted by the Codabench platform at evaluation
time from a competition-internal bundle held by the organizers
(`xiuwenz2@illinois.edu`). They are not in:

- the runtime Docker image,
- the upstream `xiuwenz2/SAPC-template` GitHub repo,
- the Track 1 or Track 2 starting kits,
- our local `utils/`.

The audit cannot proceed to Steps 2–5 as designed because Step 2
requires reading the ingestion source line by line. Per the audit's
hard rule, fabricating answers from the spec README is not acceptable.

---

## What we CAN verify (indirect evidence)

These are facts we can establish without reading the ingestion code,
and they bound the contract somewhat.

### From the scoring stage failure log (submissions 778806, 779889, 781708)

```
=== Running scoring.py ===
sclite: /app/program/bin/sclite
Detected splits: []
ERROR: No .predict.csv files found
```

What this tells us:

- **F-1.** Codabench's `scoring.py` runs at path `/app/program/scoring.py`
  (the sclite peer is at `/app/program/bin/sclite`).
- **F-2.** Scoring scans **some** directory for files matching the glob
  `*.predict.csv` and extracts a "split" identifier from each filename.
  An empty match list means our ingestion stage produced none of those
  files in the scanned location.
- **F-3.** The scoring program produces no `[scoring]` output beyond
  what is shown — there is no per-utt diagnostic, no parser error, no
  "found N files but couldn't parse them." The list is genuinely empty.

What this does **not** tell us:

- The absolute directory scoring scans (`/app/predictions/`?
  `/app/output/`? `/tmp/codabench_output/`?). The error message does
  not include the directory.
- Whether the filename pattern is exactly `<Split>.predict.csv` or
  something more permissive.

### From `track2_starting_kit/README.md` (spec, lines 1–117)

Documented intent — useful but NOT authoritative re: actual ingestion
behavior. Treated as a constraint, not a source of truth:

- **Spec-1.** The `Model` class implements 5 methods: `__init__`,
  `set_partial_callback`, `reset`, `accept_chunk`, `input_finished`.
- **Spec-2.** Two passes per audio file: Pass 1 batch
  (multiprocess parallel workers), Pass 2 streaming
  (two-thread real-time simulation).
- **Spec-3.** Chunks are 100 ms (1600 samples at 16 kHz), float32, mono.
- **Spec-4.** `setup.sh` runs **before** the model is loaded.
- **Spec-5.** `requirements.txt`, if present, is auto-installed via
  `pip install -r requirements.txt` after `setup.sh`.

Open intent-vs-implementation gaps:

- Where the multiprocessing is rooted (`fork` vs `spawn`) and how many
  workers.
- Whether the audio chunk is *exactly* 1600 samples or whether the last
  chunk per utterance is shorter.
- Whether `set_partial_callback` is called once globally or once per
  utterance, and whether the no-op callback for Pass 1 is set before
  or after `__init__`.
- Whether `reset()` is called before the first utterance.

### From our local `local_decode.py`
([track2_starting_kit/local_decode.py:1-205](../../track2_starting_kit/local_decode.py))

This is **a local equivalent**, NOT a copy of Codabench's ingestion.
Differences are exactly what this audit was meant to surface — and we
cannot surface them without the real ingestion code. What we can say
about `local_decode.py` itself:

- Lines 19–25: `setup_environment` runs `setup.sh` as a
  `subprocess.check_call` — separate process; packages installed into
  a venv are **not** automatically visible to the parent Python.
- Line 172: `sys.path.append(str(args.submission_dir))` — adds
  submission dir to `sys.path` **before** importing Model.
- Line 177: `from model import Model` — import in the parent Python.
- Lines 60–70 (`run_batch_decode`): Pass 1 is **single-process,
  serial** locally. README says real Codabench Pass 1 is multiprocess —
  we cannot reproduce that locally.
- Lines 73–147 (`run_streaming_decode`): Pass 2 spawns two threads
  exactly matching the README spec.
- Lines 183–188: predict CSV is written wherever `--out-csv` points,
  with columns `["id", "raw_hypos"]` (line 186). We pick the path; real
  Codabench picks the path.

### From our `utils/evaluate.py`
([utils/evaluate.py:1-243](../../utils/evaluate.py))

Documented as "same pipeline as scoring.py, without Codabench API"
(line 3) — i.e., scoring stage equivalent. Reads predict.csv via
`--hyp-csv` (line 30) with column `raw_hypos` (line 36, also the
default). The schema of the CSV is established: `id`, `raw_hypos`.
**Where** Codabench's scoring.py finds the CSV remains unknown.

---

## What remains UNVERIFIABLE without the real ingestion code

These are the contract points Step 2 of the audit would have answered.
Marking each as **unknown** rather than guessing:

| Point | Question | Status |
|---|---|---|
| A1 | Audio path / manifest format / location | unknown |
| A2 | How "split" is identified from manifest filename / dir | unknown |
| A3 | Whether ingestion checks paths exist before running Model | unknown |
| B1 | Import path / `sys.path` setup before `from model import Model` | unknown |
| B2 | cwd of setup.sh, ingestion, and scoring | unknown |
| B3 | Whether setup.sh is `check_call`'d or sourced; required exit code | unknown |
| C1 | Chunk size: exactly 1600 or variable on last chunk; dtype | unknown |
| C2 | Whether `reset()` is called before or after each utterance | unknown |
| C3 | `set_partial_callback` timing — once or per-utt | unknown |
| C4 | How "end of audio" is signaled | unknown — likely `input_finished()` per spec |
| C5 | What ingestion does with `input_finished()`'s return value | unknown |
| **D1** | **Directory where predict.csv is written** | **unknown — root cause candidate** |
| **D2** | **Exact filename pattern** | **unknown — root cause candidate** |
| D3 | predict.csv schema (header? column order? encoding?) | unknown — `id,raw_hypos` per scoring side |
| D4 | Handling of empty hypotheses | unknown |
| D5 | Whether ingestion requires `partial_results.json` or that's local-only | unknown |
| E1 | Behavior when `Model()` raises | unknown |
| E2 | Behavior when `accept_chunk` / `input_finished` raises | unknown |
| E3 | Per-utt or per-submission timeout | unknown (15000 s submission limit per README) |
| E4 | Memory / CPU limits in code | unknown |
| F1 | fork vs spawn; worker count; Model per-worker or shared | unknown |
| F2 | Subset-per-worker vs all-per-worker with merge | unknown |
| F3 | Whether Pass 1 / Pass 2 are actually implemented as in spec | unknown |

The two **D-row unknowns are the strongest root-cause candidates** —
"No .predict.csv files found" is a direct symptom of the wrong
directory or wrong filename, and our local test cannot distinguish
that mismatch (we always pass `--out-csv` and write where we choose).

---

## Step 5 — Hypotheses we CAN articulate (without inventing the contract)

These are ordered by how well they explain *all three* observed
failures (778806 / 779889 / 781708), without requiring us to invent
specifics of the ingestion code. Each hypothesis names what evidence
in code we *would* need to confirm it. None are confirmed.

### H1 — Ingestion does not call `Model.__init__` at all on this submission

**Symptom fit:** Strong. If Model is never instantiated, no
`[CPU_DIAGNOSTIC]` / `[MEM_DIAGNOSTIC]` / `[nemotron_streaming]`
lines are emitted, no predictions are made, no predict.csv is
written, and scoring reports `Detected splits: []`.

**Sub-hypotheses** (any of these would produce the same
visible symptom):

- **H1a.** `setup.sh` exits non-zero on the eval VM (e.g. HuggingFace
  download fails because the VM has no outbound network), and
  ingestion treats setup failure as fatal. Our local test had network;
  the eval VM may not.
  - Code we'd need: ingestion's setup invocation (does it
    `check_call`? does it short-circuit on non-zero?).
  - Code in our submission: `track2_starting_kit/nemotron_streaming/setup.sh:11` (`set -euo pipefail`) makes the HF download stage fatal.
- **H1b.** Ingestion expects `setup.sh` to install packages in the
  **base** Python (like Track 1's reference baselines), not in a local
  venv. Our `model.py:42-50` injects the venv into `sys.path`, but if
  the injection runs in a worker that has its own `sys.path` already
  locked (e.g. a `multiprocessing.spawn`-style worker that re-imports
  fresh), the injection might race with the import.
  - Code we'd need: ingestion's `from model import Model` site +
    its `multiprocessing` mode.
- **H1c.** The `model.py` import itself fails on the eval VM due to a
  library incompatibility that does not appear in our local
  `xiuwenz2/sapc2-runtime:latest` container test. Possible cause: a
  drift between the image tag pulled locally (build date 2026-02-18,
  per `docker images`) and the image actually used by Codabench.
  - Code we'd need: nothing in ingestion; just the build hash of the
    eval-side image.

### H2 — Ingestion writes predict.csv to a directory we don't write to

**Symptom fit:** Strong. If Model ran but ingestion writes
predict.csv to (say) `/app/predictions/` while scoring reads from
`/app/output/`, scoring reports the same `Detected splits: []`.

This hypothesis is **bounded by**: ingestion's directory choice is not
something *our* submission controls — Codabench's ingestion picks it.
So this would explain why three submissions (with three *different*
model implementations) fail identically. The hypothesis essentially
says "Codabench ingestion is broken or has a bug we can't fix from the
submission side," which is implausible given other competitors must be
succeeding.

Sub-hypothesis where the submission *could* affect this:

- **H2a.** Our `Model` writes predict.csv itself (it shouldn't), and
  in doing so puts files where ingestion does not expect to also
  write them, OR ingestion doesn't write predict.csv because Model
  already did and ingestion sees the file count mismatch and bails.
  - Check our model.py: grep `open(.*predict.csv|partial_results.*)`
    — no matches. Our Model does not write any files. Hypothesis
    falsified locally.

### H3 — Ingestion fails inside the per-worker subprocess and emits no log

**Symptom fit:** Moderate. Multiprocess Pass 1 workers may fail
silently if the parent doesn't propagate child stdout/stderr or
exceptions back to the main log. A common case: child process raises
during import, the pool reports a generic error, and the main log
shows nothing model-side. The parent then aggregates "no predictions
from any worker" and writes nothing.

Sub-hypotheses:

- **H3a.** Worker fork happens after the parent has loaded our Model,
  but our `model.py:42-50` venv injection is happening in the **parent**
  only — child workers may have a stale `sys.path` from before
  injection. (Unlikely with fork — children inherit; but possible
  with spawn.)
- **H3b.** ORT session handles aren't fork-safe and the children
  crash on first `session.run()`. ORT has known issues with `fork()`
  on some platforms; the documented mitigation is to construct
  sessions per-process *after* fork. Our v3 (uncommitted)
  `_ensure_loaded()` lazy-load actually moves construction to first
  inference — so v3 *might* fix H3b incidentally — but this remains
  a hypothesis, not a finding.

### Hypotheses we are removing because we have ruled them out

| Hypothesis | Why ruled out |
|---|---|
| Wrong predict.csv schema (e.g. wrong column name) | We confirmed schema = `id,raw_hypos` via `utils/evaluate.py:30,36` and matched in our writes. |
| Wrong audio path / manifest read by Model | Our Model never reads files — ingestion feeds chunks. |
| Local-test contradicts Codabench-ingestion | Demonstrated: local `local_decode.py` against our submission produces a valid predict.csv (`Test_tiny.predict.csv` from the prior run, 5 utts, real ASR output). The bug is upstream of our Model when running on Codabench. |
| OOM during inference | Tracemalloc + RSS data show per-process steady state ~2.1 GB, well under any reasonable eval VM ceiling. |

---

## What to do next

The audit's stated outcome ("if you can't find ingestion code, email
organizers is the next move") matches our situation. The email has
already been sent (2026-06-06). Recommended additions when the
organizer replies, or when sending a follow-up if a reply doesn't
arrive:

1. **Ask for the ingestion program's stdout/stderr** for submission
   781708 (most recent failing). The current "Scoring log" is only
   the scoring stage.
2. **Ask for the exact directory** Codabench's `scoring.py` scans for
   `*.predict.csv`. (Or, equivalently, the exact directory their
   ingestion writes to. They should match.)
3. **Ask whether their Pass 1 ingestion uses `fork` or `spawn`,** and
   **how many workers** it spawns on the eval VM.
4. **Ask whether the eval VM has outbound network access** during
   setup.sh, and whether the `xiuwenz2/sapc2-runtime:latest` image
   tag used at evaluation time matches the public image
   (build date 2026-02-18 per our `docker images`).

Until any of those four are answered, no further submissions provide
new information that we don't already have. The `hello_world_diagnostic.zip`
(SHA256 `b7e5ad74…`, committed `7718bb3`) remains the cheapest single
probe if we choose to spend a slot before the organizers reply.

---

## Audit log

| Time | Action |
|---|---|
| Step 1 start | Extracted `xiuwenz2/sapc2-runtime:latest` filesystem (10.7 GB). |
| Step 1 search | Ten `find` / `grep` patterns over the full FS — zero hits. |
| Step 1 cross-check | Inspected upstream root tree + Track 1 + Track 2 starter kits — no Codabench bundles. |
| Step 1 stop | Per audit hard rule, halted before Step 2. |
| Report | Wrote this doc; no model changes; no submission preparation. |
