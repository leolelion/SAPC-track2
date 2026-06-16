# Forensic diff: v7 submission vs verified-working streaming_zipformer baseline

Audit date: 2026-06-16 (task dated 2026-06-09)
Analyst: automated forensic diff, investigation-only (no submission, no code change)

- **baseline** = `track2_starting_kit/streaming_zipformer.zip`
  SHA256 `f7876a006d9ba9730c6c38236b1556b005b8abfb293cf450cb72ea6c7f99aa45`,
  7331 B, commit `ce02f68` (2026-02-19). **Confirmed working on Codabench by the
  organizer (re-uploaded 2026-06-09).**
- **v7** = `track2_starting_kit/nemotron_streaming_v7_minimal.zip`
  SHA256 `868b37328b31519fab83229168807eb87acc904bb99805da539e98adb7714e37`,
  11328 B, built 2026-06-09. **Failed; ingestion tab reported empty.**

---

## Executive summary

**The zip is not the problem.** A byte/structure-level diff finds the v7 archive
**structurally near-identical** to the working baseline in every dimension a
Codabench unpacker or pre-ingestion validator could care about:

- Both unzip **flat** to the program root (no wrapping directory) — the prime
  suspect named in the task brief is **absent**; neither has a wrapper.
- Both are clean Info-ZIP archives: deflate-compressed, OS=Unix, version-made-by
  3.0, min-extract 2.0, no data descriptors, **no `__MACOSX`, no `.DS_Store`, no
  directory entries**, no encryption, no comments. Both pass `unzip -t`.
- v7 is a **strict superset** of the baseline's files: same `model.py` + `setup.sh`
  + `config.yaml`, plus a benign `README.md`. Nothing required is missing.
- `setup.sh` matches baseline in every launch-relevant property: `#!/bin/bash`,
  `set -e`, same `$DIR` idiom, same final echo, no explicit `exit`, LF endings,
  trailing newline.
- `model.py` exposes a **byte-for-byte equivalent caller interface**: `class Model`
  with the exact 5 method signatures and `str` return types, same method order, no
  module-level print/threading/`__main__`. v7 even imports *less* at module load.
- **No manifest is required and the baseline has none** — killing the leading
  hypothesis in the task brief (missing metadata/submission.json). A no-manifest
  submission is the one that currently *works*.

**Leading conclusion = the task's "worst case":** our submission is structurally
sound and indistinguishable from the baseline in everything that should matter to
pre-ingestion. The empty ingestion tab for v6/v7 is therefore **not explained by
the zip**. Combined with the organizer's own admission that they *also* cannot
retrieve ingestion logs for 788058, the evidence points away from our artifact and
toward a **platform-side / submission-pipeline issue** (queueing, log capture, or
an account/competition-phase condition) that no change to the zip's bytes will fix.

This is a real, useful result: it exonerates the archive and redirects the next
move from "rebuild the zip differently" to "press the organizer with this diff as
proof of structural parity."

---

## Ranked differences (most→least plausible as a pre-ingestion gate)

Every item below is real and documented. **Honest caveat: even the #1 item is
low-plausibility** — none rises to a convincing gate, which is itself the finding.

### #1 — setup.sh executable bit: 755 (v7) vs 644 (baseline)
- **What:** v7 `setup.sh` has Unix mode `100755` (`-rwxr-xr-x`); baseline is
  `100644` (`-rw-r--r--`). (Evidence: zipinfo central-dir "Unix file attributes".)
- **Mechanism (why it *could* matter):** if a strict validator inspected stored
  permission bits, the two differ. **But this cuts the wrong way:** the *working*
  file is the non-executable one, and Codabench launches setup via `bash setup.sh`
  (mode-independent). An executable bit is strictly more permissive, not less.
- **Plausibility:** very low. Listed #1 only because it is the single
  launch-adjacent metadata field that differs.
- **Cost to fix:** trivial — `chmod 644 setup.sh` before zipping (1 command).
- **Safe?** Yes. Cannot regress anything; bash still runs it.

### #2 — Extra file `README.md` in v7 (4 entries vs 3)
- **What:** v7 contains `README.md` (2877 B) that baseline lacks.
- **Mechanism:** an over-strict validator could reject unexpected files. No
  evidence Codabench does this; the spec explicitly allows "Other supporting
  files," and many working submissions carry extras.
- **Plausibility:** very low.
- **Cost to fix:** trivial — omit README.md from the zip.
- **Safe?** Yes — it is documentation only, unused at runtime.

### #3 — config.yaml line endings: LF (v7) vs CRLF (baseline)
- **What:** v7 `config.yaml` is LF; baseline is CRLF.
- **Mechanism:** none credible. The file is parsed by the submission's own
  `model.py` (OmegaConf), not by Codabench, and the **CRLF** variant is the one
  that works. Direction of the difference makes this a non-issue.
- **Plausibility:** effectively nil.
- **Cost to fix:** trivial, but pointless.
- **Safe?** N/A.

### #4 — 0x7875 uid/gid + timestamps (cosmetic zip metadata)
- **What:** baseline zipped on Linux (uid 68894/gid 202, Feb timestamps); v7 on
  macOS (uid 501/gid 20, Jun timestamps). Same Info-ZIP field structure.
- **Mechanism:** none — these fields are informational; unzip ignores them for
  extraction.
- **Plausibility:** nil.
- **Cost to fix:** would require zipping on Linux; not worth it.
- **Safe?** N/A.

### Implementation deltas that are NOT pre-ingestion candidates (for completeness)
These differ but execute **during** ingestion (after setup.sh banner would have
printed), so they cannot explain an *empty* ingestion tab:
- `setup.sh` installs (onnxruntime/onnx vs k2/kaldifeat/icefall+git-clone).
- `model.py` engine (ONNX Runtime vs icefall) and its `/proc` reads +
  `/tmp/cpu_diagnostic.json` write inside a runtime diagnostic helper.
- Different `config.yaml` schemas (ONNX bundle vs zipformer knobs).
They remain relevant **only if** the "empty ingestion tab" observation is itself
mistaken (e.g., logs ran but weren't surfaced).

---

## What this rules in / out

- **Ruled out as the gate:** wrapping directory, macOS artifacts, missing required
  file, missing manifest/metadata, compression method, zip corruption, shebang/
  `set -e`/exit/newline issues in setup.sh, `Model` interface shape mismatch.
- **Not excludable by a zip diff (next frontier):** anything Codabench does to the
  submission *outside* the archive — platform queueing, ingestion-log capture,
  account/competition-phase state, or a runtime failure during model-load that the
  empty tab hid. The organizer's inability to fetch 788058's ingestion logs is
  consistent with a platform-side capture gap rather than an artifact defect.
