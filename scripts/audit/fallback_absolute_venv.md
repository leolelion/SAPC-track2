# Fallback design — absolute-path venv (`/opt/nemotron_venv/`)

**Status:** **DESIGN ONLY. No code, no zip, no submission.** Reviewed
after v4-novenv-probe (SHA `177db21f…`) lands a Codabench result.
Decide whether to build at that point based on the outcome.

**Companion docs:**
- [scripts/audit/ingestion_contract.md](ingestion_contract.md) — why
  we can't see the ingestion code, what we deduce indirectly.
- [scripts/audit/reference_submission_diff.md](reference_submission_diff.md)
  — why H1b is the leading hypothesis and what the only-known-working
  Track 2 baseline does.

---

## Purpose

This document captures the design of a *second* probe / fallback
production variant that would isolate our deps from base-Python
*without* using a submission-relative venv. It is the natural next
step if either of the following is true after v4 scores:

1. **v4 succeeds and the production version needs cleaner isolation.**
   Shipping NeMo + ORT into the system Python's `site-packages`
   (what v4 does) works for the probe, but commingles our deps with
   whatever Codabench's runtime image carries. A future submission
   that wants to be robust against image rebuilds — or against future
   `nemo_toolkit` versions that fight with system packages — should
   sit in an isolated venv.

2. **v4 fails for a dep-conflict reason specific to global install.**
   Concrete examples: NeMo silently upgrades `numpy` past our `<2`
   pin via a transitive dep; some package overrides
   `huggingface_hub`'s `hf_hub_download` signature; ORT 1.26 expects
   a different protobuf than the system has. Any of these would
   manifest as a successful pip install but a failing import or a
   weird runtime crash that doesn't appear locally.

3. **v4 succeeds but the production version needs to match Track 1's
   pattern for parity with organizer expectations.** The official
   Track 1 NeMo baselines (`parakeet`, `canary_qwen`) explicitly use
   this absolute-path venv pattern; building the production version
   the same way reduces the surface area where Track 2 ingestion
   could behave differently from Track 1 in a way that bites us.

---

## What this pattern actually is (copying the upstream Track 1 baselines)

Source of truth: the published Track 1 baseline submissions in
[track1_starting_kit/parakeet/](../../track1_starting_kit/parakeet/)
and [track1_starting_kit/canary_qwen/](../../track1_starting_kit/canary_qwen/).
The pattern is **byte-for-byte identical** between them, just with
the venv directory renamed.

### setup.sh shape (from parakeet)

Locations in
[track1_starting_kit/parakeet/setup.sh](../../track1_starting_kit/parakeet/setup.sh):

- **Line 9** — `python3 -m venv /opt/parakeet_venv` — creates the venv
  at a **fixed absolute path under `/opt/`**, NOT inside the
  submission directory.
- **Line 10** — `/opt/parakeet_venv/bin/pip install --upgrade pip -q`.
- **Lines 13–20** — install all deps using the venv's pip:
  `/opt/parakeet_venv/bin/pip install --no-cache-dir nemo_toolkit[asr]==2.5.3 …`.
- **Lines 22–29** — verify the install by running
  `/opt/parakeet_venv/bin/python3 -c "import nemo.collections.asr as
  nemo_asr; …"`.

Key shape choices we'd inherit:

- The venv path is **constant** across all runs — no glob, no nested
  layout detection. No `${DIR}/venv` trickery; no need for our
  current `_find_venv_python` helper. Path is just `/opt/<name>_venv/`.
- `--system-site-packages` is **not** passed. The venv is fully
  isolated by default — does NOT see system NumPy / Torch / etc.
  Pros: avoids the kind of subtle version-mix we hit with global
  install. Cons: must install torch ourselves into the venv (heavier
  install, may conflict with CUDA setup on eval VM).
- Track 1 baselines install **torch + torchaudio + torchvision**
  fresh inside the venv (parakeet/setup.sh:20). They do not try to
  preserve the system torch.

### model.py shape (from parakeet)

Locations in
[track1_starting_kit/parakeet/model.py](../../track1_starting_kit/parakeet/model.py):

- **Lines 8** — `import sys, os` (stdlib, no venv needed).
- **Line 10** — `_venv_site = "/opt/parakeet_venv/lib/python3.11/site-packages"`
  — **hardcoded absolute path** to the venv's site-packages.
- **Lines 11–12** — `if os.path.isdir(_venv_site):
  sys.path.insert(0, _venv_site)` — same `sys.path.insert(0, …)`
  trick as our current `nemotron_streaming/model.py:42-50`, but with
  the path pinned to `/opt/...` rather than discovered via glob
  relative to `__file__`.
- **Lines 14–15** — `import torch` / `import nemo.collections.asr as
  nemo_asr` — heavy imports that succeed because the venv path was
  injected first.

Key shape choices we'd inherit:

- The path is `os.path.isdir`-gated. If the directory doesn't exist
  (e.g. we're running model.py before setup.sh has executed), the
  injection silently skips. This means model.py is still importable
  for offline inspection. (Our nemotron_streaming/model.py does the
  same gating via the glob's truthiness.)
- **Hardcoded Python ABI version (`python3.11`)**. The parakeet kit
  assumes 3.11 because the runtime image ships 3.11. If the eval VM
  ever moves to 3.12, this breaks. This is a real risk to take on
  knowingly — accept the implicit ABI tie-down for the simplicity
  win.

---

## Concrete deltas vs `nemotron_streaming_novenv/` (the v4 probe)

If we built this as `nemotron_streaming_absvenv/` (hypothetical kit
name), the deltas would be exactly:

### `setup.sh` deltas

| From v4 (`nemotron_streaming_novenv/setup.sh`) | To absolute-venv variant |
|---|---|
| `pip install … nemo_toolkit[asr]` into base Python | `python3 -m venv /opt/nemotron_venv` first; then `/opt/nemotron_venv/bin/pip install …` |
| Torch pin via constraint file against system torch | Either (a) reuse the same constraint pinning torch to the system version + `--system-site-packages` to inherit base torch, OR (b) install torch fresh into the venv (matches Track 1's choice). Track 1 chose (b). |
| Verification step uses base `python3 -c "..."` | Verification step uses `/opt/nemotron_venv/bin/python3 -c "..."` |
| Smoke test invocation uses base `python3` | Smoke test uses `/opt/nemotron_venv/bin/python3` |
| `pip list` at end uses base python | `pip list` at end uses `/opt/nemotron_venv/bin/pip list` |

### `model.py` deltas

| From v4 | To absolute-venv variant |
|---|---|
| `# Section 1: …NO VENV` | `# Section 1: …absolute-path venv injection` |
| Direct stdlib imports → `[SETUP_VERIFY]` → direct heavy imports | Same flow PLUS `_venv_site = "/opt/nemotron_venv/lib/python3.11/site-packages"; if os.path.isdir(_venv_site): sys.path.insert(0, _venv_site)` BEFORE the `[SETUP_VERIFY]` block. The verify block must run with the venv on sys.path so it reports the venv's versions, not the system's. |
| `[SETUP_VERIFY]` block unchanged | `[SETUP_VERIFY]` block extended to log `_venv_site` existence + which `site-packages` `torch.__file__` resolves from, so we can prove we're using the venv's packages and not the system's. |

### Decision: `--system-site-packages` or not?

Track 1 baselines do **not** use `--system-site-packages`. They
install torch fresh into the venv. Two competing arguments:

**Argument for `--system-site-packages`:** Reuses the pre-installed
torch 2.5.0+cu124 from the runtime image (~3 GB saved, ~30 s shaved
off setup time, no CUDA-arch mismatch risk because we use the
known-good system build).

**Argument against:** Defeats the isolation purpose of using a venv
in the first place. If we want isolation from system packages, we
need the venv to be a true isolated environment.

**Recommendation if building this:** use `--system-site-packages`
PLUS a torch constraint file (same pattern as v4). This keeps the
isolation for new packages we install (NeMo, ORT, our deps) while
still inheriting torch from system. Cost: a ~5-min setup vs ~3-min
for v4-novenv.

---

## What changes vs `nemotron_streaming/` (the original v1/v2 layout)

The existing failed submissions use a **submission-relative venv**:
`${DIR}/venv/lib/python3.*/site-packages`. Compared to that:

- Path moves from submission-relative (`${DIR}/venv/...`) to absolute
  (`/opt/nemotron_venv/...`).
- Path globbing in model.py (currently `_glob.glob("${DIR}/venv/lib/python3.*/site-packages")`)
  collapses to a literal string assignment.
- The venv survives across submissions (since `/opt/` persists in
  the runtime image's writable layer for the duration of the eval
  job — same as Track 1's pattern). This is actually a Codabench
  contract assumption: that `/opt/` is writable from setup.sh AND
  visible to ingestion's Python process. Track 1 baselines rely on
  this.

---

## Outcomes from v4 that map to this fallback

(Reproduced from `SUBMISSIONS.md`'s v4 entry for cross-reference.)

| v4 outcome | Recommended next step |
|---|---|
| v4 scores any CER | H1b confirmed. **Decide between** "ship v4-shape as production" (global install) **or** "build absolute-venv variant for cleaner production". The absolute-venv design here is the cleaner option. |
| v4 fails with `[SETUP_VERIFY] starting...` visible but `imports OK` missing | H1b confirmed in a specific way; the failing import is known. **The absolute-venv variant might fix it** if the failure was caused by a system-package conflict (e.g. system `numpy 2.x` clashing). Build this as the next probe. |
| v4 fails with same `Detected splits: []` and no `[SETUP_VERIFY]` line in ingestion log | H1b falsified — bug is not in our venv mechanism at all. **Do not build the absolute-venv variant**; the email to organizers is the next move (because the failure isn't in our addressable surface). |
| v4 scores garbage / unusable CER but does score | H1b confirmed; the absolute-venv variant is a candidate for the *real* production submission, with whatever model improvements (lazy load + arena off + tracemalloc-confirmed memory behaviour) we want from the v3 working copy. |

---

## Required artifacts BEFORE coding this variant (none of which exist yet)

If the user decides to build this after v4 lands, before any code is
written we need:

1. **Confirmation that `/opt/` is writable from setup.sh on the eval
   VM** (not just on the local container). Track 1 baselines rely on
   this implicitly. We have no direct evidence yet — but the fact
   that parakeet/canary_qwen are documented baselines is indirect
   evidence.
2. **Confirmation that the venv created in setup.sh persists into the
   ingestion process** (not just the smoke test process). Same
   argument: Track 1 baselines depend on this.
3. **A decision on `--system-site-packages`**: see "Decision" section
   above.
4. **The v4 outcome itself**, to know which of the four rows in
   "Outcomes" table above applies.

---

## What is explicitly NOT in this design

- No code. Don't write setup.sh or model.py for this until v4 lands.
- No `nemotron_streaming_absvenv/` directory yet.
- No SHA256, no zip, no submission entry.
- No production refactor of `nemotron_streaming/` (the original
  failed-on-Codabench kit). That kit stays untouched until v4
  resolves.

---

## When this doc should be deleted or archived

Once one of these is true:

- v4 succeeds and we ship a production version based on either v4's
  global-install pattern or this design's absolute-venv pattern.
  Archive this doc with a note pointing at the production kit.
- v4 fails with `Detected splits: []` and the email to organizers
  yields information that makes this design moot. Archive with a
  note pointing at whatever the new working approach turned out to be.
- 30 days pass with no v4 result and no organizer reply, at which
  point this design and v4 are both stale. Re-evaluate.
