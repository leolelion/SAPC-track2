# Reference submission diff — SAPC2 Track 2

**Companion to:** [scripts/audit/ingestion_contract.md](ingestion_contract.md)
(the negative-result ingestion-code audit).

**Goal:** find a *publicly accessible* Track 2 submission package that
is known to have produced a Codabench score, and diff its structure
against ours. This bounds what "definitely works" looks like,
independent of whether we can read Codabench's ingestion code.

---

## TL;DR — what we found

- **One** Track 2 submission package is publicly verifiable: upstream
  `xiuwenz2/SAPC-template:track2_starting_kit/streaming_zipformer.zip`
  (SHA256 `f7876a006d9ba9730c6c38236b1556b005b8abfb293cf450cb72ea6c7f99aa45`,
  7,331 bytes, 3 files).
- Per our experiments ledger
  ([track2_starting_kit/experiments/results.jsonl](../../track2_starting_kit/experiments/results.jsonl),
  entry `hist-zipformer-test1-official`),
  this baseline scored **CER 34.59 / WER 52.77 on Test1**. Note: that
  entry is `verified: false` (backfilled from project notes).
- **It does NOT use a venv.** It installs all dependencies into the
  base Python via global `pip install`.
- **No other Track 2 NeMo-based submission with a venv has ever been
  confirmed working on Codabench.** Our own `parakeet_ctc/` kit
  (which uses an identical venv-injection pattern to ours) is *not*
  in the ledger as a successful Test1 submission.
- The closest precedent for a venv-based NeMo submission is **Track 1**
  (parakeet/canary_qwen), which uses an **absolute path**
  (`/opt/parakeet_venv/`) — not a submission-relative venv like ours.

The cleanest single explanation that fits *all three* of our failures:
**Track 2 ingestion's `from model import Model` succeeds on simple
submissions but our `<DIR>/venv/lib/python3.*/site-packages` injection
in `model.py` does not actually deliver `onnxruntime` + `nemo` to the
parent Python under Track 2's ingestion environment, in a way that
local_decode.py does not exercise.** This remains a hypothesis — see
*Differences and what they could explain* below.

---

## 1. Inventory of public reference submissions

### 1a. Upstream `xiuwenz2/SAPC-template`

Searched the upstream repository at all commits on `upstream/main`:

```
git ls-tree -r upstream/main track2_starting_kit/
```

| File | Notes |
|---|---|
| `track2_starting_kit/README.md` | Spec doc; intent only, not code. |
| `track2_starting_kit/local_decode.py` | Local equivalent of Codabench ingestion. |
| `track2_starting_kit/streaming_zipformer/{model.py,setup.sh,config.yaml}` | The one official baseline source tree. |
| `track2_starting_kit/streaming_zipformer.zip` | **A pre-packaged submission zip.** |

No `baselines/`, no `reference_submission/`, no NeMo-based example,
no `requirements.txt` example, no Track 2 NeMo baseline at all.

The pre-packaged zip extracts to exactly the source tree above:

```
$ unzip -l streaming_zipformer.zip
    16114  02-17-2026 21:23   model.py
     2924  02-19-2026 05:18   setup.sh
     1712  02-17-2026 21:23   config.yaml
SHA256(streaming_zipformer.zip)
  = f7876a006d9ba9730c6c38236b1556b005b8abfb293cf450cb72ea6c7f99aa45
```

`diff -q` confirms the zip contents are byte-identical to the
directory copy at `track2_starting_kit/streaming_zipformer/{model.py,
setup.sh,config.yaml}` — so reading the kit directory is equivalent
to reading the shipped baseline.

### 1b. HuggingFace `Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17`

This is the **model checkpoint** repo that the baseline `setup.sh`
downloads from (lines 53–62 of
[track2_starting_kit/streaming_zipformer/setup.sh](../../track2_starting_kit/streaming_zipformer/setup.sh)).
It is maintained by Zengwei (icefall maintainer at K2-FSA) and
contains the LibriSpeech-trained checkpoint plus the BPE tokenizer.

There is no SAPC2-specific `model.py` or `setup.sh` in that HF repo —
the SAPC2 wrapper is constructed entirely on the SAPC2 side.

### 1c. Our own repo's track 2 kits (additional reference points)

| Kit | Origin | Uses venv? | Verified on Codabench? |
|---|---|---|---|
| `streaming_zipformer/` | **upstream** (the canonical baseline) | no | yes — CER 34.59 (backfilled) |
| `sherpa_zipformer/` | added by user (commit `1b86646`) | no | unknown — not in ledger as a Codabench result |
| `parakeet_ctc/` | added by user (commit `dd9f264`) | **yes (same pattern as ours)** | unknown — not in ledger as a Codabench result |
| `nemotron_streaming/` | added by user (commits `4204de1` → `2920073`) | **yes (same pattern as ours)** | FAILED ×3 |
| `streaming_pruned_stateless7/`, `wenet_u2pp/`, `moonshine/`, `gigaam/`, `emformer_rnnt/`, `qwen3_asr/`, `faster_whisper/` | added by user | various | unknown |

**Conclusion:** the ONE Track 2 submission shape we can confirm
works on Codabench is the no-venv `streaming_zipformer` shape.

### 1d. Track 1 NeMo baselines (analogous-shape lookup)

Track 1 ships TWO upstream NeMo baselines that use a venv:

- [track1_starting_kit/parakeet/setup.sh](../../track1_starting_kit/parakeet/setup.sh):8–9 creates venv at **`/opt/parakeet_venv`** (absolute path).
- [track1_starting_kit/parakeet/model.py](../../track1_starting_kit/parakeet/model.py):10 injects exactly that absolute path: `_venv_site = "/opt/parakeet_venv/lib/python3.11/site-packages"`.
- [track1_starting_kit/canary_qwen/setup.sh](../../track1_starting_kit/canary_qwen/setup.sh):5–6 + corresponding model.py use the same pattern at `/opt/canary_venv`.

These are upstream-maintained kits, so the venv-at-`/opt` pattern is
explicitly supported in Track 1 ingestion. There is no analogous
upstream Track 2 NeMo baseline, so we have no evidence about Track 2
ingestion's venv handling.

---

## 2. Diff: official baseline vs our submission

### Files in each zip

| | `streaming_zipformer.zip` (upstream) | `nemotron_streaming.zip` (ours, v2-oom-fix `0c4cb384…`) |
|---|---|---|
| `model.py` | ✓ (16,114 B) | ✓ (20,643 B) |
| `setup.sh` | ✓ (2,924 B) | ✓ (5,683 B) |
| `config.yaml` | ✓ (1,712 B) | ✓ (775 B) |
| `README.md` | — | ✓ (2,044 B) |
| total | 20,750 B / 3 files | 29,045 B / 4 files |

Both stay under any reasonable submission-size limit. The extra
README in our zip is unlikely to be load-bearing but is not standard
across baselines.

### `setup.sh` — line-level comparison

| Aspect | Upstream `streaming_zipformer/setup.sh` | Ours `nemotron_streaming/setup.sh` |
|---|---|---|
| Shebang + strict mode | `#!/bin/bash` + `set -e` (line 12) | `#!/usr/bin/env bash` + `set -euo pipefail` (line 11) |
| Venv? | **none — installs globally** (line 35: `pip install -q numpy …`) | **creates `${DIR}/venv` with `--system-site-packages`** (lines 14, 57) |
| pip install scope | base Python | venv only |
| Heavy deps | k2 + kaldifeat wheels (lines 26-27, 38-41), icefall git clone + `pip install -e .` (lines 44-45) | `nemo_toolkit[asr]>=2.5.0,<2.6` + `onnxruntime` + `onnx` + `huggingface_hub` (lines 70-77) |
| Torch constraint | none | yes — extracts system torch version, writes constraint file (lines 59-65) |
| Weights | HF download via `huggingface_hub` python (lines 53-62) | HF download via `hf_hub_download` python (lines 98-121) |
| Smoke test? | **none** | yes, but wrapped in `set +e` so non-fatal (lines 130-151) |
| Exit | implicit (success after weights download) | `echo "=== setup.sh complete ===" + exit 0` |

**Three observable shape differences:**

1. **venv vs global**. The upstream baseline does not create a venv.
2. **strict-mode level**. Upstream uses `set -e`; ours uses
   `set -euo pipefail`. Ours treats any unset variable or any failed
   pipe stage as fatal. In a setup.sh body that uses pipes (`free -h
   | head -5` etc. don't appear in ours, but the pattern is risky).
3. **smoke test presence**. Ours runs a smoke test that, even when
   non-fatal, *prints diagnostic output*. Upstream doesn't.

### `model.py` — interface-level comparison

| Aspect | Upstream `streaming_zipformer/model.py` | Ours `nemotron_streaming/model.py` (v2-oom-fix shipped) |
|---|---|---|
| `sys.path` manipulation | lines 63-64: `sys.path.insert(0, str(_ICEFALL))` and `_ZIPFORMER` — **subdirs of the submission dir** | lines 42-50: `sys.path.insert(0, _venv_candidates[0])` — **venv site-packages of the submission dir** |
| Module-level heavy imports | yes (`torch`, `kaldifeat`, `icefall.*`) | yes (`torch`, `onnxruntime`, `omegaconf`) |
| Heavy load in `__init__` | yes (`get_model`, `load_checkpoint` on first construction) | yes (encoder + decoder ORT sessions + NeMo preprocessor) |
| `_partial_callback` initial | line 257: `self._partial_callback = None` (would crash on call until set) | line 214: `lambda _t: None` (safe default) |
| `accept_chunk` returns | partial transcript (line 312) | partial transcript (line 268) |
| `input_finished` returns | final transcript (line 321) | final transcript (line 283) |
| File writes from Model | **none** (verified by grep — no `open(.*csv|.*json`, w` writes) | **none** (verified by grep — no predict.csv or partial.json writes) |
| Conf load | line 80: `config = OmegaConf.load(_DIR / "config.yaml")` at module scope | inside `__init__`: `OmegaConf.load(cfg_path)` (line 191 of shipped) |

**Three observable shape differences:**

1. **Where `sys.path` is augmented.** Upstream points at local
   sub-directories of the submission dir (`icefall/`, `zipformer/`)
   that exist after `git clone` — every interpreter that imports
   `from model import Model` will see this injection. Ours points at
   a venv site-packages dir that lives under the submission dir but
   requires the *running* Python's ABI to match the venv's Python
   ABI (only Python 3.11 vs 3.11 — we confirmed match locally, but
   we cannot confirm on the eval VM).
2. **`_partial_callback` initialization.** Defensive `lambda _t: None`
   is safer; upstream's `None` would crash on call if ingestion
   forgot `set_partial_callback`. (This bounds an ingestion contract
   point: *it MUST call `set_partial_callback` before the first
   `accept_chunk`*, otherwise the baseline crashes. So we can deduce
   ingestion DOES call it. Useful contract fact.)
3. **Config load timing.** Upstream loads config at module import
   time. Ours loads inside `__init__`. Under multiprocess, this
   means each worker repeats the load — small cost, not load-bearing.

### `config.yaml` — irrelevant for the contract

Both are kit-internal and not consumed by ingestion. Skipping diff.

---

## 3. Things we can rule in / out from the diff

| Hypothesis | Status from diff |
|---|---|
| Model writes predict.csv itself (would conflict with ingestion's write) | **Ruled out for both.** Neither Model writes any file. |
| Wrong return type from `input_finished` | **Ruled out.** Both return `str`. Ours `_last_emitted` can be `""` but that's still a string. |
| Wrong column name in predict.csv | **Not affected by Model.** The column choice is made by ingestion. |
| set_partial_callback contract (called or not) | **Confirmed called** — upstream baseline depends on it. |
| chunk dtype | **Not directly observable** from the diff. Upstream baseline accepts `np.ndarray` via `torch.from_numpy(audio_chunk)` (line 309). Ours `astype(np.float32, copy=False)` if not already (line 264). Both tolerate float32; upstream may tolerate more dtypes via Torch's coercion. |

| Hypothesis | Strengthened by diff |
|---|---|
| **H1b (venv injection is the failure point under Track 2 ingestion)** | **Strengthened.** The ONLY confirmed-working Track 2 baseline uses no venv. The venv-injection pattern in `model.py` has *no upstream Track 2 precedent*. Track 1's analogous pattern uses an absolute path (`/opt/parakeet_venv`), not a submission-relative venv. |
| **H1a (setup.sh non-zero exit on HF download failure)** | Same as before — upstream's setup also depends on HF reachable; if it works for them it presumably works for us. Slightly weakened. |
| **H1c (image tag drift)** | Unchanged — diff doesn't speak to this. |
| **H3 (per-worker subprocess silent failure)** | Slightly weakened — upstream's torch-based heavy `__init__` would trigger the same per-worker reload, yet works. Not exonerated for our ORT-specific case but harder to argue. |

---

## 4. Required-file check

Things we might be missing that the upstream baseline has:

| Required by upstream baseline | Present in our zip? |
|---|---|
| `model.py` with `Model` class | ✓ |
| `setup.sh` | ✓ |
| `config.yaml` | ✓ — present though our Model loads it lazily |

Things upstream does NOT have, that we DO have:

| In our zip but not in upstream | Risk |
|---|---|
| `README.md` | low — extra text file, ingestion shouldn't care |
| venv mechanism | **possibly load-bearing** (see Section 3) |

Things mentioned in spec README but not present in either:

| Mentioned in README | Present in any baseline? |
|---|---|
| `requirements.txt` (auto-installed after setup.sh per spec line 105) | not used by upstream; not used by us — both rely on setup.sh exclusively |

No required files appear to be missing from our zip relative to upstream.

---

## 5. Updated hypothesis ranking

Ranked by how well each fits the failure pattern (3 byte-identical
failures across substantively different model implementations), with
the new diff evidence factored in.

### H1b-refined — venv injection silently fails under Track 2 ingestion

**Symptom fit:** Strong. If `from model import Model` succeeds
(because model.py's top-level code runs before any heavy import) but
then `import onnxruntime as ort` (line 57) fails because the venv
injection at lines 42-50 didn't actually deliver onnxruntime to the
running Python's import system, **the result is an `ImportError`
raised inside the import of model.py**. Ingestion's `from model
import Model` then fails. No `[CPU_DIAGNOSTIC]`, no `[MEM_DIAGNOSTIC]`,
no `[nemotron_streaming]` lines — exactly what we see.

Why this fails on Codabench but not in our local container:
- Locally we tested with our own `local_decode.py` running as a
  single process from a known cwd. The `os.path.dirname(__file__)`
  in `model.py:43` resolves cleanly.
- On Codabench, if ingestion changes cwd, or runs the import from
  a subprocess that doesn't inherit our sys.path mods, the
  injection target path is computed from `__file__` which is the
  zip-extracted path. Glob `${DIR}/venv/lib/python3.*/site-packages`
  matches only if the venv was created at exactly that location.

Sub-checks that would *confirm* this:
- ingestion log line `ImportError: No module named 'onnxruntime'`
- or `ImportError` from inside model.py at line 57 specifically.

Code references:
- our [model.py:42-50](../../track2_starting_kit/nemotron_streaming/model.py#L42-L50) — the injection.
- our [model.py:57](../../track2_starting_kit/nemotron_streaming/model.py#L57) — first import that needs the venv.
- upstream [model.py:63-64](../../track2_starting_kit/streaming_zipformer/model.py#L63-L64) — comparison: injects local submission sub-dir paths, not venv site-packages.
- our [setup.sh:14, 57](../../track2_starting_kit/nemotron_streaming/setup.sh#L14) — creates venv at `${DIR}/venv`.

### H1a — setup.sh exits non-zero before Model is ever imported

**Symptom fit:** Strong. If `set -euo pipefail` (our setup.sh line 11)
kills setup.sh during pip install or HF download, ingestion never
reaches `from model import Model`.

Sub-checks:
- ingestion log line `Failed to download` or `pip install … exit
  status N` or similar.
- HF rate-limit / network refusal.

Code references:
- our [setup.sh:11](../../track2_starting_kit/nemotron_streaming/setup.sh#L11) — strict mode.
- our [setup.sh:97-121](../../track2_starting_kit/nemotron_streaming/setup.sh#L97-L121) — HF download python heredoc, NOT wrapped in `set +e`.
- upstream's setup.sh [line 12](../../track2_starting_kit/streaming_zipformer/setup.sh#L12) uses only `set -e` — same risk profile but with simpler dep tree.

### H1c — image tag drift

**Symptom fit:** Moderate. Our local `docker images` shows
`xiuwenz2/sapc2-runtime:latest` was pulled 2026-02-18; competition is
ongoing into 2026-08-31. The image may have been republished. The
single-tag pattern (`:latest`) makes this invisible to us.

Sub-check:
- Compare image digest on disk vs current on Docker Hub via
  `docker manifest inspect xiuwenz2/sapc2-runtime:latest`.

No code reference — this is environmental.

### H3 — per-worker subprocess silent failure (de-prioritised)

The diff slightly weakens H3. Upstream's baseline also instantiates
heavy torch/k2 state per `Model()` and works. If our submission's
multiprocess issue were a generic per-worker problem (not specific to
ORT's fork-unsafety), upstream would hit it too.

H3b (ORT-specific) is still possible but less load-bearing than
H1b. Our v3 lazy-load (uncommitted) would coincidentally mitigate
H3b but the priority is H1b.

---

## 6. What this changes about next steps

The diff makes one concrete suggestion for an *empirical probe* that
costs less than the hello-world zip and would directly test H1b:

**Build a "minimal NeMo Track 2" submission that copies the
streaming_zipformer global-install pattern, with our Nemotron
weights and a stripped-down model.py.** Specifically:

- setup.sh installs `onnxruntime` + `nemo_toolkit[asr]` globally
  (no venv). Mirror upstream baseline's pattern exactly.
- model.py removes lines 42-50 entirely — no venv injection.
- model.py keeps the rest of the Nemotron decoder logic.

If THIS submission still fails identically, H1b is falsified and the
problem is elsewhere. If it succeeds, H1b is confirmed and the venv
mechanism is the bug — which we can then fix in the main submission.

That probe is **not** small (it requires real refactoring of model.py
and a fresh pip-install pass in the eval VM). It is, however, a single
slot that produces a binary answer about the strongest remaining
hypothesis.

The hello-world zip remains the cheapest probe IF the question is
"does the contract work at all" rather than "is venv the bug." For
H1b specifically, hello-world doesn't help because hello-world's
setup.sh installs nothing and doesn't exercise the venv mechanism.

**No action proposed yet.** Per the audit's hard rule, this is
investigation only. The user decides which (if any) probe is worth
spending a slot on, and when to send the organizer email (still not
sent as of this audit).

---

## 7. Audit log

| Step | Action |
|---|---|
| 1a | Searched `upstream/main` for `baselines/`, `reference_submission/`, etc. — none. Found `streaming_zipformer.zip` + source dir. |
| 1b | HF model repo is checkpoint only, no SAPC2 wrapper. |
| 1c | Inventoried local kits to identify which have been confirmed Codabench-scored — only `streaming_zipformer` (CER 34.59). |
| 1d | Cross-checked Track 1 venv patterns for analogous precedent — uses `/opt/<name>_venv` absolute, not submission-relative. |
| 2 | Line-level diff of upstream baseline vs our v2-oom-fix submission across setup.sh, model.py, config.yaml. |
| 3 | Ruled in/out hypotheses based on the diff. |
| 4 | Required-file check — nothing missing from our zip. |
| 5 | Updated hypothesis ranking: **H1b (venv injection failure under Track 2 ingestion) is now the strongest**. |
| 6 | Identified a single high-value empirical probe (Nemotron-without-venv submission). No action taken. |

No model changes. No submission preparation. No zip rebuilds.
