# Step 6 — Config / manifest files comparison

## What config-type files each zip contains

| File | baseline | v7 |
|---|---|---|
| `config.yaml` | yes (1712 B, **CRLF**) | yes (775 B, LF) |
| `requirements.txt` | **no** | **no** |
| `metadata` / `metadata.json` | **no** | **no** |
| `submission.json` | **no** | **no** |
| `README.md` (inside zip) | no | yes (2877 B) |

## The manifest hypothesis — TESTED AND REJECTED

The task flagged: *"does baseline have a metadata file that declares submission
type / interface version? If baseline has one and ours doesn't, that's almost
certainly the culprit."*

**Result: the baseline has NO manifest of any kind.** It ships exactly three files
— `model.py`, `setup.sh`, `config.yaml` — and no `metadata`, `submission.json`,
`metadata.json`, or `requirements.txt`. The spec (`track2_starting_kit/README.md`,
"How to Submit") confirms the required package is just `model.py` + (recommended)
`setup.sh` + "Other supporting files". **No manifest is required, and the working
reference proves it by not having one.**

→ The "missing manifest causes pre-ingestion failure" hypothesis is dead: a
submission with no manifest is currently passing on Codabench. v7 not having one
cannot be the gate.

## config.yaml differences (both valid, schema is submission-specific)

`config.yaml` is read by the submission's own `model.py` (via OmegaConf), not by
Codabench. The two files have entirely different schemas because they configure
different models:

- baseline: `audio/weights/encoder/decoder/features/decoding` (zipformer knobs).
- v7: `model.num_threads` + `weights.{dir,encoder,decoder,tokens}` (ONNX bundle).

This is expected and correct — each model.py reads its own config. The only
format curiosity is that the **baseline's config.yaml uses CRLF** and works fine,
so line endings here are not load-bearing.

## Verdict (Step 6)
No manifest/metadata file is required or present in the working baseline, so its
absence in v7 is not a failure cause. config.yaml differences are schema-by-design
and parsed only by the submission itself. No pre-ingestion gate found here.
