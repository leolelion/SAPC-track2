# Step 3 — File presence symmetric diff

baseline = `streaming_zipformer.zip` (3 files)
v7       = `nemotron_streaming_v7_minimal.zip` (4 files)

## Present in BOTH
| File | baseline size | v7 size |
|---|---|---|
| `model.py` | 16114 | 25132 |
| `setup.sh` | 2924 | 2662 |
| `config.yaml` | 1712 | 775 |

## Present in baseline but NOT in ours (missing from v7)
- **none**

## Present in ours but NOT in baseline (extra in v7)
- `README.md` (2877 B) — author notes describing the v7 submission. Benign.

## Spec compliance (per `track2_starting_kit/README.md`)
- **Required: `model.py`** with the 5-method `Model` interface → present in both. ✅
- **Recommended: `setup.sh`** → present in both. ✅
- **Optional: `requirements.txt`** → absent in both. (Not required; setup.sh does
  the installs.) ✅ neither is penalized.
- **No manifest / metadata / submission.json is required by the spec.** The spec's
  zip example is literally `model.py` + `setup.sh` + "Other supporting files".
- Files are expected **flat at the zip root** (spec example shows no wrapping dir);
  both zips comply.

## Verdict (Step 3)
v7 is a **strict superset** of the baseline's file set (same three files, plus a
benign README.md). Nothing required by the spec or present in the baseline is
missing from v7. There is no missing-file explanation for the failure.
