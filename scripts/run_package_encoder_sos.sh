#!/usr/bin/env bash
# Build and smoke-test the encoder-only + SOS-fixed Nemotron offline submission zip.
set -euo pipefail

OUT=/workspace/finetune/nemo_ft
LOG="$OUT/package_encoder_sos.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate

SRC=/workspace/finetune/nemo_submission
EXPORT="$OUT/export_full70_1"
ART="$OUT/artifacts"
SUB="$OUT/submission_encoder_sos"
ZIP="$ART/nemotron_encoder_sos_submission.zip"
SHA="$ZIP.sha256"
LD=/workspace/sapc-nemotron/track2_starting_kit/local_decode.py
DATA=/workspace/SAPC2
MAN_DIR="$DATA/manifest"
SMOKE_N=${SMOKE_N:-20}
FORCE_REBUILD=${FORCE_REBUILD:-0}
SMOKE_MAN="/dev/shm/Dev_pkg_smoke_${SMOKE_N}.csv"
STREAM_SMOKE=/dev/shm/Dev_streaming_1.csv
EXTRACT=/dev/shm/nemotron_encoder_sos_extract

step(){ echo; echo "############### $* ###############"; }

make_manifest_sample () {
  python3 - "$MAN_DIR/Dev.csv" "$SMOKE_MAN" "$SMOKE_N" <<'PY'
import csv, sys
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = list(csv.DictReader(open(src, newline="")))[:n]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"smoke manifest: {out} rows={len(rows)}")
PY
  python3 - "$MAN_DIR/Dev_streaming.csv" "$STREAM_SMOKE" <<'PY'
import csv, sys
src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src, newline="")))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerow(rows[0])
print(f"stream smoke manifest: {out}")
PY
}

check_tree () {
  fail=0
  require_file(){ if ! test -f "$1"; then echo "MISSING_FILE $1"; fail=1; fi; }
  require_dir(){ if ! test -d "$1"; then echo "MISSING_DIR $1"; fail=1; fi; }
  require_file "$SUB/model.py"
  require_file "$SUB/localmel.py"
  require_file "$SUB/config.yaml"
  require_file "$SUB/setup.sh"
  require_file "$SUB/weights/encoder_model.onnx"
  require_file "$SUB/weights/decoder_model.onnx"
  require_file "$SUB/weights/tokens.txt"
  if [ "$(find "$SUB/weights" -type f | wc -l)" -lt 10 ]; then
    echo "TOO_FEW_WEIGHT_FILES"
    fail=1
  fi
  require_dir "$SUB/wheels"
  if ! grep -q 'np.array(\[\[BLANK_ID\]\], dtype=np.int32)' "$SUB/model.py"; then
    echo "MISSING_SOS_BLANK_PATCH"
    fail=1
  fi
  if grep -q 'np.array(\[\[0\]\], dtype=np.int32)' "$SUB/model.py"; then
    echo "FOUND_OLD_SOS_ZERO"
    fail=1
  fi
  if grep -Riq 'nemo_toolkit\|huggingface\|from nemo\|import nemo' "$SUB/setup.sh" "$SUB/model.py"; then
    echo "FOUND_FORBIDDEN_NEMO_OR_HF_REFERENCE"
    fail=1
  fi
  if [ "$fail" -ne 0 ]; then
    echo "--- weights listing ---"
    ls -lh "$SUB/weights" || true
    return 1
  fi
}

step "Build submission dir"
rm -rf "$SUB"; cp -r "$SRC" "$SUB"
rm -f "$SUB"/weights/encoder_model.onnx* "$SUB"/weights/decoder_model.onnx* 2>/dev/null
cp "$EXPORT"/* "$SUB"/weights/
mv "$SUB"/weights/encoder-model.onnx "$SUB"/weights/encoder_model.onnx
mv "$SUB"/weights/decoder_joint-model.onnx "$SUB"/weights/decoder_model.onnx
sed -i 's/^CHUNK_NEW = 56/CHUNK_NEW = 16/' "$SUB"/model.py
before=$(grep -c 'np.array(\[\[0\]\], dtype=np.int32)' "$SUB"/model.py)
sed -i 's/np.array(\[\[0\]\], dtype=np.int32)/np.array([[BLANK_ID]], dtype=np.int32)/g' "$SUB"/model.py
after=$(grep -c 'np.array(\[\[BLANK_ID\]\], dtype=np.int32)' "$SUB"/model.py)
echo "SOS patch replacements: before=$before blank_occurrences=$after"
check_tree

step "Build zip"
mkdir -p "$ART"
if [ "$FORCE_REBUILD" = "1" ] || [ ! -f "$ZIP" ]; then
  rm -f "$ZIP" "$SHA"
  python3 - "$SUB" "$ZIP" <<'PY'
import os, sys, zipfile
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])
tmp = out.with_suffix(out.suffix + ".tmp")
if tmp.exists():
    tmp.unlink()
with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        arc = path.relative_to(src).as_posix()
        zf.write(path, arc)
        if arc == "setup.sh":
            zf.getinfo(arc).external_attr = (0o755 & 0xFFFF) << 16
tmp.replace(out)
print(f"wrote {out}")
PY
else
  echo "reusing existing zip: $ZIP"
fi
sha256sum "$ZIP" | tee "$SHA"
ls -lh "$ZIP" "$SHA"
python3 - "$ZIP" <<'PY'
import sys, zipfile
zf = zipfile.ZipFile(sys.argv[1])
names = set(zf.namelist())
required = {
    "model.py", "localmel.py", "config.yaml", "setup.sh",
    "weights/encoder_model.onnx",
    "weights/decoder_model.onnx",
    "weights/tokens.txt",
}
missing = sorted(required - names)
weight_files = [n for n in names if n.startswith("weights/") and not n.endswith("/")]
bad_nested = [n for n in names if n.startswith("submission_encoder_sos/")]
print(f"zip entries={len(names)} weight_files={len(weight_files)} missing={missing} nested_root={bool(bad_nested)}")
assert not missing
assert len(weight_files) >= 10
assert not bad_nested
PY

step "Extracted package smoke"
rm -rf "$EXTRACT"; mkdir -p "$EXTRACT"
python3 - "$ZIP" "$EXTRACT" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    zf.extractall(sys.argv[2])
print(f"extracted {sys.argv[1]} -> {sys.argv[2]}")
PY
make_manifest_sample
( cd "$EXTRACT" && SAPC2_THREADS=4 python3 "$LD" --submission-dir "$EXTRACT" \
    --manifest-csv "$SMOKE_MAN" --streaming-manifest-csv "$STREAM_SMOKE" \
    --streaming-interval 0 --data-root "$DATA" \
    --out-csv /dev/shm/pkg_smoke.csv --out-partial-json /dev/shm/pkg_smoke.json ) 2>&1 | tail -12
python3 /workspace/streaming_cer_bootstrap.py --manifest-csv "$SMOKE_MAN" \
  --hyp-csv /dev/shm/pkg_smoke.csv --utils-dir /workspace/sapc-nemotron/utils \
  --bootstrap-iters 200 --out-json "$ART/pkg_smoke_cer.json"

echo "PACKAGE_ENCODER_SOS_DONE zip=$ZIP sha=$SHA"
