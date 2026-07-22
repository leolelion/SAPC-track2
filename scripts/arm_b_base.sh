#!/usr/bin/env bash
# ARM B (direct clincher): export the SAME base .nemo -> ONNX via OUR export path -> package into the
# nemo_submission model.py (canonical deploy patches: CHUNK_NEW=16, SOS=BLANK_ID) -> run the REAL
# local_decode.py streaming accuracy pass on Dev_diag -> score with the IDENTICAL two-ref scorer used
# for A (43.4%) / C (43.8%). If B >> ~44% -> OUR ONNX export is where the historical 28.19 collapse lived.
set -uo pipefail
OUT=/workspace/finetune/nemo_base_ac
LOG="$OUT/arm_b.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
DATA=/workspace/SAPC2
DIAG=$DATA/manifest/Dev_diag.csv
PROJ=/workspace/sapc-nemotron
LD=$PROJ/track2_starting_kit/local_decode.py
SRC=/workspace/finetune/nemo_submission
EXPDIR=$OUT/export_base_b
SUB=$OUT/submission_base_b

SNAP=$(find /workspace/.cache -path '*nemotron-speech-streaming-en-0.6b*' -name '*.nemo' 2>/dev/null | head -1)
BASE=$(readlink -f "$SNAP" 2>/dev/null || echo "$SNAP")
[ -f "$BASE" ] || { echo "FATAL: base .nemo missing"; exit 2; }
[ -f "$LD" ] || { echo "FATAL: local_decode.py missing"; exit 2; }
echo "BASE=$BASE"

echo "### export base -> ONNX [70,1] cache_support ###"
rm -rf "$EXPDIR"; mkdir -p "$EXPDIR"
python3 - "$BASE" "$EXPDIR" <<'PY'
import sys, os, nemo.collections.asr as na
p, od = sys.argv[1], sys.argv[2]
m = na.models.EncDecRNNTBPEModel.restore_from(p, map_location="cpu")
m.encoder.set_default_att_context_size([70, 1])
try: m.encoder.setup_streaming_params()
except Exception as e: print("setup_streaming_params:", e)
m.set_export_config({'cache_support': 'True'})
m.export(os.path.join(od, "model.onnx"))
print("exported:", sorted(os.listdir(od)))
PY
test -f "$EXPDIR/encoder-model.onnx" && test -f "$EXPDIR/decoder_joint-model.onnx" || { echo "EXPORT_MISSING_ONNX"; ls -la "$EXPDIR"; exit 3; }

echo "### package submission (swap ONNX; CHUNK_NEW=16; SOS=BLANK_ID) ###"
rm -rf "$SUB"; cp -r "$SRC" "$SUB"
rm -f "$SUB"/weights/encoder_model.onnx* "$SUB"/weights/decoder_model.onnx*
cp "$EXPDIR"/* "$SUB"/weights/
mv "$SUB"/weights/encoder-model.onnx "$SUB"/weights/encoder_model.onnx
mv "$SUB"/weights/decoder_joint-model.onnx "$SUB"/weights/decoder_model.onnx
sed -i 's/^CHUNK_NEW = 56/CHUNK_NEW = 16/' "$SUB"/model.py
sed -i 's/np.array(\[\[0\]\], dtype=np.int32)/np.array([[BLANK_ID]], dtype=np.int32)/g' "$SUB"/model.py
grep -qE '^CHUNK_NEW = 16' "$SUB"/model.py && echo "OK CHUNK_NEW=16" || echo "WARN CHUNK_NEW not patched"
grep -q 'np.array(\[\[BLANK_ID\]\], dtype=np.int32)' "$SUB"/model.py && echo "OK SOS=BLANK_ID" || echo "WARN SOS not patched"

echo "### 1-row stream smoke + 16-shard accuracy decode on Dev_diag ###"
SMOKE=/dev/shm/stream_smoke.csv
python3 - "$DATA/manifest/Dev_streaming.csv" "$SMOKE" <<'PY'
import csv, sys
r = list(csv.DictReader(open(sys.argv[1], newline="")))
w = csv.DictWriter(open(sys.argv[2], "w", newline=""), fieldnames=list(r[0].keys())); w.writeheader(); w.writerow(r[0])
PY
N=16; SD=/dev/shm/armb; rm -rf "$SD"; mkdir -p "$SD"
python3 - "$DIAG" "$SD" "$N" <<'PY'
import csv, sys
src, d, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = list(csv.DictReader(open(src, newline=""))); h = list(rows[0].keys())
for i in range(n):
    w = csv.DictWriter(open(f"{d}/s_{i:02d}.csv", "w", newline=""), fieldnames=h); w.writeheader(); w.writerows(rows[i::n])
print("sharded", len(rows), "rows into", n)
PY
pids=()
for i in $(seq -w 0 $((N-1))); do
  ( cd "$SUB" && SAPC2_THREADS=2 OMP_NUM_THREADS=2 python3 "$LD" --submission-dir "$SUB" \
      --manifest-csv "$SD/s_${i}.csv" --streaming-manifest-csv "$SMOKE" --streaming-interval 0 \
      --data-root "$DATA" --out-csv "$SD/h_${i}.csv" --out-partial-json "$SD/p_${i}.json" ) >"$SD/log_${i}.txt" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "shards done (failures=$fail)"
ls "$SD"/h_*.csv >/dev/null 2>&1 || { echo "FATAL: no shard outputs — sample shard log:"; tail -20 "$SD"/log_00.txt; exit 4; }

python3 - "$SD" "$OUT/arm_b_hyp.csv" <<'PY'
import csv, sys, glob
d, out = sys.argv[1], sys.argv[2]; rows = []
for fp in sorted(glob.glob(f"{d}/h_*.csv")): rows += list(csv.DictReader(open(fp, newline="")))
if not rows: print("FATAL: 0 merged rows"); sys.exit(4)
w = csv.DictWriter(open(out, "w", newline=""), fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("merged rows", len(rows), "->", out)
PY

echo "### score ARM B (hand-rolled two-ref, same scorer as A/C) keyed by id ###"
python3 - "$DIAG" "$OUT/arm_b_hyp.csv" <<'PY'
import csv, os, sys, ast
sys.path.insert(0, "/workspace/sapc-nemotron/utils")
from normalizer.text_normalizer_hf import EnglishTextNormalizer
nm = EnglishTextNormalizer()
diag = {r["id"]: r for r in csv.DictReader(open(sys.argv[1]))}
def clean(s):
    s = (s or "").strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list): return " ".join(map(str, v))
        except Exception: pass
    return s
def cer(h, r):
    h, r = list(h), list(r)
    if not r: return 0.0 if not h else 1.0
    dp = list(range(len(r) + 1))
    for i in range(1, len(h) + 1):
        p = dp[0]; dp[0] = i
        for j in range(1, len(r) + 1):
            c = dp[j]; dp[j] = min(dp[j] + 1, dp[j - 1] + 1, p + (h[i - 1] != r[j - 1])); p = c
    return min(1.0, dp[len(r)] / len(r))
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[2])))
hdr = rows[0].keys() if rows else []
hyp_col = "raw_hypos" if "raw_hypos" in hdr else ([c for c in hdr if c != "id"][0] if hdr else "raw_hypos")
agg = defaultdict(lambda: [0, 0.0, 0]); tot = [0, 0.0, 0]; un = 0
for r in rows:
    m = diag.get(r.get("id", ""))
    if not m: un += 1; continue
    pred = clean(r.get(hyp_col, ""))
    h = nm.norm(pred)
    c = min(cer(h, nm.norm(m.get("norm_text_with_disfluency", ""))), cer(h, nm.norm(m.get("norm_text_without_disfluency", ""))))
    e = 1 if not pred.strip() else 0
    for b in (agg[m["etiology"]], tot): b[0] += 1; b[1] += c; b[2] += e
print(f"\n=== ARM B our-ONNX-export streaming (local_decode.py, hyp_col={hyp_col})  matched={tot[0]} unmatched={un} ===")
print(f"  ALL CER={100*tot[1]/max(1,tot[0]):.1f}% empty={tot[2]} ({100*tot[2]/max(1,tot[0]):.0f}%)")
for k in sorted(agg):
    b = agg[k]; print(f"  {k:20s} n={b[0]:4d} CER={100*b[1]/max(1,b[0]):.1f}% empty={b[2]}")
print("\n[compare] ARM A transcribe 43.4% | ARM C NeMo-streaming 43.8%")
PY
echo "ARM_B_DONE"
