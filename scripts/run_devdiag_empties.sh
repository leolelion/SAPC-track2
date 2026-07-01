#!/usr/bin/env bash
# Empties audit on the severe-enriched Dev_diag set with the SHIPPED Nemotron enc-SOS int8.
# Re-decodes Dev_diag (faithful local_decode, sharded for speed), then bootstrap CER +
# empties decomposition (how much of the severe-set CER is "model emitted nothing"). Also
# writes the list of empty utterance ids so a no-empty fallback can be tested on exactly those.
# CPU-only, no training, no model change.
set -uo pipefail

N=${N:-16}
THREADS=${THREADS:-2}
OUT=/workspace/finetune/nemo_ft
ART="$OUT/artifacts"
LOG="$OUT/devdiag_empties.log"; mkdir -p "$ART"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
export PATH="/workspace/SCTK/bin:$PATH"
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS

PROJ=/workspace/sapc-nemotron
LD="$PROJ/track2_starting_kit/local_decode.py"
DATA=/workspace/SAPC2
SUB="$OUT/submission_encoder_sos_int8"
BOOT=/workspace/streaming_cer_bootstrap.py
EMP=/workspace/empties_analysis.py
MAN="$DATA/manifest/Dev_diag.csv"
STREAM_SMOKE=/dev/shm/Dev_streaming_1.csv
SHARD_DIR=/dev/shm/devdiag_shards
HYP=/dev/shm/int8_devdiag.csv

step(){ echo; echo "############### $* ###############"; }

step "Preflight"
for f in "$LD" "$SUB/model.py" "$MAN" "$BOOT" "$EMP" "$PROJ/utils"; do test -e "$f" || { echo "MISSING $f"; exit 1; }; done
grep -q 'np.array(\[\[BLANK_ID\]\], dtype=np.int32)' "$SUB/model.py" || { echo "BAD_SOS"; exit 1; }
echo "Dev_diag rows: $(tail -n +2 "$MAN" | wc -l)"

python3 - "$DATA/manifest/Dev_streaming.csv" "$STREAM_SMOKE" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerow(rows[0])
PY

step "Shard Dev_diag round-robin into $N"
rm -rf "$SHARD_DIR"; mkdir -p "$SHARD_DIR"
python3 - "$MAN" "$SHARD_DIR" "$N" <<'PY'
import csv, sys
src, d, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = list(csv.DictReader(open(src, newline="")))
hdr = rows[0].keys()
for i in range(n):
    with open(f"{d}/shard_{i:02d}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
        w.writerows(rows[i::n])
print(f"sharded {len(rows)} rows into {n}")
PY

step "Decode $N shards in parallel (faithful, THREADS=$THREADS)"
t0=$(date +%s); pids=()
for i in $(seq -w 0 $((N-1))); do
  ( cd "$SUB" && SAPC2_THREADS=$THREADS python3 "$LD" --submission-dir "$SUB" \
      --manifest-csv "$SHARD_DIR/shard_${i}.csv" --streaming-manifest-csv "$STREAM_SMOKE" \
      --streaming-interval 0 --data-root "$DATA" \
      --out-csv "$SHARD_DIR/hyp_${i}.csv" --out-partial-json "$SHARD_DIR/part_${i}.json" ) \
      >"$SHARD_DIR/log_${i}.txt" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "shards done in $(( $(date +%s) - t0 ))s (fail=$fail)"

step "Concat -> $HYP"
python3 - "$SHARD_DIR" "$HYP" "$MAN" <<'PY'
import csv, sys, glob
d, out, man = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for fp in sorted(glob.glob(f"{d}/hyp_*.csv")): rows += list(csv.DictReader(open(fp, newline="")))
exp = sum(1 for _ in csv.DictReader(open(man, newline="")))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"concat rows={len(rows)} expected={exp}")
assert len(rows) == exp, "ROW MISMATCH"
PY

step "Bootstrap CER (overall + per-etiology + empties)"
python3 "$BOOT" --manifest-csv "$MAN" --hyp-csv "$HYP" --utils-dir "$PROJ/utils" \
  --out-json "$ART/devdiag_int8_bootstrap.json"

step "Empties decomposition"
python3 "$EMP" --manifest-csv "$MAN" --hyp-csv "$HYP" --utils-dir "$PROJ/utils" \
  --out-json "$ART/devdiag_int8_empties.json" --empty-ids-out "$ART/devdiag_int8_empty_ids.txt"

echo "DEVDIAG_EMPTIES_DONE"
