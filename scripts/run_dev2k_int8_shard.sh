#!/usr/bin/env bash
# Faithful but PARALLEL eval of the shipped Nemotron encoder-SOS int8 on the etiology-balanced
# 2k Dev ruler. The bundled local_decode.py batch pass is serial (~6s/utt -> ~3h for 2k), so we
# shard the manifest round-robin across N independent local_decode procs (identical per-utt code
# path => identical CER), concat the hyps, then score ONCE with the official evaluate.sh + bootstrap
# + per-etiology. No model changes, no training.
set -uo pipefail

N=${N:-40}
THREADS=${THREADS:-1}
OUT=/workspace/finetune/nemo_ft
ART="$OUT/artifacts"
LOG="$OUT/dev2k_int8_shard.log"; mkdir -p "$ART"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
export PATH="/workspace/SCTK/bin:$PATH"
# Clamp all thread pools to 1 so N single-thread procs do not oversubscribe the 192 cores
# (20 procs x ~16 threads = 321 threads thrashed). N procs x THREADS threads instead.
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS

PROJ=/workspace/sapc-nemotron
LD="$PROJ/track2_starting_kit/local_decode.py"
DATA=/workspace/SAPC2
SUB="$OUT/submission_encoder_sos_int8"
BOOT=/workspace/streaming_cer_bootstrap.py
MAN=/workspace/finetune/eval/dev_eval_2k.csv
STREAM_SMOKE=/dev/shm/Dev_streaming_1.csv
SHARD_DIR=/dev/shm/dev2k_shards
HYP=/dev/shm/int8_dev2k.csv

step(){ echo; echo "############### $* ###############"; }

step "Preflight"
for f in "$LD" "$SUB/model.py" "$MAN" "$BOOT" "$PROJ/steps/eval/prepare_ref_trn.sh" "$PROJ/steps/eval/evaluate.sh"; do
  test -e "$f" || { echo "MISSING $f"; exit 1; }
done
grep -q 'np.array(\[\[BLANK_ID\]\], dtype=np.int32)' "$SUB/model.py" || { echo "BAD_SOS"; exit 1; }

# one-row streaming smoke (interval 0 -> latency pass is a no-op)
python3 - "$DATA/manifest/Dev_streaming.csv" "$STREAM_SMOKE" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerow(rows[0])
PY

step "Shard manifest round-robin into $N shards"
rm -rf "$SHARD_DIR"; mkdir -p "$SHARD_DIR"
python3 - "$MAN" "$SHARD_DIR" "$N" <<'PY'
import csv, sys
src, d, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = list(csv.DictReader(open(src, newline="")))
hdr = rows[0].keys()
buckets = [[] for _ in range(n)]
for i, r in enumerate(rows):
    buckets[i % n].append(r)
for i, b in enumerate(buckets):
    with open(f"{d}/shard_{i:02d}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(b)
print(f"sharded {len(rows)} rows into {n} shards (~{len(rows)//n}/shard)")
PY

step "Launch $N parallel faithful local_decode procs"
t0=$(date +%s)
pids=()
for i in $(seq -w 0 $((N-1))); do
  sm="$SHARD_DIR/shard_${i}.csv"
  so="$SHARD_DIR/hyp_${i}.csv"
  sj="$SHARD_DIR/part_${i}.json"
  ( cd "$SUB" && SAPC2_THREADS=$THREADS python3 "$LD" --submission-dir "$SUB" \
      --manifest-csv "$sm" --streaming-manifest-csv "$STREAM_SMOKE" \
      --streaming-interval 0 --data-root "$DATA" \
      --out-csv "$so" --out-partial-json "$sj" ) >"$SHARD_DIR/log_${i}.txt" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} procs: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || { echo "shard pid $p FAILED"; fail=1; }; done
t1=$(date +%s); echo "all shards done in $((t1-t0))s (fail=$fail)"

step "Concat shard hyps -> $HYP"
python3 - "$SHARD_DIR" "$HYP" "$N" "$MAN" <<'PY'
import csv, sys, glob
d, out, n, man = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rows = []
for fp in sorted(glob.glob(f"{d}/hyp_*.csv")):
    rows += list(csv.DictReader(open(fp, newline="")))
n_expected = sum(1 for _ in csv.DictReader(open(man, newline="")))
ids = {r["id"] for r in rows}
print(f"concat rows={len(rows)} unique_ids={len(ids)} expected={n_expected}")
fn = rows[0].keys()
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fn); w.writeheader(); w.writerows(rows)
assert len(rows) == n_expected, f"ROW COUNT MISMATCH {len(rows)} != {n_expected}"
print("concat OK")
PY

step "OFFICIAL evaluate.sh (Test1-comparable headline)"
REF_DIR="$ART/official_refs_dev2k"; mkdir -p "$REF_DIR"
bash "$PROJ/steps/eval/prepare_ref_trn.sh" "$PROJ" \
  --split dev2k_int8 --csv "$MAN" \
  --ref1-col norm_text_with_disfluency --ref2-col norm_text_without_disfluency \
  --out-dir "$REF_DIR"
bash "$PROJ/steps/eval/evaluate.sh" "$PROJ" "$DATA" \
  --split dev2k_int8 --hyp-csv "$HYP" --hyp-col raw_hypos \
  --manifest-csv "$MAN" --ref-dir "$REF_DIR" \
  --out-json "$ART/dev2k_int8_official_metrics.json"
echo "OFFICIAL_JSON:"; cat "$ART/dev2k_int8_official_metrics.json"; echo

step "Bootstrap CER overall (cross-check)"
python3 "$BOOT" --manifest-csv "$MAN" --hyp-csv "$HYP" \
  --utils-dir "$PROJ/utils" --out-json "$ART/dev2k_int8_overall_bootstrap.json"
cat "$ART/dev2k_int8_overall_bootstrap.json"; echo

step "Per-etiology CER (bootstrap on each subset)"
for E in "ALS" "Cerebral Palsy" "Down Syndrome" "Parkinson's Disease" "Stroke"; do
  SAFE=$(echo "$E" | tr " '" "__")
  SM="/dev/shm/dev2k_${SAFE}.csv"
  python3 - "$MAN" "$SM" "$E" <<'PY'
import csv, sys
src, out, et = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(src, newline="")))
sel = [r for r in rows if r["etiology"] == et]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(sel)
PY
  python3 "$BOOT" --manifest-csv "$SM" --hyp-csv "$HYP" \
    --utils-dir "$PROJ/utils" --out-json "$ART/dev2k_int8_${SAFE}_bootstrap.json" 2>/dev/null
  echo -n "  $E -> "; python3 - "$ART/dev2k_int8_${SAFE}_bootstrap.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print({k: d[k] for k in d if "cer" in k.lower()})
PY
done

echo "DEV2K_INT8_SHARD_DONE"
