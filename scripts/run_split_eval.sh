#!/usr/bin/env bash
# Stage-0 like-for-like: banked beam-4 zipformer on the SAME 2000-utt val set the parakeet
# arms are gated on, through the REAL harness + OFFICIAL two-ref scorer.
# CPU-only; sharded so it does not contend with the GPU training job.
set -uo pipefail
N=${N:-16}; THREADS=${THREADS:-2}
SUB=${SUB:-/workspace/finetune/zf_a1_beam4_sub}
TAG=${TAG:-zf_beam4}
DATA=/workspace/SAPC2; MAN=${MAN:-$DATA/manifest/Dev_clean2k.csv}
PROJ=/workspace/SAPC-template; LD=$PROJ/track2_starting_kit/local_decode.py
OUT=/workspace/parakeet_ft/eval_$TAG; SHARDS=$OUT/shards
rm -rf "$OUT"; mkdir -p "$SHARDS"
source /workspace/nemoenv/bin/activate
export PATH="/workspace/SCTK/bin:$PATH"
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

# one-row streaming manifest: local_decode needs the streaming pass to exist, but this run
# only scores accuracy (pass 1). Keeping it to 1 row makes pass 2 a no-op cost.
STREAM1=$OUT/stream1.csv
python3 - "$DATA/manifest/Dev_streaming.csv" "$STREAM1" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1],newline="")))
w=csv.DictWriter(open(sys.argv[2],"w",newline=""),fieldnames=list(rows[0].keys()));w.writeheader();w.writerow(rows[0])
PY

python3 - "$MAN" "$SHARDS" "$N" <<'PY'
import csv,sys
src,d,n=sys.argv[1],sys.argv[2],int(sys.argv[3])
rows=list(csv.DictReader(open(src,newline=""))); hdr=list(rows[0].keys())
for i in range(n):
    w=csv.DictWriter(open(f"{d}/shard_{i:02d}.csv","w",newline=""),fieldnames=hdr);w.writeheader();w.writerows(rows[i::n])
print(f"sharded {len(rows)} into {n}")
PY

t0=$(date +%s); pids=()
for i in $(seq 0 $((N-1))); do
  i=$(printf "%02d" "$i")   # shards are written %02d; `seq -w` widths by max, which broke N<10
  ( cd "$SUB" && SAPC2_THREADS=$THREADS python3 "$LD" --submission-dir "$SUB" \
      --manifest-csv "$SHARDS/shard_${i}.csv" --streaming-manifest-csv "$STREAM1" \
      --streaming-interval 0 --data-root "$DATA" \
      --out-csv "$SHARDS/hyp_${i}.csv" --out-partial-json "$SHARDS/p_${i}.json" ) \
      > "$SHARDS/log_${i}.txt" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[$TAG] decoded in $(( $(date +%s)-t0 ))s fail=$fail"

python3 - "$SHARDS" "$OUT/val2k.predict.csv" "$MAN" <<'PY'
import csv,sys,glob
d,out,man=sys.argv[1],sys.argv[2],sys.argv[3]
rows=[]
for fp in sorted(glob.glob(f"{d}/hyp_*.csv")): rows+=list(csv.DictReader(open(fp,newline="")))
exp=sum(1 for _ in csv.DictReader(open(man,newline="")))
w=csv.DictWriter(open(out,"w",newline=""),fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
print(f"concat={len(rows)} expected={exp}"); assert len(rows)==exp
PY

cd $PROJ
./evaluate.sh --split ${SPLIT:-Dev_clean2k} --hyp-csv "$OUT/val2k.predict.csv" --start_stage 1 --stop_stage 2 \
  > "$OUT/official.log" 2>&1 && echo "[official ok]" || echo "[official FAIL]"
grep -aiE "cer|wer" "$OUT/official.log" | tail -10
python3 - "$OUT/val2k.predict.csv" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1],newline="")))
col=[c for c in rows[0] if c!="id"][0]
empt=sum(1 for r in rows if not (r[col] or "").strip())
print(f"[{len(rows)} utts] empty hypotheses = {empt} ({100.0*empt/len(rows):.2f}%)")
PY
echo "VAL2K_${TAG}_DONE"
