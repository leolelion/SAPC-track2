#!/usr/bin/env bash
# Net effect of CONDITIONAL gain normalization (boost only utts with RMS<target) on the FULL
# Dev_diag severe set vs the 23.58% baseline. Surgical: loud utts untouched, only quiet ones boosted.
set -uo pipefail
N=${N:-16}; THREADS=${THREADS:-2}
MODE=${MODE:-rms_cond}
OUT=/workspace/finetune/nemo_ft; ART="$OUT/artifacts"
LOG="$OUT/devdiag_norm_full.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
export PATH="/workspace/SCTK/bin:$PATH"
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
PROJ=/workspace/sapc-nemotron; LD="$PROJ/track2_starting_kit/local_decode.py"; DATA=/workspace/SAPC2
ORIG="$OUT/submission_encoder_sos_int8"; NORM="$OUT/submission_encoder_sos_int8_norm"
BOOT=/workspace/streaming_cer_bootstrap.py; EMP=/workspace/empties_analysis.py
MAN="$DATA/manifest/Dev_diag.csv"; STREAM_SMOKE=/dev/shm/Dev_streaming_1.csv
SHARD_DIR=/dev/shm/ddnorm_shards; HYP=/dev/shm/int8_devdiag_${MODE}.csv
step(){ echo; echo "############### $* ###############"; }

step "(Re)build norm submission with rms_cond support"
rm -rf "$NORM"; cp -r "$ORIG" "$NORM"
python3 /workspace/patch_audionorm.py "$NORM/model.py"
python3 -m py_compile "$NORM/model.py" && echo "py_compile OK"
grep -q "rms_cond" "$NORM/model.py" && echo "rms_cond present"

test -f "$STREAM_SMOKE" || python3 - "$DATA/manifest/Dev_streaming.csv" "$STREAM_SMOKE" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1],newline="")))
w=csv.DictWriter(open(sys.argv[2],"w",newline=""),fieldnames=r[0].keys());w.writeheader();w.writerow(r[0])
PY

step "Shard Dev_diag into $N"
rm -rf "$SHARD_DIR"; mkdir -p "$SHARD_DIR"
python3 - "$MAN" "$SHARD_DIR" "$N" <<'PY'
import csv,sys
src,d,n=sys.argv[1],sys.argv[2],int(sys.argv[3])
rows=list(csv.DictReader(open(src,newline=""))); hdr=rows[0].keys()
for i in range(n):
    w=csv.DictWriter(open(f"{d}/shard_{i:02d}.csv","w",newline=""),fieldnames=hdr);w.writeheader();w.writerows(rows[i::n])
print(f"sharded {len(rows)} into {n}")
PY

step "Decode FULL Dev_diag with NORM=$MODE (parallel)"
t0=$(date +%s); pids=()
for i in $(seq -w 0 $((N-1))); do
  ( cd "$NORM" && SAPC2_THREADS=$THREADS SAPC2_AUDIO_NORM="$MODE" python3 "$LD" --submission-dir "$NORM" \
      --manifest-csv "$SHARD_DIR/shard_${i}.csv" --streaming-manifest-csv "$STREAM_SMOKE" \
      --streaming-interval 0 --data-root "$DATA" \
      --out-csv "$SHARD_DIR/hyp_${i}.csv" --out-partial-json "$SHARD_DIR/p_${i}.json" ) \
      >"$SHARD_DIR/log_${i}.txt" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "decoded in $(( $(date +%s)-t0 ))s fail=$fail"

python3 - "$SHARD_DIR" "$HYP" "$MAN" <<'PY'
import csv,sys,glob
d,out,man=sys.argv[1],sys.argv[2],sys.argv[3]
rows=[]
for fp in sorted(glob.glob(f"{d}/hyp_*.csv")): rows+=list(csv.DictReader(open(fp,newline="")))
exp=sum(1 for _ in csv.DictReader(open(man,newline="")))
csv.DictWriter(open(out,"w",newline=""),fieldnames=rows[0].keys()).writeheader() or None
w=csv.DictWriter(open(out,"w",newline=""),fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(f"concat={len(rows)} expected={exp}"); assert len(rows)==exp
PY

step "Score NORM=$MODE full Dev_diag (baseline was 23.58%, 31 empty)"
python3 "$BOOT" --manifest-csv "$MAN" --hyp-csv "$HYP" --utils-dir "$PROJ/utils" --out-json "$ART/devdiag_${MODE}_bootstrap.json"
python3 "$EMP" --manifest-csv "$MAN" --hyp-csv "$HYP" --utils-dir "$PROJ/utils" --out-json "$ART/devdiag_${MODE}_empties.json"
echo "DEVDIAG_NORM_FULL_DONE"
