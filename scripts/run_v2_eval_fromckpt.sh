#!/usr/bin/env bash
# Gate 1b+c, robust variant: rebuild the v2 model by AVERAGING the 4 intact epoch checkpoints,
# export to ONNX DIRECTLY (skips the truncation-prone .nemo save), package, faithfully eval
# (Dev_diag + Dev-500 + PD probe), print the pre-registered verdict.
set -uo pipefail
OUT=/workspace/finetune/nemo_ft; ART="/tmp/v2_artifacts"
mkdir -p "$ART"; LOG="/tmp/v2_eval_fromckpt.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
export PATH="/workspace/SCTK/bin:$PATH"

BASE="$OUT/nemotron-speech-streaming-en-0.6b.nemo"
CKDIR="$OUT/v2_guardrail_armA"
PROJ=/workspace/sapc-nemotron; LD="$PROJ/track2_starting_kit/local_decode.py"
DATA=/workspace/SAPC2; BOOT=/workspace/streaming_cer_bootstrap.py; EMP=/workspace/empties_analysis.py
EXPDIR="/tmp/export_v2_armA"; SUB="/tmp/submission_v2_armA"; SRC=/workspace/finetune/nemo_submission
STREAM_SMOKE=/dev/shm/Dev_streaming_1.csv
step(){ echo; echo "############### $* ###############"; }

step "Average 4 ckpts -> export ONNX at [70,1] (direct, no .nemo)"
rm -rf "$EXPDIR"; mkdir -p "$EXPDIR"
python3 - "$BASE" "$CKDIR" "$EXPDIR" <<'PY'
import sys, os, glob, torch, nemo.collections.asr as na
base, ckdir, od = sys.argv[1], sys.argv[2], sys.argv[3]
m = na.models.EncDecRNNTBPEModel.restore_from(base, map_location="cpu")
ckpts = sorted(glob.glob(os.path.join(ckdir, "ft-epoch=*.ckpt")))
print("ckpts:", [os.path.basename(c) for c in ckpts])
avg=None; n=0
for c in ckpts:
    sd = torch.load(c, map_location="cpu", weights_only=False).get("state_dict", {})
    if not sd: continue
    n += 1
    if avg is None: avg = {k: v.float().clone() for k, v in sd.items()}
    else:
        for k in avg:
            if k in sd: avg[k] += sd[k].float()
for k in avg: avg[k] /= n
miss = m.load_state_dict(avg, strict=False)
print(f"averaged {n} ckpts; load missing={len(getattr(miss,'missing_keys',[]))} unexpected={len(getattr(miss,'unexpected_keys',[]))}")
m.encoder.set_default_att_context_size([70, 1])
try: m.encoder.setup_streaming_params()
except Exception as e: print("setup_streaming_params:", e)
m.set_export_config({'cache_support': 'True'})
m.export(os.path.join(od, "model.onnx"))
print("exported onnx:", sorted([f for f in os.listdir(od) if f.endswith('.onnx')]))
PY
test -f "$EXPDIR/encoder-model.onnx" && test -f "$EXPDIR/decoder_joint-model.onnx" || { echo "EXPORT_MISSING_ONNX"; ls "$EXPDIR" | head; exit 1; }

step "Package submission_v2_armA"
rm -rf "$SUB"; cp -r "$SRC" "$SUB"
rm -f "$SUB"/weights/encoder_model.onnx* "$SUB"/weights/decoder_model.onnx* 2>/dev/null
cp "$EXPDIR"/* "$SUB"/weights/
mv "$SUB"/weights/encoder-model.onnx "$SUB"/weights/encoder_model.onnx
mv "$SUB"/weights/decoder_joint-model.onnx "$SUB"/weights/decoder_model.onnx
sed -i 's/^CHUNK_NEW = 56/CHUNK_NEW = 16/' "$SUB"/model.py
sed -i 's/np.array(\[\[0\]\], dtype=np.int32)/np.array([[BLANK_ID]], dtype=np.int32)/g' "$SUB"/model.py
grep -q 'np.array(\[\[BLANK_ID\]\], dtype=np.int32)' "$SUB"/model.py && echo "OK SOS" || { echo "FAIL SOS"; exit 1; }
grep -q '^CHUNK_NEW = 16' "$SUB"/model.py && echo "OK CHUNK16" || echo "WARN CHUNK"

step "Build PD-probe CSV + stream smoke"
python3 - "$DATA/manifest/Dev_diag.csv" /dev/shm/pd_probe.csv <<'PY'
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1],newline="")) if r["etiology"]=="Parkinson's Disease"]
csv.DictWriter(open(sys.argv[2],"w",newline=""),fieldnames=rows[0].keys()).writeheader()
w=csv.DictWriter(open(sys.argv[2],"w",newline=""),fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print("pd_probe rows:", len(rows))
PY
python3 - "$DATA/manifest/Dev_streaming.csv" "$STREAM_SMOKE" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1],newline="")))
w=csv.DictWriter(open(sys.argv[2],"w",newline=""),fieldnames=r[0].keys());w.writeheader();w.writerow(r[0])
PY

decode_eval () {  # $1=tag $2=manifest $3=N
  tag="$1"; man="$2"; N="${3:-16}"; sd="/dev/shm/v2e_${tag}"; rm -rf "$sd"; mkdir -p "$sd"
  python3 - "$man" "$sd" "$N" <<'PY'
import csv,sys
src,d,n=sys.argv[1],sys.argv[2],int(sys.argv[3])
rows=list(csv.DictReader(open(src,newline=""))); h=rows[0].keys()
for i in range(n):
    w=csv.DictWriter(open(f"{d}/s_{i:02d}.csv","w",newline=""),fieldnames=h);w.writeheader();w.writerows(rows[i::n])
PY
  pids=()
  for i in $(seq -w 0 $((N-1))); do
    ( cd "$SUB" && SAPC2_THREADS=2 OMP_NUM_THREADS=2 python3 "$LD" --submission-dir "$SUB" \
        --manifest-csv "$sd/s_${i}.csv" --streaming-manifest-csv "$STREAM_SMOKE" --streaming-interval 0 \
        --data-root "$DATA" --out-csv "$sd/h_${i}.csv" --out-partial-json "$sd/p_${i}.json" ) >"$sd/log_${i}.txt" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || echo "shard $p fail"; done
  python3 - "$sd" "/dev/shm/v2e_${tag}.csv" <<'PY'
import csv,sys,glob
d,out=sys.argv[1],sys.argv[2]; rows=[]
for fp in sorted(glob.glob(f"{d}/h_*.csv")): rows+=list(csv.DictReader(open(fp,newline="")))
w=csv.DictWriter(open(out,"w",newline=""),fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(f"{out} rows={len(rows)}")
PY
  python3 "$BOOT" --manifest-csv "$man" --hyp-csv "/dev/shm/v2e_${tag}.csv" --utils-dir "$PROJ/utils" --out-json "$ART/v2_${tag}_boot.json"
}

step "Eval Dev_diag (baseline v1 23.58% / empties 7.3%)"
decode_eval devdiag "$DATA/manifest/Dev_diag.csv" 16
python3 "$EMP" --manifest-csv "$DATA/manifest/Dev_diag.csv" --hyp-csv /dev/shm/v2e_devdiag.csv --utils-dir "$PROJ/utils" --out-json "$ART/v2_devdiag_empties.json"
step "Eval Dev-500 (baseline ~9.91%)"; decode_eval dev500 "$OUT/Dev_rep500_seed23.csv" 16
step "Eval PD-probe (forgetting; baseline 4.49%)"; decode_eval pd /dev/shm/pd_probe.csv 8

step "PRE-REGISTERED VERDICT"
python3 /workspace/v2_gate_verdict.py \
  --devdiag-json "$ART/v2_devdiag_boot.json" --devdiag-empties-json "$ART/v2_devdiag_empties.json" \
  --dev500-json "$ART/v2_dev500_boot.json" --pd-json "$ART/v2_pd_boot.json" || true
echo "V2_EVAL_FROMCKPT_DONE"
