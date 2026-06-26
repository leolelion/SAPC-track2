#!/usr/bin/env bash
# Deploy loop step 5b: fix "Wh" via SKIP-EMISSION-ON-FIRST-CHUNK (leading silence; negative TTFT confirms speech
# starts ~0.9s in). Validate on winner (Wh gone? CER->~5?) AND base (no content loss?). Re-measure latency.
set -uo pipefail
OUT=/workspace/finetune/nemo_ft
LOG="$OUT/deploy_fix2.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
LD=/workspace/sapc-nemotron/track2_starting_kit/local_decode.py
DIAG100=/workspace/SAPC2/manifest/Dev_100.csv
STREAM=/workspace/SAPC2/manifest/Dev_streaming.csv
step(){ echo; echo "############### $* ###############"; }

build () {  # $1=tag $2=export_dir
  SUB="$OUT/deploy_$1_fix2"
  rm -rf "$SUB"; cp -r /workspace/finetune/nemo_submission "$SUB"
  rm -f "$SUB"/weights/encoder_model.onnx* "$SUB"/weights/decoder_model.onnx* 2>/dev/null
  cp "$2"/* "$SUB"/weights/ 2>/dev/null
  mv "$SUB"/weights/encoder-model.onnx "$SUB"/weights/encoder_model.onnx
  mv "$SUB"/weights/decoder_joint-model.onnx "$SUB"/weights/decoder_model.onnx
  sed -i 's/^CHUNK_NEW = 56/CHUNK_NEW = 16/' "$SUB"/model.py
  python3 - "$SUB/model.py" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
old="        for f_idx in range(n_enc):"
new=("        if self._step_num == 0 and not is_drain:\n"
     "            return   # skip emission on warmup first chunk (leading silence)\n"
     "        for f_idx in range(n_enc):")
assert old in s; open(p,"w").write(s.replace(old,new)); print("skip-first-chunk patched OK")
PY
  echo "$SUB"
}
cer_check () {  # $1=tag $2=csv
  python3 - "$DIAG100" "$2" <<'PY'
import csv,sys
sys.path.insert(0,"/workspace/sapc-nemotron/utils")
from normalizer.text_normalizer_hf import EnglishTextNormalizer
nm=EnglishTextNormalizer(); R={r['id']:r for r in csv.DictReader(open(sys.argv[1]))}
def cer(h,r):
    h,r=list(h),list(r)
    if not r:return 0.0 if not h else 1.0
    dp=list(range(len(r)+1))
    for i in range(1,len(h)+1):
        p=dp[0];dp[0]=i
        for j in range(1,len(r)+1):
            c=dp[j];dp[j]=min(dp[j]+1,dp[j-1]+1,p+(h[i-1]!=r[j-1]));p=c
    return min(1.0,dp[len(r)]/len(r))
rows=list(csv.DictReader(open(sys.argv[2])));tot=0;n=0
for r in rows:
    m=R.get(r['id']);
    if not m:continue
    h=nm.norm(r['raw_hypos']);c=min(cer(h,nm.norm(m.get('norm_text_with_disfluency',''))),cer(h,nm.norm(m.get('norm_text_without_disfluency',''))));tot+=c;n+=1
wh=sum(1 for r in rows if r['raw_hypos'].strip().startswith('Wh '))
print(f"  meanCER={100*tot/max(1,n):.1f}%  'Wh ' prefixes={wh}/{len(rows)}")
for r in rows[:5]: print("   ",repr(r['raw_hypos'][:55]))
PY
}

step "A. WINNER fix2 -> CER + Wh check + LATENCY"
W=$(build full70_1 "$OUT/export_full70_1")
( cd "$W" && SAPC2_THREADS=4 python3 "$LD" --submission-dir "$W" --manifest-csv "$DIAG100" \
   --streaming-manifest-csv "$STREAM" --data-root /workspace/SAPC2 --out-csv /dev/shm/w2.csv --out-partial-json /dev/shm/w2.json ) 2>&1 | tail -5
cer_check full70_1 /dev/shm/w2.csv
echo "--- latency ---"; python3 /workspace/sapc-nemotron/utils/compute_latency.py --partial-json /dev/shm/w2.json --manifest-csv "$STREAM" 2>&1 | grep -E 'ttft|ttlt|p50|p90|median' | head -12

step "B. BASE fix2 -> CER (confirm no content loss from skip)"
B=$(build base70_1 "$OUT/export_base70_1")
( cd "$B" && SAPC2_THREADS=4 python3 "$LD" --submission-dir "$B" --manifest-csv "$DIAG100" \
   --streaming-manifest-csv "$DIAG100" --data-root /workspace/SAPC2 --out-csv /dev/shm/b2.csv --out-partial-json /dev/shm/_b2.json ) 2>&1 | tail -4
cer_check base70_1 /dev/shm/b2.csv
echo "DEPLOY_FIX2_DONE"
