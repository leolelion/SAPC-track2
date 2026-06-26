#!/usr/bin/env bash
# Deployment loop step 5: fix the stream-start "Wh" artifact (drop drop_extra_pre_encoded=2 warmup frames on
# the first chunk) and measure latency. CER on Dev_100, TTFT/TTLT on Dev_streaming.
set -uo pipefail
OUT=/workspace/finetune/nemo_ft
LOG="$OUT/deploy_fix.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
LD=/workspace/sapc-nemotron/track2_starting_kit/local_decode.py
SUB="$OUT/deploy_full70_1_fix"
DIAG100=/workspace/SAPC2/manifest/Dev_100.csv
STREAM=/workspace/SAPC2/manifest/Dev_streaming.csv
step(){ echo; echo "############### $* ###############"; }

step "0. build winner deploy dir + CHUNK_NEW=16 + WARMUP-DROP patch"
rm -rf "$SUB"; cp -r /workspace/finetune/nemo_submission "$SUB"
rm -f "$SUB"/weights/encoder_model.onnx* "$SUB"/weights/decoder_model.onnx* 2>/dev/null
cp "$OUT"/export_full70_1/* "$SUB"/weights/ 2>/dev/null
mv "$SUB"/weights/encoder-model.onnx "$SUB"/weights/encoder_model.onnx
mv "$SUB"/weights/decoder_joint-model.onnx "$SUB"/weights/decoder_model.onnx
sed -i 's/^CHUNK_NEW = 56/CHUNK_NEW = 16/' "$SUB"/model.py
python3 - "$SUB/model.py" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
old="        for f_idx in range(n_enc):"
new="        _sf = 2 if self._step_num == 0 else 0   # drop_extra_pre_encoded warmup frames on first chunk\n        for f_idx in range(_sf, n_enc):"
assert old in s, "decode-loop line not found!"
open(p,"w").write(s.replace(old,new)); print("warmup-drop patched OK")
PY

step "1. local_decode (CER on Dev_100, latency partials on Dev_streaming)"
( cd "$SUB" && SAPC2_THREADS=4 python3 "$LD" --submission-dir "$SUB" --manifest-csv "$DIAG100" \
    --streaming-manifest-csv "$STREAM" --data-root /workspace/SAPC2 \
    --out-csv /dev/shm/fix.csv --out-partial-json /dev/shm/fix.json ) 2>&1 | tail -8

step "2. CER + 'Wh' check on Dev_100"
python3 - "$DIAG100" /dev/shm/fix.csv <<'PY'
import csv,sys
sys.path.insert(0,"/workspace/sapc-nemotron/utils")
from normalizer.text_normalizer_hf import EnglishTextNormalizer
nm=EnglishTextNormalizer()
R={r['id']:r for r in csv.DictReader(open(sys.argv[1]))}
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
    m=R.get(r['id'])
    if not m:continue
    h=nm.norm(r['raw_hypos']);c=min(cer(h,nm.norm(m.get('norm_text_with_disfluency',''))),cer(h,nm.norm(m.get('norm_text_without_disfluency',''))));tot+=c;n+=1
print(f"  N={n} meanCER={100*tot/max(1,n):.1f}%")
wh=sum(1 for r in rows if r['raw_hypos'].strip().startswith('Wh ') )
print(f"  hyps starting with spurious 'Wh ': {wh}/{len(rows)}")
for r in rows[:6]: print("   ",repr(r['raw_hypos'][:55]))
PY

step "3. LATENCY (TTFT/TTLT) via compute_latency.py on Dev_streaming"
echo "--- compute_latency argparse ---"; grep -nE "add_argument" /workspace/sapc-nemotron/utils/compute_latency.py | head -20
python3 /workspace/sapc-nemotron/utils/compute_latency.py --partial-json /dev/shm/fix.json --manifest-csv "$STREAM" 2>&1 | tail -25 || \
python3 /workspace/sapc-nemotron/utils/compute_latency.py --partial_json /dev/shm/fix.json --streaming-manifest-csv "$STREAM" 2>&1 | tail -25 || echo "LATENCY_CALL_FAIL (see argparse above)"
echo "DEPLOY_FIX_DONE"
