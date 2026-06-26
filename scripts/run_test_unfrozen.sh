#!/usr/bin/env bash
# Deploy loop step 6: the "Wh" artifact is from the finetuned ENCODER (encoder-only => frozen joint can't
# compensate). Test the FULL-UNFROZEN model (joint adapted to its encoder) -> does it stream CLEANLY?
# If yes, full-unfrozen is the better DEPLOYMENT model (the agent's "decide on the real harness" point).
set -uo pipefail
OUT=/workspace/finetune/nemo_ft
LOG="$OUT/test_unfrozen.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
LD=/workspace/sapc-nemotron/track2_starting_kit/local_decode.py
DIAG100=/workspace/SAPC2/manifest/Dev_100.csv
STREAM=/workspace/SAPC2/manifest/Dev_streaming.csv
ED="$OUT/export_unfrozen70_1"
step(){ echo; echo "############### $* ###############"; }

step "1. export full-unfrozen at [70,1]"
python3 - <<PY 2>&1 | tail -4
import os, nemo.collections.asr as na
m = na.models.EncDecRNNTBPEModel.restore_from("$OUT/full_unfrozen/ft_smoke_full.nemo", map_location="cpu")
m.encoder.set_default_att_context_size([70,1]); m.set_export_config({'cache_support':'True'})
os.makedirs("$ED", exist_ok=True); m.export(os.path.join("$ED","model.onnx"))
print("exported:", sorted([f for f in os.listdir("$ED") if f.endswith('.onnx')]))
PY

step "2. build deploy dir + stream Dev_100 (Wh? CER?) + latency"
SUB="$OUT/deploy_unfrozen"
rm -rf "$SUB"; cp -r /workspace/finetune/nemo_submission "$SUB"
rm -f "$SUB"/weights/encoder_model.onnx* "$SUB"/weights/decoder_model.onnx* 2>/dev/null
cp "$ED"/* "$SUB"/weights/ 2>/dev/null
mv "$SUB"/weights/encoder-model.onnx "$SUB"/weights/encoder_model.onnx
mv "$SUB"/weights/decoder_joint-model.onnx "$SUB"/weights/decoder_model.onnx
sed -i 's/^CHUNK_NEW = 56/CHUNK_NEW = 16/' "$SUB"/model.py
( cd "$SUB" && SAPC2_THREADS=4 python3 "$LD" --submission-dir "$SUB" --manifest-csv "$DIAG100" \
   --streaming-manifest-csv "$STREAM" --data-root /workspace/SAPC2 --out-csv /dev/shm/uf.csv --out-partial-json /dev/shm/uf.json ) 2>&1 | tail -5
python3 - "$DIAG100" /dev/shm/uf.csv <<'PY'
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
    m=R.get(r['id'])
    if not m:continue
    h=nm.norm(r['raw_hypos']);c=min(cer(h,nm.norm(m.get('norm_text_with_disfluency',''))),cer(h,nm.norm(m.get('norm_text_without_disfluency',''))));tot+=c;n+=1
wh=sum(1 for r in rows if r['raw_hypos'].strip().startswith('Wh '))
print(f"  UNFROZEN streaming: meanCER={100*tot/max(1,n):.1f}%  'Wh ' prefixes={wh}/{len(rows)}")
for r in rows[:6]: print("   ",repr(r['raw_hypos'][:55]))
PY
echo "--- latency ---"; python3 /workspace/sapc-nemotron/utils/compute_latency.py --partial-json /dev/shm/uf.json --manifest-csv "$STREAM" 2>&1 | grep -E 'median|p90' | head -6
echo "TEST_UNFROZEN_DONE"
