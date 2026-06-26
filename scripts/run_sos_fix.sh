#!/usr/bin/env bash
# Deploy loop step 7: test the SOS-token-init bug (model.py primes RNN-T with _last_token=0 = embed(BPE#0);
# NeMo uses SOS=blank=1024 with blank_as_pad). Set _last_token=[[BLANK_ID]] and re-stream BOTH arms on Dev_100.
# DISCRIMINATING test the prior fixes never ran. If "Wh" vanishes on enc-only -> it was the SOS bug, salvage the
# better model.
set -uo pipefail
OUT=/workspace/finetune/nemo_ft
LOG="$OUT/sos_fix.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
LD=/workspace/sapc-nemotron/track2_starting_kit/local_decode.py
DIAG100=/workspace/SAPC2/manifest/Dev_100.csv
STREAM=/workspace/SAPC2/manifest/Dev_streaming.csv
step(){ echo; echo "############### $* ###############"; }

build_sosfix () {  # $1=tag $2=export_dir
  SUB="$OUT/sos_$1"
  rm -rf "$SUB"; cp -r /workspace/finetune/nemo_submission "$SUB"
  rm -f "$SUB"/weights/encoder_model.onnx* "$SUB"/weights/decoder_model.onnx* 2>/dev/null
  cp "$2"/* "$SUB"/weights/ 2>/dev/null
  mv "$SUB"/weights/encoder-model.onnx "$SUB"/weights/encoder_model.onnx
  mv "$SUB"/weights/decoder_joint-model.onnx "$SUB"/weights/decoder_model.onnx
  sed -i 's/^CHUNK_NEW = 56/CHUNK_NEW = 16/' "$SUB"/model.py
  # THE FIX: SOS token blank(1024), not 0
  before=$(grep -c 'np.array(\[\[0\]\], dtype=np.int32)' "$SUB"/model.py)
  sed -i 's/np.array(\[\[0\]\], dtype=np.int32)/np.array([[BLANK_ID]], dtype=np.int32)/g' "$SUB"/model.py
  after=$(grep -c 'np.array(\[\[BLANK_ID\]\], dtype=np.int32)' "$SUB"/model.py)
  echo "SOS-fix: replaced $before occurrence(s) of [[0]] -> [[BLANK_ID]] (now $after)" >&2
  echo "$SUB"
}
cer_check () {  # $1=csv
  python3 - "$DIAG100" "$1" <<'PY'
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
print(f"  meanCER={100*tot/max(1,n):.1f}%  'Wh ' prefixes={wh}/{len(rows)}")
for r in rows[:5]: print("   ",repr(r['raw_hypos'][:55]))
PY
}
decode () {  # $1=SUB $2=out
  ( cd "$1" && SAPC2_THREADS=4 python3 "$LD" --submission-dir "$1" --manifest-csv "$DIAG100" \
     --streaming-manifest-csv "$STREAM" --data-root /workspace/SAPC2 --out-csv "$2" --out-partial-json "${2%.csv}.json" ) 2>&1 | tail -4
}

step "A. ENCODER-ONLY + SOS-fix (does 'Wh' vanish? CER -> 23.6 territory?)"
S=$(build_sosfix enc "$OUT/export_full70_1"); decode "$S" /dev/shm/sos_enc.csv; cer_check /dev/shm/sos_enc.csv
echo "--- latency enc+sosfix ---"; python3 /workspace/sapc-nemotron/utils/compute_latency.py --partial-json /dev/shm/sos_enc.json --manifest-csv "$STREAM" 2>&1 | grep -E 'median|p90' | head -6

step "B. FULL-UNFROZEN + SOS-fix (CER/latency change?)"
S=$(build_sosfix uf "$OUT/export_unfrozen70_1"); decode "$S" /dev/shm/sos_uf.csv; cer_check /dev/shm/sos_uf.csv
echo "--- latency uf+sosfix ---"; python3 /workspace/sapc-nemotron/utils/compute_latency.py --partial-json /dev/shm/sos_uf.json --manifest-csv "$STREAM" 2>&1 | grep -E 'median|p90' | head -6
echo "SOS_FIX_DONE"
