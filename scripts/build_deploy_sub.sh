#!/usr/bin/env bash
# Deployment loop step 4: build [70,1] submission dirs from the existing (working) Nemotron model.py +
# NeMo's [70,1] ONNX export (I/O matches; only CHUNK_NEW 56->16 differs), and run the REAL local_decode.py
# -> streaming CER + TTFT/TTLT latency. Validate on ZERO-SHOT base first, then the winner.
set -uo pipefail
OUT=/workspace/finetune/nemo_ft
LOG="$OUT/deploy_local_decode.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
LD=/workspace/sapc-nemotron/track2_starting_kit/local_decode.py
SRC=/workspace/finetune/nemo_submission
MAN=/workspace/SAPC2/manifest/Dev_100.csv          # small subset to validate harness/constants fast
step(){ echo; echo "############### $* ###############"; }

build_and_decode () {  # $1=tag  $2=export_dir
  SUB="$OUT/deploy_$1"
  rm -rf "$SUB"; cp -r "$SRC" "$SUB"
  rm -f "$SUB"/weights/encoder_model.onnx* "$SUB"/weights/decoder_model.onnx* 2>/dev/null
  cp "$2"/* "$SUB"/weights/ 2>/dev/null
  mv "$SUB"/weights/encoder-model.onnx        "$SUB"/weights/encoder_model.onnx
  mv "$SUB"/weights/decoder_joint-model.onnx  "$SUB"/weights/decoder_model.onnx
  sed -i 's/^CHUNK_NEW = 56/CHUNK_NEW = 16/' "$SUB"/model.py
  echo "CHUNK_NEW now: $(grep -m1 'CHUNK_NEW =' "$SUB"/model.py)"
  ( cd "$SUB" && SAPC2_THREADS=4 python3 "$LD" --submission-dir "$SUB" --manifest-csv "$MAN" \
      --streaming-manifest-csv "$MAN" --data-root /workspace/SAPC2 \
      --out-csv /dev/shm/$1.csv --out-partial-json /dev/shm/$1.json ) 2>&1 | tail -8
  echo "--- CER (two-ref official norm) + hyp sanity for $1 ---"
  python3 - "$MAN" /dev/shm/$1.csv <<'PY'
import csv,sys
sys.path.insert(0,"/workspace/sapc-nemotron/utils")
from normalizer.text_normalizer_hf import EnglishTextNormalizer
nm=EnglishTextNormalizer()
man,hyp=sys.argv[1],sys.argv[2]
R={r['id']:r for r in csv.DictReader(open(man))}
def cer(h,r):
    h,r=list(h),list(r)
    if not r:return 0.0 if not h else 1.0
    dp=list(range(len(r)+1))
    for i in range(1,len(h)+1):
        p=dp[0];dp[0]=i
        for j in range(1,len(r)+1):
            c=dp[j];dp[j]=min(dp[j]+1,dp[j-1]+1,p+(h[i-1]!=r[j-1]));p=c
    return min(1.0,dp[len(r)]/len(r))
tot=0;n=0;emp=0;rows=list(csv.DictReader(open(hyp)))
for r in rows:
    m=R.get(r['id']);
    if not m: continue
    raw=r['raw_hypos']; h=nm.norm(raw)
    c=min(cer(h,nm.norm(m.get('norm_text_with_disfluency',''))),cer(h,nm.norm(m.get('norm_text_without_disfluency',''))))
    tot+=c;n+=1; emp+= (1 if not raw.strip() else 0)
print(f"  N={n} meanCER={100*tot/max(1,n):.1f}% empty={emp}")
for r in rows[:5]: print(f"   HYP={r['raw_hypos'][:55]!r}")
PY
}

step "A. ZERO-SHOT base @ [70,1] via local_decode (harness validation)"
build_and_decode base70_1 "$OUT/export_base70_1"

step "B. WINNER (full encoder-only) @ [70,1] via local_decode"
build_and_decode full70_1 "$OUT/export_full70_1"

step "C. latency partial-json sanity (TTFT/TTLT come from compute_latency on full Dev_streaming later)"
ls -la /dev/shm/base70_1.json /dev/shm/full70_1.json 2>/dev/null
echo "DEPLOY_DECODE_DONE"
