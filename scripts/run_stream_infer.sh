#!/usr/bin/env bash
# Deployment loop step 3: TRUE chunk-faithful streaming CER via NeMo's authoritative cache-aware streaming
# inference (handles chunk_size/valid_out_len/drop_extra_pre_encoded correctly) on winner + base at [70,1].
# Resolves confound C and gives a reference to validate the submission model.py against.
set -uo pipefail
OUT=/workspace/finetune/nemo_ft
LOG="$OUT/stream_infer.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
SCRIPT=/workspace/stream_infer.py
DIAG=/workspace/SAPC2/manifest/Dev_diag.csv
MAN=$OUT/dev_diag_nemo.json
[ -f "$SCRIPT" ] || curl -sL -o "$SCRIPT" https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/main/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py
[ -f "$MAN" ] || python3 - <<PY
import csv,json,os
rows=list(csv.DictReader(open("$DIAG")))
open("$MAN","w").write("\n".join(json.dumps({"audio_filepath":os.path.join("/workspace/SAPC2",r["audio_filepath"]),
  "duration":float(r["duration"]),"text":r["norm_text_without_disfluency"]}) for r in rows))
print("manifest rows",len(rows))
PY

echo "=== stream_infer.py argparse (exact arg names) ==="
grep -nE "add_argument" "$SCRIPT" 2>/dev/null | sed 's/.*add_argument(//' | head -40

run_stream () { # $1=tag $2=nemo_path
  echo "###### STREAM $1 ($2) at [70,1] ######"
  python3 "$SCRIPT" --asr_model "$2" --manifest_file "$MAN" --batch_size 32 \
    --att_context_size 70 1 --output_path "$OUT/stream_$1.json" 2>&1 | tail -20 || echo "$1 STREAM_FAIL"
}
run_stream full70_1 "$OUT/full_enc/ft_smoke_encoder_only.nemo"
run_stream base70_1 "$OUT/nemotron-speech-streaming-en-0.6b.nemo"

echo "=== CER from streaming hyps (two-ref official normalizer, per etiology) ==="
python3 - <<'PY'
import csv,os,glob,json,sys
sys.path.insert(0,"/workspace/sapc-nemotron/utils")
from normalizer.text_normalizer_hf import EnglishTextNormalizer
nm=EnglishTextNormalizer()
diag={os.path.basename(r["audio_filepath"]):r for r in csv.DictReader(open("/workspace/SAPC2/manifest/Dev_diag.csv"))}
def cer(h,r):
    h,r=list(h),list(r)
    if not r: return 0.0 if not h else 1.0
    dp=list(range(len(r)+1))
    for i in range(1,len(h)+1):
        p=dp[0]; dp[0]=i
        for j in range(1,len(r)+1):
            c=dp[j]; dp[j]=min(dp[j]+1,dp[j-1]+1,p+(h[i-1]!=r[j-1])); p=c
    return min(1.0,dp[len(r)]/len(r))
for f in sorted(glob.glob("/workspace/finetune/nemo_ft/stream_*.json")):
    rows=[json.loads(l) for l in open(f) if l.strip()]
    from collections import defaultdict
    agg=defaultdict(lambda:[0,0.0,0]); tot=[0,0.0,0]
    for r in rows:
        bn=os.path.basename(r.get("audio_filepath","")); m=diag.get(bn)
        if not m: continue
        pred=r.get("pred_text", r.get("pred",""))
        h=nm.norm(pred or "")
        c=min(cer(h,nm.norm(m.get("norm_text_with_disfluency",""))),cer(h,nm.norm(m.get("norm_text_without_disfluency",""))))
        e=1 if not (pred or "").strip() else 0
        for b in (agg[m["etiology"]],tot): b[0]+=1; b[1]+=c; b[2]+=e
    print(f"\n=== {os.path.basename(f)} ===")
    print(f"  ALL n={tot[0]} CER={100*tot[1]/max(1,tot[0]):.1f}% empty={tot[2]} ({100*tot[2]/max(1,tot[0]):.0f}%)")
    for k in sorted(agg): print(f"  {k:18s} n={agg[k][0]:4d} CER={100*agg[k][1]/max(1,agg[k][0]):.1f}% empty={agg[k][2]}")
PY
echo "STREAM_INFER_DONE"
