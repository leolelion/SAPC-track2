#!/usr/bin/env bash
# H7 base-model MECHANISM diagnostic (finetuned .nemo lost pod+local, 2026-07-22).
# Arm A = transcribe() masked single-pass; Arm C = NeMo-native cache-aware streaming.
# Same BASE nemotron, same [70,1], same Dev_diag, SAME two-ref scorer. Isolates whether
# NeMo's OWN chunked-cache streaming diverges from its masked pass (weight-independent
# mechanism test for the ~10pt collapse). NOT a submission; hand-rolled CER = proxy, diag only.
set -uo pipefail
OUT=/workspace/finetune/nemo_base_ac; mkdir -p "$OUT"
LOG="$OUT/base_ac.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
DIAG=/workspace/SAPC2/manifest/Dev_diag.csv
DATA=/workspace/SAPC2
NM_UTILS=/workspace/sapc-nemotron/utils

# --- resolve base .nemo (follow HF snapshot symlink -> blob) ---
SNAP=$(find /workspace/.cache -path '*nemotron-speech-streaming-en-0.6b*' -name '*.nemo' 2>/dev/null | head -1)
BASE=$(readlink -f "$SNAP" 2>/dev/null || echo "$SNAP")
[ -f "$BASE" ] || { echo "FATAL: base .nemo not resolvable ($SNAP -> $BASE)"; exit 2; }
SZ=$(stat -c %s "$BASE" 2>/dev/null || echo 0)
echo "BASE=$BASE size=$SZ"
[ "$SZ" -gt 1000000000 ] || { echo "FATAL: base .nemo too small ($SZ) — blob missing"; exit 2; }

# --- build NeMo json manifest from Dev_diag ---
MAN=$OUT/dev_diag_nemo.json
python3 - "$DIAG" "$DATA" "$MAN" <<'PY'
import csv,json,os,sys
diag,data,out=sys.argv[1],sys.argv[2],sys.argv[3]
rows=list(csv.DictReader(open(diag)))
with open(out,"w") as f:
    for r in rows:
        f.write(json.dumps({"audio_filepath":os.path.join(data,r["audio_filepath"]),
          "duration":float(r["duration"]),"text":r["norm_text_without_disfluency"]})+"\n")
print("manifest rows",len(rows))
PY

# --- ARM A: transcribe() @ [70,1] (masked single pass) ---
echo "###### ARM A  base transcribe() @ [70,1] ######"
python3 - "$BASE" "$MAN" "$OUT/transcribe_base.json" <<'PY'
import sys,json,torch,nemo.collections.asr as na
nemo_path,man,out=sys.argv[1],sys.argv[2],sys.argv[3]
rows=[json.loads(l) for l in open(man) if l.strip()]
wavs=[r["audio_filepath"] for r in rows]
m=na.models.ASRModel.restore_from(nemo_path,map_location="cuda")
try: m.encoder.set_default_att_context_size([70,1])
except Exception as e: print("att set failed:",e)
m.eval()
with torch.no_grad():
    hyps=m.transcribe(wavs,batch_size=16)
hyps=[(h.text if hasattr(h,"text") else h) for h in hyps]
with open(out,"w") as f:
    for r,h in zip(rows,hyps):
        f.write(json.dumps({"audio_filepath":r["audio_filepath"],"pred_text":h or ""})+"\n")
print("ARM_A rows",len(hyps))
PY

# --- ARM C: NeMo-native cache-aware streaming @ [70,1] ---
SCRIPT=/workspace/stream_infer.py
NEMO_REF="${NEMO_REF:-main}"
INFER_URL="https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/${NEMO_REF}/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"
[ -f "$SCRIPT" ] || curl -sL -o "$SCRIPT" "$INFER_URL"
[ -s "$SCRIPT" ] || { echo "FATAL: infer script download failed from $INFER_URL"; exit 3; }
for a in --asr_model --manifest_file --att_context_size --output_path; do
  grep -q -- "$a" "$SCRIPT" || { echo "FATAL: infer script missing arg '$a' (NeMo drift ref=$NEMO_REF)"; exit 3; }
done
echo "###### ARM C  base NeMo-native streaming @ [70,1] ######"
python3 "$SCRIPT" --asr_model "$BASE" --manifest_file "$MAN" --batch_size 32 \
  --att_context_size 70 1 --output_path "$OUT/stream_base70_1.json" 2>&1 | tail -25 || echo "ARM_C STREAM_FAIL"

# --- SCORE both with IDENTICAL two-ref CER + normalizer (apples-to-apples A vs C) ---
echo "=== CER (two-ref official normalizer, per etiology) — ARM A (transcribe) vs ARM C (stream) ==="
python3 - "$DIAG" "$NM_UTILS" "$OUT/transcribe_base.json" "$OUT/stream_base70_1.json" <<'PY'
import csv,os,json,sys
sys.path.insert(0,sys.argv[2])
from normalizer.text_normalizer_hf import EnglishTextNormalizer
nm=EnglishTextNormalizer()
diag={os.path.basename(r["audio_filepath"]):r for r in csv.DictReader(open(sys.argv[1]))}
def cer(h,r):
    h,r=list(h),list(r)
    if not r: return 0.0 if not h else 1.0
    dp=list(range(len(r)+1))
    for i in range(1,len(h)+1):
        p=dp[0]; dp[0]=i
        for j in range(1,len(r)+1):
            c=dp[j]; dp[j]=min(dp[j]+1,dp[j-1]+1,p+(h[i-1]!=r[j-1])); p=c
    return min(1.0,dp[len(r)]/len(r))
from collections import defaultdict
for f in sys.argv[3:]:
    if not os.path.exists(f): print(f"\n=== MISSING {f} ==="); continue
    rows=[json.loads(l) for l in open(f) if l.strip()]
    agg=defaultdict(lambda:[0,0.0,0]); tot=[0,0.0,0]
    for r in rows:
        bn=os.path.basename(r.get("audio_filepath","")); m=diag.get(bn)
        if not m: continue
        pred=r.get("pred_text",r.get("pred","")) or ""
        h=nm.norm(pred)
        c=min(cer(h,nm.norm(m.get("norm_text_with_disfluency",""))),cer(h,nm.norm(m.get("norm_text_without_disfluency",""))))
        e=1 if not pred.strip() else 0
        for b in (agg[m["etiology"]],tot): b[0]+=1;b[1]+=c;b[2]+=e
    print(f"\n=== {os.path.basename(f)} ===")
    print(f"  ALL n={tot[0]} CER={100*tot[1]/max(1,tot[0]):.1f}% empty={tot[2]} ({100*tot[2]/max(1,tot[0]):.0f}%)")
    for k in sorted(agg): print(f"  {k:20s} n={agg[k][0]:4d} CER={100*agg[k][1]/max(1,agg[k][0]):.1f}% empty={agg[k][2]}")
PY
echo "BASE_AC_DONE"
</content>
