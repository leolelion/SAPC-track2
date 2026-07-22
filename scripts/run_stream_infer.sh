#!/usr/bin/env bash
# Deployment loop step 3: TRUE chunk-faithful streaming CER via NeMo's authoritative cache-aware streaming
# inference (handles chunk_size/valid_out_len/drop_extra_pre_encoded correctly) on winner + base at [70,1].
# Resolves confound C and gives a reference to validate the submission model.py against.
# NOTE (research/45, 2026-07-22): the base-model A/B/C run already showed the pipeline is FAITHFUL
# (transcribe 43.4 ≈ NeMo-stream 43.8 ≈ our-export 43.4). This script is the finetuned re-test — only
# useful once a finetuned .nemo exists again. Uses NeMo's CURRENT Hydra interface (argparse was removed).
set -uo pipefail
OUT=/workspace/finetune/nemo_ft
LOG="$OUT/stream_infer.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
SCRIPT=/workspace/stream_infer.py
NEMO_REF="${NEMO_REF:-main}"   # pin to a known-good tag/commit at launch (NEMO_REF=...) to defend vs NeMo API drift
INFER_URL="https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/${NEMO_REF}/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"
DIAG=/workspace/SAPC2/manifest/Dev_diag.csv
MAN=$OUT/dev_diag_nemo.json
[ -f "$SCRIPT" ] || curl -sL -o "$SCRIPT" "$INFER_URL"
[ -s "$SCRIPT" ] || { echo "FATAL: infer script download empty/failed from $INFER_URL -> STOP"; exit 2; }
[ -f "$MAN" ] || python3 - <<PY
import csv,json,os
rows=list(csv.DictReader(open("$DIAG")))
open("$MAN","w").write("\n".join(json.dumps({"audio_filepath":os.path.join("/workspace/SAPC2",r["audio_filepath"]),
  "duration":float(r["duration"]),"text":r["norm_text_without_disfluency"]}) for r in rows))
print("manifest rows",len(rows))
PY

echo "=== stream_infer.py Hydra config fields ==="
grep -nE "model_path|dataset_manifest|output_path|att_context_size" "$SCRIPT" 2>/dev/null | head -10
# HARD GATE: NeMo migrated this example argparse->Hydra (~2026-07; research/45). Abort if the Hydra
# field names this script depends on ever drift again (defends vs future API drift).
for f in model_path dataset_manifest output_path att_context_size; do
  grep -q -- "$f" "$SCRIPT" || { echo "FATAL: infer script missing Hydra field '$f' (NeMo API drift on ref=$NEMO_REF) -> STOP"; exit 3; }
done
echo "OK: all required Hydra fields present (ref=$NEMO_REF)"

# ARM A (reference): transcribe() @ [70,1] on the SAME v1 checkpoint, SAME scorer -> fresh internal control.
# Decision rule: the streaming collapse is OUR export bug IFF arm-C CER <= this arm-A CER + 2pts.
echo "###### ARM A (reference) transcribe() @ [70,1] on full_enc/ft_smoke_encoder_only.nemo ######"
python3 /workspace/nemo_eval_diag.py "$OUT/full_enc/ft_smoke_encoder_only.nemo" "$DIAG" /workspace/SAPC2 "[70,1]" 2>&1 | tail -30 || echo "ARM_A_FAIL (transcribe reference did not complete)"

run_stream () { # $1=tag $2=nemo_path  (Hydra iface: output_path is treated as a DIRECTORY; the actual
  echo "###### STREAM $1 ($2) at [70,1] (Hydra iface) ######"  # hyps land inside as streaming_out_*.json)
  rm -rf "$OUT/stream_$1.json"
  python3 "$SCRIPT" model_path="$2" dataset_manifest="$MAN" batch_size=32 \
    'att_context_size=[70,1]' output_path="$OUT/stream_$1.json" 2>&1 | tail -20 || echo "$1 STREAM_FAIL"
}
run_stream full70_1 "$OUT/full_enc/ft_smoke_encoder_only.nemo"
run_stream base70_1 "$OUT/nemotron-speech-streaming-en-0.6b.nemo"

echo "=== CER from streaming hyps (two-ref normalizer, per etiology; position-aligned to Dev_diag) ==="
# Hydra infer rows carry {pred_text,text,wer} with NO audio_filepath -> align by manifest position and
# ASSERT text match (streaming order == manifest order == Dev_diag order; verified 0 mismatches base run).
python3 - full70_1 base70_1 <<'PY'
import csv,os,glob,json,sys
sys.path.insert(0,"/workspace/sapc-nemotron/utils")
from normalizer.text_normalizer_hf import EnglishTextNormalizer
nm=EnglishTextNormalizer()
diag=list(csv.DictReader(open("/workspace/SAPC2/manifest/Dev_diag.csv")))
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
for tag in sys.argv[1:]:
    base=f"/workspace/finetune/nemo_ft/stream_{tag}.json"
    cand=sorted(glob.glob(f"{base}/streaming_out_*.json")) or ([base] if os.path.isfile(base) else [])
    if not cand: print(f"\n=== {tag}: NO OUTPUT ==="); continue
    rows=[json.loads(l) for l in open(cand[0]) if l.strip()]
    if len(rows)!=len(diag): print(f"\n=== {tag}: LEN MISMATCH stream={len(rows)} diag={len(diag)} -> skip ==="); continue
    mism=sum(1 for i in range(len(diag)) if rows[i].get("text","").strip()!=diag[i]["norm_text_without_disfluency"].strip())
    agg=defaultdict(lambda:[0,0.0,0]); tot=[0,0.0,0]
    for i in range(len(diag)):
        m=diag[i]; pred=rows[i].get("pred_text","") or ""
        h=nm.norm(pred)
        c=min(cer(h,nm.norm(m.get("norm_text_with_disfluency",""))),cer(h,nm.norm(m.get("norm_text_without_disfluency",""))))
        e=1 if not pred.strip() else 0
        for b in (agg[m["etiology"]],tot): b[0]+=1; b[1]+=c; b[2]+=e
    print(f"\n=== stream_{tag} (align_mismatch={mism}/{len(diag)}) ===")
    print(f"  ALL n={tot[0]} CER={100*tot[1]/max(1,tot[0]):.1f}% empty={tot[2]} ({100*tot[2]/max(1,tot[0]):.0f}%)")
    for k in sorted(agg): print(f"  {k:18s} n={agg[k][0]:4d} CER={100*agg[k][1]/max(1,agg[k][0]):.1f}% empty={agg[k][2]}")
PY
echo "STREAM_INFER_DONE"
