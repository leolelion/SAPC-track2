#!/usr/bin/env bash
# Re-run of ARM C only (NeMo-native cache-aware streaming) after the main script's arg-gate caught
# NeMo main migrating this example argparse -> Hydra. Arm A (transcribe_base.json) already succeeded;
# reuse it. Hydra fields: model_path / dataset_manifest / output_path / att_context_size / batch_size.
set -uo pipefail
OUT=/workspace/finetune/nemo_base_ac
LOG="$OUT/base_c.log"; : > "$LOG"; exec > >(tee -a "$LOG") 2>&1
source /workspace/nemoenv/bin/activate
DIAG=/workspace/SAPC2/manifest/Dev_diag.csv
NM_UTILS=/workspace/sapc-nemotron/utils
MAN=$OUT/dev_diag_nemo.json
SCRIPT=/workspace/stream_infer.py

SNAP=$(find /workspace/.cache -path '*nemotron-speech-streaming-en-0.6b*' -name '*.nemo' 2>/dev/null | head -1)
BASE=$(readlink -f "$SNAP" 2>/dev/null || echo "$SNAP")
[ -f "$BASE" ] || { echo "FATAL: base .nemo not resolvable"; exit 2; }
[ -f "$MAN" ] || { echo "FATAL: manifest missing ($MAN) — arm A must have run"; exit 2; }
[ -f "$OUT/transcribe_base.json" ] || { echo "FATAL: transcribe_base.json missing — arm A did not save"; exit 2; }

# Hydra-interface gate (defends the NEW arg names the way the old gate defended the old ones).
for f in model_path dataset_manifest output_path att_context_size; do
  grep -q "$f" "$SCRIPT" || { echo "FATAL: infer script missing Hydra field '$f' — unexpected version"; exit 3; }
done
echo "OK: Hydra fields present"

echo "###### ARM C  base NeMo-native streaming @ [70,1] (Hydra) ######"
python3 "$SCRIPT" \
  model_path="$BASE" \
  dataset_manifest="$MAN" \
  output_path="$OUT/stream_base70_1.json" \
  batch_size=32 \
  'att_context_size=[70,1]' 2>&1 | tail -30 || echo "ARM_C STREAM_FAIL"

echo "=== output preview (first line of stream json) ==="
sed -n '1p' "$OUT/stream_base70_1.json" 2>/dev/null | cut -c1-300

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
    print(f"\n=== {os.path.basename(f)} (n_scored={tot[0]}) ===")
    print(f"  ALL n={tot[0]} CER={100*tot[1]/max(1,tot[0]):.1f}% empty={tot[2]} ({100*tot[2]/max(1,tot[0]):.0f}%)")
    for k in sorted(agg): print(f"  {k:20s} n={agg[k][0]:4d} CER={100*agg[k][1]/max(1,agg[k][0]):.1f}% empty={agg[k][2]}")
PY
echo "BASE_C_DONE"
</content>
