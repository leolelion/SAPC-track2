#!/usr/bin/env bash
# Unattended post-training chain for one Arm B rung.
#   armb_post.sh <tag>       (expects /workspace/parakeet_ft/armB_<tag>/ft_smoke_joint_unfreeze.nemo)
#
# Order matters: the CHEAP proxy gate runs first (GATE-TRAIN), the EXPENSIVE real harness second.
# The proxy NEVER authorizes a ship claim — only the official scorer's Dev_diag CER does (GATE-SHIP).
set -uo pipefail
TAG="$1"
DIR=/workspace/parakeet_ft/armB_${TAG}
NEMO=$DIR/ft_smoke_joint_unfreeze.nemo
OUT=/workspace/parakeet_ft/gate_armB_${TAG}
mkdir -p "$OUT"
source /workspace/nemoenv/bin/activate
export PATH="/workspace/SCTK/bin:$PATH"
DATA=/workspace/SAPC2; MAN=$DATA/manifest
PROJ=/workspace/SAPC-template
SUB=$PROJ/track2_starting_kit/parakeet_realtime_ft

test -f "$NEMO" || { echo "ARMB_POST_${TAG}_FAIL missing $NEMO"; exit 2; }
echo "=== [$TAG] post-chain start $(date -u +%FT%TZ) ==="

# ---- 1. GATE-TRAIN: val CER / empties / insertion-rate, SAME val json + ctx as Arm A ----
python3 /workspace/val_metrics.py "$NEMO" /workspace/nemo_ft/val.json \
  "$OUT/valmetrics.json" 0 32 > "$OUT/valmetrics.log" 2>&1
echo "--- GATE-TRAIN (proxy: punct-stripped single-ref) ---"
python3 -c "
import json; a=json.load(open('/workspace/parakeet_ft/valmetrics_armA.json')); b=json.load(open('$OUT/valmetrics.json'))
for k in ['cer_pct','empty_count','insertion_rate_pct','deletion_rate_pct','substitution_rate_pct']:
    print(f'{k:24s} armA={a[k]:10.4f}  armB_${TAG}={b[k]:10.4f}')
g1=b['cer_pct']<a['cer_pct']; g2=b['empty_count']<a['empty_count']; g3=b['insertion_rate_pct']<=a['insertion_rate_pct']*1.15
print('GATE-TRAIN cer_better=',g1,'empties_better=',g2,'insertions_ok=',g3,'PASS=',g1 and g2 and g3)
"

# ---- 2. real harness, both passes (gate.sh already encodes the organizers' loop) ----
cp "$NEMO" "$SUB/weights/parakeet_realtime.nemo" || { echo "ARMB_POST_${TAG}_FAIL cp weights"; exit 3; }
cd $PROJ/track2_starting_kit

echo "--- real streaming pass: Dev_streaming (CER + TTFT/TTLT) ---"
python3 local_decode.py --submission-dir ./parakeet_realtime_ft \
  --manifest-csv $MAN/Dev_streaming.csv --streaming-manifest-csv $MAN/Dev_streaming.csv \
  --data-root $DATA --out-csv $OUT/devstream.predict.csv \
  --out-partial-json $OUT/devstream.partial.json > $OUT/devstream.decode.log 2>&1 \
  && echo "[devstream decode ok]" || echo "[devstream decode FAIL]"

echo "--- real accuracy pass: Dev_diag severe ---"
python3 local_decode.py --submission-dir ./parakeet_realtime_ft \
  --manifest-csv $MAN/Dev_diag.csv --data-root $DATA \
  --out-csv $OUT/devdiag.predict.csv > $OUT/devdiag.decode.log 2>&1 \
  && echo "[devdiag decode ok]" || echo "[devdiag decode FAIL]"

# ---- 3. OFFICIAL scorer (the only ship authority) ----
cd $PROJ
echo "--- GATE-SHIP-CER: official evaluate.sh, Dev_diag (two-ref sclite) ---"
./evaluate.sh --split Dev_diag --hyp-csv $OUT/devdiag.predict.csv \
  --start_stage 1 --stop_stage 2 > $OUT/devdiag.official.log 2>&1 \
  && echo "[official devdiag ok]" || echo "[official devdiag FAIL]"
grep -aiE "cer|wer" $OUT/devdiag.official.log | tail -12

echo "--- GATE-SHIP-LAT: official latency, Dev_streaming ---"
./evaluate.sh --start_stage 3 --stop_stage 3 \
  --partial-json $OUT/devstream.partial.json \
  --manifest-csv $MAN/Dev_streaming.csv > $OUT/devstream.latency.log 2>&1 \
  && echo "[official latency ok]" || echo "[official latency FAIL]"
grep -aiE "ttft|ttlt|p50|p90|mean" $OUT/devstream.latency.log | tail -12

echo "ARMB_POST_${TAG}_DONE $(date -u +%FT%TZ)"
