#!/usr/bin/env bash
# Mac-local: wait for a GPU slot, start pod, run exp A + beam-4 gate, pull RESULTS, stop pod.
ID=3dwiczo41jeg1y
KEY=/Users/o/.runpod/ssh/RunPod-Key-Go
O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=40 -o ServerAliveInterval=20"
REPO=/Users/o/Downloads/SAPC-template
SCRIPTS="run_expA_beam4.sh build_diag_manifest.py diag_crosstab.py nemotron_repro_analyze.py setup_zf_sub.sh"

echo "[autorun] $(date) waiting for GPU slot ..."
for i in $(seq 1 60); do
  runpodctl pod start $ID >/dev/null 2>&1; sleep 5
  runpodctl get pod $ID 2>/dev/null | grep -q RUNNING && { echo "[autorun] RUNNING (try $i)"; break; }
  sleep 55
done
runpodctl get pod $ID 2>/dev/null | grep -q RUNNING || { echo "[autorun] POD_NEVER_STARTED"; exit 1; }
sleep 25
EP=$(runpodctl get pod $ID -a 2>/dev/null | grep -oE '[0-9.]+:[0-9]+->22' | sed 's/->22//' | sed -n '1p')
HOST=${EP%:*}; PORT=${EP##*:}; echo "[autorun] $HOST:$PORT"
[ -n "$PORT" ] || { echo "[autorun] NO_PORT"; exit 1; }

for f in $SCRIPTS; do scp -P $PORT -i $KEY $O "$REPO/scripts/$f" root@$HOST:/workspace/ >/dev/null 2>&1; done
ssh -p $PORT -i $KEY $O root@$HOST 'mkdir -p /workspace/finetune/zf_a1_sub' 2>/dev/null
scp -P $PORT -i $KEY $O /tmp/a1_pkg/model.py      root@$HOST:/workspace/finetune/zf_a1_sub/model.py   >/dev/null 2>&1
scp -P $PORT -i $KEY $O /tmp/zf_assets/tokens.txt root@$HOST:/workspace/finetune/zf_a1_sub/tokens.txt >/dev/null 2>&1
scp -P $PORT -i $KEY $O /tmp/beam4_build/model.py root@$HOST:/workspace/zf_beam4_model.py             >/dev/null 2>&1

echo "[autorun] launching exp A + beam-4 ..."
ssh -p $PORT -i $KEY $O root@$HOST 'cd /workspace && nohup bash run_expA_beam4.sh >/workspace/expA.log 2>&1 & echo launched' 2>/dev/null
for i in $(seq 1 120); do
  ssh -p $PORT -i $KEY $O root@$HOST 'grep -q BATTERY_DONE /workspace/finetune/eval/expA_beam4/RESULTS.txt 2>/dev/null && echo OK' 2>/dev/null | grep -q OK && { echo "[autorun] DONE (poll $i)"; break; }
  sleep 30
done
scp -P $PORT -i $KEY $O root@$HOST:/workspace/finetune/eval/expA_beam4/RESULTS.txt /Users/o/Downloads/expA_RESULTS.txt 2>/dev/null
echo "========== RESULTS =========="; cat /Users/o/Downloads/expA_RESULTS.txt 2>/dev/null || echo "(none)"; echo "============================="
echo "[autorun] stopping pod ..."; runpodctl pod stop $ID 2>&1 | grep -i desiredStatus
echo "[autorun] DONE $(date)"
