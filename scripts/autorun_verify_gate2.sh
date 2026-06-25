#!/usr/bin/env bash
# Mac-local: wait for GPU, start pod, run Gate-2 verification (speaker disjointness + apples-to-apples
# zero-shot baseline), pull log, stop pod. Port-race fixed.
ID=3dwiczo41jeg1y
KEY=/Users/o/.runpod/ssh/RunPod-Key-Go
O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=40 -o ServerAliveInterval=20"
REPO=/Users/o/Downloads/SAPC-template

echo "[autorun] $(date) waiting for GPU slot (~6h window) ..."
for i in $(seq 1 360); do
  runpodctl pod start $ID >/dev/null 2>&1; sleep 5
  runpodctl get pod $ID 2>/dev/null | grep -q RUNNING && { echo "[autorun] RUNNING (try $i) $(date)"; break; }
  sleep 55
done
runpodctl get pod $ID 2>/dev/null | grep -q RUNNING || { echo "[autorun] POD_NEVER_STARTED"; exit 1; }
sleep 20
PORT=""; HOST=""
for k in $(seq 1 15); do
  EP=$(runpodctl get pod $ID -a 2>/dev/null | grep -oE '[0-9.]+:[0-9]+->22' | sed 's/->22//' | sed -n '1p')
  HOST=${EP%:*}; PORT=${EP##*:}
  [ -n "$PORT" ] && { echo "[autorun] endpoint $HOST:$PORT (after $k tries)"; break; }
  sleep 10
done
[ -n "$PORT" ] || { echo "[autorun] NO_PORT -> stopping pod"; runpodctl pod stop $ID >/dev/null 2>&1; exit 1; }

for f in verify_gate2.sh nemo_eval_diag.py; do
  scp -P $PORT -i $KEY $O "$REPO/scripts/$f" root@$HOST:/workspace/ >/dev/null 2>&1
done
echo "[autorun] launching verification (speaker check + zero-shot baseline ~10min) ..."
ssh -p $PORT -i $KEY $O root@$HOST 'cd /workspace && nohup bash verify_gate2.sh >/workspace/verify_outer.log 2>&1 & echo launched' 2>/dev/null

for i in $(seq 1 80); do        # up to ~40 min
  ssh -p $PORT -i $KEY $O root@$HOST 'grep -q VERIFY_DONE /workspace/finetune/nemo_ft/verify_gate2.log 2>/dev/null && echo OK' 2>/dev/null | grep -q OK && { echo "[autorun] DONE (poll $i)"; break; }
  sleep 30
done
scp -P $PORT -i $KEY $O root@$HOST:/workspace/finetune/nemo_ft/verify_gate2.log /Users/o/Downloads/verify_gate2.log 2>/dev/null
echo "========== VERIFY LOG =========="; grep -avE 'CPU_DIAGNOSTIC|it/s\]' /Users/o/Downloads/verify_gate2.log 2>/dev/null | tail -50 || echo "(none)"; echo "================================"
echo "[autorun] stopping pod ..."; runpodctl pod stop $ID 2>&1 | grep -i desiredStatus
echo "[autorun] DONE $(date)"
