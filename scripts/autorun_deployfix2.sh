#!/usr/bin/env bash
# Mac-local: wait for GPU, start pod, inspect NeMo ONNX export I/O, pull log, stop pod. PID-lock guard.
ID=3dwiczo41jeg1y
KEY=/Users/o/.runpod/ssh/RunPod-Key-Go
O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=40 -o ServerAliveInterval=20"
REPO=/Users/o/Downloads/SAPC-template
echo "$$" > /tmp/sapc_autorun.lock
echo "[autorun] $(date) waiting for GPU slot ..."
for i in $(seq 1 360); do
  runpodctl pod start $ID >/dev/null 2>&1; sleep 5
  runpodctl get pod $ID 2>/dev/null | grep -q RUNNING && { echo "[autorun] RUNNING (try $i)"; break; }
  sleep 55
done
runpodctl get pod $ID 2>/dev/null | grep -q RUNNING || { echo "[autorun] POD_NEVER_STARTED"; exit 1; }
sleep 20
PORT=""; HOST=""
for k in $(seq 1 15); do
  EP=$(runpodctl get pod $ID -a 2>/dev/null | grep -oE '[0-9.]+:[0-9]+->22' | sed 's/->22//' | sed -n '1p')
  HOST=${EP%:*}; PORT=${EP##*:}
  [ -n "$PORT" ] && { echo "[autorun] endpoint $HOST:$PORT"; break; }
  sleep 10
done
[ -n "$PORT" ] || { echo "[autorun] NO_PORT -> stopping pod"; runpodctl pod stop $ID >/dev/null 2>&1; exit 1; }
scp -P $PORT -i $KEY $O "$REPO/scripts/run_deploy_fix2.sh" root@$HOST:/workspace/ >/dev/null 2>&1
ssh -p $PORT -i $KEY $O root@$HOST 'cd /workspace && bash run_deploy_fix2.sh' 2>/dev/null
scp -P $PORT -i $KEY $O root@$HOST:/workspace/finetune/nemo_ft/deploy_fix2.log /Users/o/Downloads/deploy_fix2.log 2>/dev/null
echo "========== INSPECT LOG =========="; cat /Users/o/Downloads/deploy_fix2.log 2>/dev/null; echo "================================="
if [ "$(cat /tmp/sapc_autorun.lock 2>/dev/null)" = "$$" ]; then
  echo "[autorun] stopping pod ..."; runpodctl pod stop $ID 2>&1 | grep -i desiredStatus
else echo "[autorun] newer autorun owns pod -> not stopping"; fi
echo "[autorun] DONE $(date)"
