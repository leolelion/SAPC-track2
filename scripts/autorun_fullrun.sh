#!/usr/bin/env bash
# Mac-local: wait for GPU, start pod, run the full two-arm finetune, pull log, stop pod. Port-race fixed.
ID=3dwiczo41jeg1y
KEY=/Users/o/.runpod/ssh/RunPod-Key-Go
O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=40 -o ServerAliveInterval=20"
REPO=/Users/o/Downloads/SAPC-template

echo "$$" > /tmp/sapc_autorun.lock   # claim ownership; a newer autorun overwrites this -> older won't pod-stop
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

for f in run_fullrun.sh nemo_finetune.py nemo_eval_diag.py prep_nemo_manifest.py build_diag_manifest.py; do
  scp -P $PORT -i $KEY $O "$REPO/scripts/$f" root@$HOST:/workspace/ >/dev/null 2>&1
done
echo "[autorun] launching FULL RUN (wiring gate -> 2 arms, ~4-8h) ..."
ssh -p $PORT -i $KEY $O root@$HOST 'cd /workspace && nohup bash run_fullrun.sh >/workspace/fullrun_outer.log 2>&1 & echo launched' 2>/dev/null

for i in $(seq 1 800); do        # up to ~6.6h
  ssh -p $PORT -i $KEY $O root@$HOST 'grep -q FULLRUN_DONE /workspace/finetune/nemo_ft/fullrun.log 2>/dev/null && echo OK' 2>/dev/null | grep -q OK && { echo "[autorun] DONE (poll $i)"; break; }
  sleep 30
done
scp -P $PORT -i $KEY $O root@$HOST:/workspace/finetune/nemo_ft/fullrun.log /Users/o/Downloads/fullrun.log 2>/dev/null
echo "========== FULL RUN LOG =========="; grep -avE 'CPU_DIAGNOSTIC|CUDA graphs|it/s\]|Epoch [0-9]+:' /Users/o/Downloads/fullrun.log 2>/dev/null | tail -120 || echo "(none)"; echo "================================"
if [ "$(cat /tmp/sapc_autorun.lock 2>/dev/null)" = "$$" ]; then
  echo "[autorun] stopping pod ..."; runpodctl pod stop $ID 2>&1 | grep -i desiredStatus
else
  echo "[autorun] a NEWER autorun owns the pod -> NOT stopping (avoids killing a live run)"
fi
echo "[autorun] DONE $(date)"
