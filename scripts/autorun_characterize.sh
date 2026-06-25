#!/usr/bin/env bash
# Mac-local: wait for GPU, start pod, run Gate-0 characterization (NeMo install + .nemo download +
# load + cfg dump), pull results, stop pod. Read-only (no training, no submission).
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
for k in $(seq 1 15); do   # poll until the network endpoint is assigned (race fix)
  EP=$(runpodctl get pod $ID -a 2>/dev/null | grep -oE '[0-9.]+:[0-9]+->22' | sed 's/->22//' | sed -n '1p')
  HOST=${EP%:*}; PORT=${EP##*:}
  [ -n "$PORT" ] && { echo "[autorun] endpoint $HOST:$PORT (after ${k} tries)"; break; }
  sleep 10
done
[ -n "$PORT" ] || { echo "[autorun] NO_PORT after retries -> STOPPING pod to avoid cost leak"; runpodctl pod stop $ID >/dev/null 2>&1; exit 1; }

scp -P $PORT -i $KEY $O "$REPO/scripts/nemo_characterize.py" "$REPO/scripts/char_run.sh" root@$HOST:/workspace/ >/dev/null 2>&1
echo "[autorun] launching characterization (install can take 10-20 min) ..."
ssh -p $PORT -i $KEY $O root@$HOST 'cd /workspace && nohup bash char_run.sh >/workspace/char_outer.log 2>&1 & echo launched' 2>/dev/null

for i in $(seq 1 120); do          # up to ~60 min (install + 2.47GB download + load)
  ssh -p $PORT -i $KEY $O root@$HOST 'grep -q CHARACTERIZE_DONE /workspace/finetune/nemo_ft/characterize.log 2>/dev/null && echo OK' 2>/dev/null | grep -q OK && { echo "[autorun] DONE (poll $i)"; break; }
  sleep 30
done
scp -P $PORT -i $KEY $O root@$HOST:/workspace/finetune/nemo_ft/characterize.log /Users/o/Downloads/characterize.log 2>/dev/null
scp -P $PORT -i $KEY $O root@$HOST:/workspace/finetune/nemo_ft/cfg_dump.yaml     /Users/o/Downloads/nemo_cfg_dump.yaml 2>/dev/null
echo "========== CHARACTERIZE LOG =========="; cat /Users/o/Downloads/characterize.log 2>/dev/null || echo "(none)"; echo "======================================"
echo "[autorun] stopping pod ..."; runpodctl pod stop $ID 2>&1 | grep -i desiredStatus
echo "[autorun] DONE $(date)"
