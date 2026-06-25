#!/usr/bin/env bash
# Mac-local: wait for GPU, start pod, build isolated NeMo venv + verify .nemo load + characterize,
# pull log, stop pod. Has the port-race fix (won't leave a pod billing).
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

scp -P $PORT -i $KEY $O "$REPO/scripts/setup_nemo_venv.sh" "$REPO/scripts/nemo_characterize.py" root@$HOST:/workspace/ >/dev/null 2>&1
echo "[autorun] launching venv build (install ~15-30 min) ..."
ssh -p $PORT -i $KEY $O root@$HOST 'cd /workspace && nohup bash setup_nemo_venv.sh >/workspace/venv_outer.log 2>&1 & echo launched' 2>/dev/null

for i in $(seq 1 180); do        # up to ~90 min (install + .nemo load)
  ssh -p $PORT -i $KEY $O root@$HOST 'grep -q NEMO_VENV_DONE /workspace/finetune/nemo_ft/venv_setup.log 2>/dev/null && echo OK' 2>/dev/null | grep -q OK && { echo "[autorun] DONE (poll $i)"; break; }
  sleep 30
done
scp -P $PORT -i $KEY $O root@$HOST:/workspace/finetune/nemo_ft/venv_setup.log /Users/o/Downloads/nemo_venv_setup.log 2>/dev/null
echo "========== VENV SETUP LOG =========="; grep -avE 'CPU_DIAGNOSTIC' /Users/o/Downloads/nemo_venv_setup.log 2>/dev/null | tail -70 || echo "(none)"; echo "===================================="
echo "[autorun] stopping pod ..."; runpodctl pod stop $ID 2>&1 | grep -i desiredStatus
echo "[autorun] DONE $(date)"
