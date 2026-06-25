#!/usr/bin/env bash
# Mac-local: wait for GPU, start pod, run the 2 pre-full-run experiments, pull log, stop pod. Port-race fixed.
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

for f in run_prerun_exp.sh prep_nemo_manifest.py nemo_finetune.py nemo_eval_diag.py build_diag_manifest.py; do
  scp -P $PORT -i $KEY $O "$REPO/scripts/$f" root@$HOST:/workspace/ >/dev/null 2>&1
done
echo "[autorun] launching pre-run experiments (ablation train ~1.5h) ..."
ssh -p $PORT -i $KEY $O root@$HOST 'cd /workspace && nohup bash run_prerun_exp.sh >/workspace/prerun_outer.log 2>&1 & echo launched' 2>/dev/null

for i in $(seq 1 400); do        # up to ~3.3h
  ssh -p $PORT -i $KEY $O root@$HOST 'grep -q PRERUN_DONE /workspace/finetune/nemo_ft/prerun_exp.log 2>/dev/null && echo OK' 2>/dev/null | grep -q OK && { echo "[autorun] DONE (poll $i)"; break; }
  sleep 30
done
scp -P $PORT -i $KEY $O root@$HOST:/workspace/finetune/nemo_ft/prerun_exp.log /Users/o/Downloads/prerun_exp.log 2>/dev/null
echo "========== PRE-RUN EXP LOG =========="; grep -avE 'CPU_DIAGNOSTIC|CUDA graphs|it/s\]|Epoch [0-9]' /Users/o/Downloads/prerun_exp.log 2>/dev/null | tail -90 || echo "(none)"; echo "===================================="
echo "[autorun] stopping pod ..."; runpodctl pod stop $ID 2>&1 | grep -i desiredStatus
echo "[autorun] DONE $(date)"
