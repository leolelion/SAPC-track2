#!/usr/bin/env bash
# Mac-local orchestrator: wait for a GPU slot, start the pod, run the full diagnostic battery,
# pull RESULTS locally, then stop the pod. Resilient to transient ssh/start failures.
ID=3dwiczo41jeg1y
KEY=/Users/o/.runpod/ssh/RunPod-Key-Go
O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=40 -o ServerAliveInterval=20"
REPO=/Users/o/Downloads/SAPC-template
SCRIPTS="build_diag_manifest.py diag_crosstab.py nemotron_blank_probe.py nemotron_altdecode_empties.py setup_zf_sub.sh run_diag_battery.sh nemotron_repro_run.sh nemotron_repro_analyze.py"

echo "[autorun] $(date) waiting for a GPU slot to start pod $ID ..."
for i in $(seq 1 60); do            # up to ~60 min of retries
  runpodctl pod start $ID >/dev/null 2>&1
  sleep 5
  if runpodctl get pod $ID 2>/dev/null | grep -q RUNNING; then echo "[autorun] pod RUNNING (try $i)"; break; fi
  sleep 55
done
runpodctl get pod $ID 2>/dev/null | grep -q RUNNING || { echo "[autorun] POD_NEVER_STARTED — aborting"; exit 1; }

sleep 25                            # let sshd come up
EP=$(runpodctl get pod $ID -a 2>/dev/null | grep -oE '[0-9.]+:[0-9]+->22' | sed 's/->22//' | sed -n '1p')
HOST=${EP%:*}; PORT=${EP##*:}
echo "[autorun] endpoint $HOST:$PORT"
[ -n "$PORT" ] || { echo "[autorun] NO_PORT — aborting"; exit 1; }

echo "[autorun] staging scripts + zipformer assets ..."
for f in $SCRIPTS; do scp -P $PORT -i $KEY $O "$REPO/scripts/$f" root@$HOST:/workspace/ >/dev/null 2>&1; done
ssh -p $PORT -i $KEY $O root@$HOST 'mkdir -p /workspace/finetune/zf_a1_sub' 2>/dev/null
scp -P $PORT -i $KEY $O /tmp/a1_pkg/model.py     root@$HOST:/workspace/finetune/zf_a1_sub/model.py   >/dev/null 2>&1
scp -P $PORT -i $KEY $O /tmp/zf_assets/tokens.txt root@$HOST:/workspace/finetune/zf_a1_sub/tokens.txt >/dev/null 2>&1

echo "[autorun] launching battery ..."
ssh -p $PORT -i $KEY $O root@$HOST 'cd /workspace && nohup bash run_diag_battery.sh >/workspace/battery.log 2>&1 & echo launched' 2>/dev/null

echo "[autorun] waiting for BATTERY_DONE ..."
for i in $(seq 1 120); do           # up to ~60 min
  if ssh -p $PORT -i $KEY $O root@$HOST 'grep -q BATTERY_DONE /workspace/finetune/eval/diag_battery/RESULTS.txt 2>/dev/null && echo OK' 2>/dev/null | grep -q OK; then
    echo "[autorun] BATTERY_DONE (poll $i)"; break; fi
  sleep 30
done

echo "[autorun] pulling RESULTS ..."
scp -P $PORT -i $KEY $O root@$HOST:/workspace/finetune/eval/diag_battery/RESULTS.txt /Users/o/Downloads/diag_RESULTS.txt 2>/dev/null
echo "========== RESULTS =========="
cat /Users/o/Downloads/diag_RESULTS.txt 2>/dev/null || echo "(no RESULTS pulled)"
echo "============================="

echo "[autorun] stopping pod ..."
runpodctl pod stop $ID 2>&1 | grep -i desiredStatus
echo "[autorun] DONE $(date)"
