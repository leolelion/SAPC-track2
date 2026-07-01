#!/usr/bin/env bash
# Mac-local: run encoder-only SOS dynamic-int8 build and gates on RunPod.
set -uo pipefail

ID=3dwiczo41jeg1y
KEY=/Users/o/.runpod/ssh/RunPod-Key-Go
O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=40 -o ServerAliveInterval=20"
REPO=/Users/o/Downloads/SAPC-template
RUN_LOG=/Users/o/Downloads/encoder_sos_int8_autorun.log
LOCAL_ART=/Users/o/Downloads/nemotron_encoder_sos_int8_artifacts
: > "$RUN_LOG"; exec > >(tee -a "$RUN_LOG") 2>&1
echo "$$" > /tmp/sapc_autorun.lock

stop_if_owner () {
  if [ "$(cat /tmp/sapc_autorun.lock 2>/dev/null)" = "$$" ]; then
    echo "[autorun] stopping pod ..."
    runpodctl pod stop $ID 2>&1 | grep -i desiredStatus
  else
    echo "[autorun] newer autorun owns pod -> not stopping"
  fi
}

try_start_pod () {
  runpodctl pod start "$ID" >/dev/null 2>&1 &
  spid=$!
  for _ in $(seq 1 45); do
    kill -0 "$spid" 2>/dev/null || { wait "$spid"; return $?; }
    sleep 1
  done
  echo "[autorun] pod start still blocked after 45s; retrying later"
  kill "$spid" 2>/dev/null
  wait "$spid" 2>/dev/null
  return 124
}

echo "[autorun] $(date) waiting for GPU slot ..."
for i in $(seq 1 360); do
  echo "[autorun] start attempt $i ..."
  try_start_pod; rc=$?
  echo "[autorun] start attempt $i exit=$rc"
  sleep 5
  runpodctl get pod $ID 2>/dev/null | grep -q RUNNING && { echo "[autorun] RUNNING (try $i)"; break; }
  echo "[autorun] still not RUNNING (try $i)"
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
[ -n "$PORT" ] || { echo "[autorun] NO_PORT"; stop_if_owner; exit 1; }

upload_one () {
  src="$1"; dst="$2"
  if ! scp -P $PORT -i $KEY $O "$src" root@$HOST:"$dst" >/dev/null; then
    echo "[autorun] UPLOAD_FAILED $src -> $dst"
    stop_if_owner
    exit 1
  fi
}

upload_one "$REPO/scripts/run_encoder_sos_int8.sh" /workspace/run_encoder_sos_int8.sh
upload_one "$REPO/scripts/run_package_encoder_sos.sh" /workspace/run_package_encoder_sos.sh
upload_one "$REPO/scripts/quantize_nemotron_encoder.py" /workspace/quantize_nemotron_encoder.py
upload_one "$REPO/scripts/compare_cer_pairs.py" /workspace/compare_cer_pairs.py
upload_one "$REPO/scripts/streaming_cer_bootstrap.py" /workspace/streaming_cer_bootstrap.py
upload_one "$REPO/scripts/bench_nemotron_multiproc.py" /workspace/bench_nemotron_multiproc.py
upload_one "$REPO/scripts/debug_sgml_pred_mismatch.py" /workspace/debug_sgml_pred_mismatch.py

if ! ssh -p $PORT -i $KEY $O root@$HOST 'cd /workspace && bash run_encoder_sos_int8.sh'; then
  echo "[autorun] REMOTE_RUN_FAILED"
fi

mkdir -p "$LOCAL_ART"
if ! scp -P $PORT -i $KEY $O root@$HOST:/workspace/finetune/nemo_ft/encoder_sos_int8.log /Users/o/Downloads/encoder_sos_int8.log >/dev/null; then
  echo "[autorun] LOG_PULL_FAILED"
fi
scp -P $PORT -i $KEY $O root@$HOST:'/workspace/finetune/nemo_ft/artifacts/enc_sos_int8_*' "$LOCAL_ART"/ >/dev/null 2>&1
scp -P $PORT -i $KEY $O root@$HOST:'/workspace/finetune/nemo_ft/artifacts/official_debug_enc_sos_int8_*' "$LOCAL_ART"/ >/dev/null 2>&1
scp -P $PORT -i $KEY $O root@$HOST:/workspace/finetune/nemo_ft/artifacts/nemotron_encoder_sos_int8_submission.zip.sha256 "$LOCAL_ART"/ >/dev/null 2>&1

echo "========== INSPECT LOG =========="
cat /Users/o/Downloads/encoder_sos_int8.log 2>/dev/null
echo "================================="
stop_if_owner
echo "[autorun] DONE $(date)"
