#!/usr/bin/env bash
# Pull encoder-SOS int8 run artifacts from RunPod, then stop the pod by default.
#
# This is intended for the "manual run already finished" case where the pod has
# results on /workspace (and possibly /dev/shm) but the local artifact pull was
# interrupted. It does not start the pod.
set -euo pipefail

ID=${ID:-3dwiczo41jeg1y}
KEY=${KEY:-/Users/o/.runpod/ssh/RunPod-Key-Go}
HOST=${HOST:-}
PORT=${PORT:-}
STOP_POD=${STOP_POD:-1}
LOCAL_ART=${LOCAL_ART:-/Users/o/Downloads/nemotron_encoder_sos_int8_artifacts}
REMOTE_TGZ=${REMOTE_TGZ:-/workspace/finetune/nemo_ft/artifacts/enc_sos_int8_pull_latest.tgz}
O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=40 -o ServerAliveInterval=20"

if [ -z "$HOST" ] || [ -z "$PORT" ]; then
  EP=$(runpodctl get pod "$ID" -a 2>/dev/null | grep -oE '[0-9.]+:[0-9]+->22' | sed 's/->22//' | sed -n '1p')
  HOST=${EP%:*}
  PORT=${EP##*:}
fi

if [ -z "$HOST" ] || [ -z "$PORT" ] || [ "$HOST" = "$PORT" ]; then
  echo "ERROR: could not determine SSH endpoint. Set HOST=... PORT=... explicitly."
  exit 1
fi

mkdir -p "$LOCAL_ART"
echo "[pull] endpoint $HOST:$PORT"
echo "[pull] collecting remote artifacts into $REMOTE_TGZ"

ssh -p "$PORT" -i "$KEY" $O root@"$HOST" "REMOTE_TGZ='$REMOTE_TGZ' bash -s" <<'REMOTE'
set -euo pipefail
REMOTE_TGZ="${REMOTE_TGZ:-/workspace/finetune/nemo_ft/artifacts/enc_sos_int8_pull_latest.tgz}"
tmp=$(mktemp -d /tmp/enc_sos_int8_pull.XXXXXX)
mkdir -p "$tmp/artifacts" "$tmp/dev_shm" "$tmp/logs"
cp -f /workspace/finetune/nemo_ft/encoder_sos_int8.log "$tmp/logs/" 2>/dev/null || true
cp -f /workspace/finetune/nemo_ft/artifacts/enc_sos_int8_* "$tmp/artifacts/" 2>/dev/null || true
cp -f /workspace/finetune/nemo_ft/artifacts/official_debug_enc_sos_int8_* "$tmp/artifacts/" 2>/dev/null || true
cp -f /workspace/finetune/nemo_ft/artifacts/nemotron_encoder_sos_int8_submission.zip.sha256 "$tmp/artifacts/" 2>/dev/null || true
cp -f /dev/shm/int8_devdiag.csv /dev/shm/int8_devdiag.json "$tmp/dev_shm/" 2>/dev/null || true
cp -f /dev/shm/fp32_devdiag.csv /dev/shm/fp32_devdiag.json "$tmp/dev_shm/" 2>/dev/null || true
cp -f /dev/shm/int8_devrep500.csv /dev/shm/int8_devrep500.json "$tmp/dev_shm/" 2>/dev/null || true
cp -f /dev/shm/fp32_devrep500.csv /dev/shm/fp32_devrep500.json "$tmp/dev_shm/" 2>/dev/null || true
tar -C "$tmp" -czf "$REMOTE_TGZ" .
tar -tzf "$REMOTE_TGZ"
rm -rf "$tmp"
REMOTE

scp -P "$PORT" -i "$KEY" $O root@"$HOST":"$REMOTE_TGZ" "$LOCAL_ART"/
echo "[pull] wrote $LOCAL_ART/$(basename "$REMOTE_TGZ")"

if [ "$STOP_POD" = "1" ]; then
  echo "[pull] stopping pod $ID"
  runpodctl pod stop "$ID" 2>&1 | grep -i desiredStatus || true
else
  echo "[pull] STOP_POD=$STOP_POD; leaving pod state unchanged"
fi
