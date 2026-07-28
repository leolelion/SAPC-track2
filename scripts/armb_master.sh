#!/usr/bin/env bash
# Overnight master chain for D1 (Arm B ladder). Runs unattended; every decision it makes is
# pre-registered here, not improvised at 3am.
#
#   rung l0    = joint_unfreeze at the base checkpoint's INHERITED fastemit_lambda=0.03
#                (verified on the pod: RNNTLossNumba.fastemit_lambda=0.03 — the runbook's
#                "lambda=0 control" does not exist; l0 is nonetheless the correct control for
#                the UNFREEZE, since Arm A trained under the same 0.03).
#   rung fe06  = joint_unfreeze + fastemit_lambda=0.06 (2x base). Direction is UP, not the
#                pre-registered {0.003,0.005,0.01} — those would LOWER emission pressure below
#                the base. Justified by the measured error profile: deletions 24.76% vs
#                insertions 1.38% on val, i.e. the model is under-emitting with huge headroom
#                before FastEmit's over-emission failure mode can bite.
#
# Ladder rule (runbook): stop at the first rung that clears GATE-TRAIN. So fe06 runs ONLY if
# l0 fails to clear. GATE-TRAIN is a PROXY (punct-stripped single-ref) and never authorizes a
# ship claim — GATE-SHIP (official two-ref sclite on Dev_diag) is the only ship authority.
set -uo pipefail
LOG=/workspace/armb_master.log
exec >> "$LOG" 2>&1
echo "=== master chain start $(date -u +%FT%TZ) ==="

wait_for_nemo () {  # $1 = tag
  local f=/workspace/parakeet_ft/armB_$1/ft_smoke_joint_unfreeze.nemo
  local waited=0
  while [ ! -f "$f" ]; do
    if ! pgrep -f nemo_finetune_v2.py > /dev/null; then
      sleep 60   # trainer gone: allow for the post-fit averaging/save to land
      [ -f "$f" ] && break
      echo "!! trainer for $1 exited without producing $f — stopping chain"
      return 1
    fi
    sleep 120; waited=$((waited+120))
  done
  echo "[$1] .nemo present after ${waited}s"
  return 0
}

gate_train_passed () {  # $1 = tag ; echoes PASS/FAIL from the post-chain output
  grep -a "GATE-TRAIN cer_better" /workspace/parakeet_ft/gate_armB_$1/gatetrain.txt 2>/dev/null \
    | grep -q "PASS= True"
}

run_post () {  # $1 = tag
  bash /workspace/armb_post.sh "$1" > /workspace/parakeet_ft/gate_armB_$1_post.log 2>&1
  mkdir -p /workspace/parakeet_ft/gate_armB_$1
  grep -a "GATE-TRAIN" /workspace/parakeet_ft/gate_armB_$1_post.log \
    > /workspace/parakeet_ft/gate_armB_$1/gatetrain.txt 2>/dev/null
}

# ---------------- rung 1: l0 ----------------
wait_for_nemo l0 || exit 1
echo "--- [l0] post-chain $(date -u +%FT%TZ) ---"
run_post l0

if gate_train_passed l0; then
  echo "=== [l0] CLEARED GATE-TRAIN -> ladder stops here (runbook rule). No fe06. ==="
  echo "MASTER_CHAIN_DONE $(date -u +%FT%TZ)"
  exit 0
fi

echo "=== [l0] did not clear GATE-TRAIN -> launching rung fe06 (lambda=0.06) $(date -u +%FT%TZ) ==="
bash /workspace/run_armb_rung.sh fe06 0.06
wait_for_nemo fe06 || exit 1
echo "--- [fe06] post-chain $(date -u +%FT%TZ) ---"
run_post fe06
echo "MASTER_CHAIN_DONE $(date -u +%FT%TZ)"
