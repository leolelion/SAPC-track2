#!/usr/bin/env bash
# =====================================================================
# Parakeet Arm A -> ONNX submission: the whole paid-pod session, in order.
#
# Option 2 of investigations/parakeet_ship_runbook.md (Q's decision, 2026-07-29):
# export to ONNX and run under onnxruntime, because every submission that has ever
# scored for us did exactly that and no NeMo submission ever has.
#
# Every stage is gated: a stage that fails STOPS the run (set -e) instead of handing
# the next stage a broken artifact. Stages are selectable so a rerun does not repeat
# work already banked on the pod disk.
#
#   STAGES="wheels export parity int8 parity8 package offline gate_lat gate_acc" \
#   bash scripts/run_parakeet_onnx_pod.sh
#
# PRE-REGISTERED SHIP CRITERION (write it before the run; submit IFF all hold):
#   1. parity text stage: >=90% exact-match vs the NeMo wrapper on the parity set
#   2. Dev_diag severe CER  <= 20%   (banked Arm A via NeMo: 18.69%)
#   3. Dev_clean2k CER      <= 15%   (banked 13.51%)
#   4. mean(TTFT p50, TTLT p50) <= 420 ms (banked 356 ms)
#   5. projected Test wall-clock <= 15000 s with >= 30% margin
# Any miss -> do not submit, report the delta. Never submit to test a hypothesis.
# =====================================================================
set -euo pipefail

REPO=${REPO:-/workspace/SAPC-template}
CKPT=${CKPT:-/workspace/parakeet_ft/armA_full/ft_smoke_encoder_only.nemo}
DATA=${DATA:?set DATA to the SAP data root (must contain manifest/)}
VENV=${VENV:-/workspace/nemoenv}
WORK=${WORK:-/workspace/parakeet_onnx}
ART=${ART:-/workspace/artifacts/parakeet_onnx}
PARITY_WAVS=${PARITY_WAVS:-}          # glob of wavs for the fidelity gate (>= 20 utts)
STAGES=${STAGES:-"wheels export parity int8 parity8 package offline gate_lat gate_acc"}

FP32=$WORK/fp32
INT8=$WORK/int8
SRC=$REPO/track2_starting_kit/parakeet_onnx
mkdir -p "$ART" "$WORK"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

step() { echo -e "\n=========== $* ($(date -u +%H:%M:%S)) ==========="; }
has()  { [[ " $STAGES " == *" $1 "* ]]; }

# --------------------------------------------------------------- wheels
# Bundle onnxruntime so setup.sh can install with --no-index. cp311/manylinux to match
# the Codabench runtime (python 3.11, linux x86_64).
if has wheels; then
  step "wheels"
  rm -rf "$SRC/wheels"; mkdir -p "$SRC/wheels"
  pip download onnxruntime --only-binary=:all: --dest "$SRC/wheels" \
      --python-version 311 --implementation cp --platform manylinux_2_28_x86_64
  ls -la "$SRC/wheels"
fi

# --------------------------------------------------------------- export
if has export; then
  step "export fp32"
  python3 "$REPO/scripts/export_parakeet_onnx.py" \
    --nemo "$CKPT" --src-dir "$SRC" --out-dir "$FP32" \
    --work-dir "$WORK/raw" --att-context 70 1 2>&1 | tee "$ART/export.log"
  cp -r "$SRC/wheels" "$FP32/wheels"
fi

# --------------------------------------------------------------- parity (fp32)
# THE fidelity gate. Also decides encoder.{drop,trim}_policy, which cannot be reasoned
# out from the NeMo source alone (the exporter may or may not bake the trims in).
if has parity; then
  step "parity fp32 vs NeMo"
  [[ -n "$PARITY_WAVS" ]] || { echo "set PARITY_WAVS to a wav glob"; exit 1; }
  # shellcheck disable=SC2086
  python3 "$REPO/scripts/parity_parakeet_onnx.py" \
    --onnx-dir "$FP32" --nemo-dir "$REPO/track2_starting_kit/parakeet_realtime_ft" \
    --wavs $PARITY_WAVS --stages feat enc text \
    --out-json "$ART/parity_fp32.json" 2>&1 | tee "$ART/parity_fp32.log"

  BEST=$(python3 -c "import json;print(json.load(open('$ART/parity_fp32.json'))['stages']['enc']['best_policy'])")
  DROP=${BEST%%,*}; DROP=${DROP#drop=}
  TRIM=${BEST##*,}; TRIM=${TRIM#trim=}
  echo "[parity] winning policies: drop=$DROP trim=$TRIM -> writing into config.yaml"
  for CFG in "$SRC/config.yaml" "$FP32/config.yaml"; do
    sed -i "s/^  drop_policy: .*/  drop_policy: $DROP/" "$CFG"
    sed -i "s/^  trim_policy: .*/  trim_policy: $TRIM/" "$CFG"
  done
  grep -E "^  (drop|trim)_policy" "$FP32/config.yaml"
fi

# --------------------------------------------------------------- int8
# Dynamic int8 on the ENCODER only (fp32 decoder), exactly as the Nemotron submission
# that scored: ~3.6x smaller, faster on CPU, accuracy within noise there. The generic
# quantizer is reused as-is; it asserts the graph I/O names survive quantization.
if has int8; then
  step "int8 encoder"
  python3 "$REPO/scripts/quantize_nemotron_encoder.py" \
    --source-submission "$FP32" --output-submission "$INT8" \
    --work-dir "$WORK/quant" --out-json "$ART/quantize.json" 2>&1 | tee "$ART/quantize.log"
  du -sh "$FP32" "$INT8"
fi

# --------------------------------------------------------------- parity (int8)
# int8 changes numerics, so the text-level gate is re-run. Tensor tolerances are not
# meaningful post-quantization; transcript agreement is.
if has parity8; then
  step "parity int8 vs NeMo"
  # shellcheck disable=SC2086
  python3 "$REPO/scripts/parity_parakeet_onnx.py" \
    --onnx-dir "$INT8" --nemo-dir "$REPO/track2_starting_kit/parakeet_realtime_ft" \
    --wavs $PARITY_WAVS --stages text --min-exact-rate 0.85 \
    --out-json "$ART/parity_int8.json" 2>&1 | tee "$ART/parity_int8.log"
fi

# --------------------------------------------------------------- package
if has package; then
  step "package"
  python3 "$REPO/scripts/package_parakeet_onnx.py" \
    --submission-dir "$INT8" --zip "$ART/parakeet_armA_int8.zip" \
    --parity-json "$ART/parity_fp32.json" 2>&1 | tee "$ART/package.log"
fi

# --------------------------------------------------------------- offline validation
# Prove the EXTRACTED zip installs and runs with no network: fresh venv with
# --system-site-packages (numpy/torch present, onnxruntime absent) + dead proxies +
# HF_HUB_OFFLINE. If it transcribes under those conditions, it is offline-valid.
if has offline; then
  step "offline validation of the extracted zip"
  rm -rf "$WORK/extract" "$WORK/offlinevenv"
  mkdir -p "$WORK/extract"
  python3 -c "import zipfile;zipfile.ZipFile('$ART/parakeet_armA_int8.zip').extractall('$WORK/extract')"
  python3 -m venv --system-site-packages "$WORK/offlinevenv"
  # shellcheck disable=SC1091
  source "$WORK/offlinevenv/bin/activate"
  python3 -c "import onnxruntime" 2>/dev/null && { echo "venv already has ort — test is not proving anything"; exit 1; }
  env http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 HF_HUB_OFFLINE=1 \
      bash "$WORK/extract/setup.sh" 2>&1 | tee "$ART/offline_setup.log"
  env http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 HF_HUB_OFFLINE=1 \
      python3 "$REPO/track2_starting_kit/local_decode.py" \
      --submission-dir "$WORK/extract" \
      --manifest-csv "$DATA/manifest/Dev_smoke.csv" \
      --streaming-manifest-csv "$DATA/manifest/Dev_smoke_streaming.csv" \
      --streaming-interval 0 --data-root "$DATA" \
      --out-csv "$ART/offline_smoke.csv" --out-partial-json "$ART/offline_smoke.json" \
      2>&1 | tee "$ART/offline_decode.log"
  deactivate
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

# --------------------------------------------------------------- latency + threads
# The one number still guessed: threads. Default 1 protects the wall-clock budget, but
# if per-chunk compute at 1 thread exceeds the 100 ms chunk period the stream falls
# behind real time and the 356 ms corner — our whole reason to ship this — collapses.
# Pre-registered: pick the LOWEST thread count whose mean(TTFT,TTLT) <= 420 ms. If even
# threads=4 misses, the corner is thread-bound: STOP and report, do not raise blindly.
if has gate_lat; then
  step "thread sweep + latency (extracted zip)"
  for T in 1 2 4; do
    SAPC2_THREADS=$T python3 "$REPO/track2_starting_kit/local_decode.py" \
      --submission-dir "$WORK/extract" \
      --manifest-csv "$DATA/manifest/Dev_streaming.csv" \
      --streaming-manifest-csv "$DATA/manifest/Dev_streaming.csv" \
      --data-root "$DATA" \
      --out-csv "$ART/stream_t$T.csv" --out-partial-json "$ART/stream_t$T.json" \
      2>&1 | tee "$ART/stream_t$T.log"
    ( cd "$REPO" && ./evaluate.sh --start_stage 3 --stop_stage 3 \
        --partial-json "$ART/stream_t$T.json" \
        --manifest-csv "$DATA/manifest/Dev_streaming.csv" ) 2>&1 | tee "$ART/latency_t$T.log"
  done
fi

# --------------------------------------------------------------- accuracy gates
# House rule validate-against-real-harness: the organizers' exact pipeline, run from the
# EXTRACTED zip, on both slices. Hand-rolled decoders are exploration, never sign-off.
if has gate_acc; then
  step "accuracy gates on the extracted zip"
  for SPLIT in Dev_diag Dev_clean2k; do
    python3 "$REPO/track2_starting_kit/local_decode.py" \
      --submission-dir "$WORK/extract" \
      --manifest-csv "$DATA/manifest/$SPLIT.csv" \
      --streaming-manifest-csv "$DATA/manifest/${SPLIT}_streaming.csv" \
      --streaming-interval 0 --data-root "$DATA" \
      --out-csv "$ART/$SPLIT.predict.csv" --out-partial-json "$ART/$SPLIT.partial.json" \
      2>&1 | tee "$ART/decode_$SPLIT.log"
    ( cd "$REPO" && ./evaluate.sh --split "$SPLIT" --hyp-csv "$ART/$SPLIT.predict.csv" \
        --start_stage 0 --stop_stage 2 ) 2>&1 | tee "$ART/eval_$SPLIT.log"
  done
fi

step "DONE — copy $ART back to the Mac, then STOP THE POD"
echo "Artifacts:"; ls -la "$ART"
