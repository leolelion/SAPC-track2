# Finetuning Streaming Zipformer on SAPC Data

Finetune the icefall streaming Zipformer-Transducer (70M params, LibriSpeech-pretrained) on SAPC dysarthric speech, then export to ONNX for the `sherpa_zipformer/` submission wrapper.

---

## Prerequisites

- RunPod GPU with CUDA 12.4 and PyTorch 2.5.0 (the SAPC2 runtime image `xiuwenz2/sapc2-runtime:latest`)
- SAPC dataset available (e.g. at `/workspace/data`)
- This repo cloned on the pod

---

## Step-by-step

### 1. SSH to RunPod and pull latest code

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<host> -p <port>
cd /workspace
git clone <your-repo> sapc && cd sapc
# or if already cloned:
git pull
```

### 2. Install environment

Installs k2, kaldifeat, icefall, lhotse, and downloads the pretrained checkpoint.

```bash
bash finetuning/setup_finetune.sh
```

This takes ~5–10 minutes. It downloads `epoch-30.pt` (~280 MB) from HuggingFace.

### 3. Prepare data

Converts SAPC manifest CSVs to lhotse CutSets.

```bash
python3 finetuning/prepare_sapc_lhotse.py \
    --data-root /workspace/data \
    --output-dir finetuning/data
```

Output:
- `finetuning/data/cuts_train.jsonl.gz`
- `finetuning/data/cuts_dev.jsonl.gz`

### 4. Finetune

```bash
python3 finetuning/finetune.py --config finetuning/finetune_config.yaml
```

**To resume a crashed run:**
```bash
python3 finetuning/finetune.py --config finetuning/finetune_config.yaml --resume
```

**Override hyperparameters on the CLI:**
```bash
python3 finetuning/finetune.py --config finetuning/finetune_config.yaml \
    --base-lr 0.0003 --num-epochs 30
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir finetuning/exp/tensorboard --port 6006
```

Checkpoints are saved to `finetuning/exp/epoch-*.pt` every 2 epochs.

### 5. Export to ONNX

Average the last 5 checkpoints and export to ONNX:

```bash
bash finetuning/export_onnx.sh --epoch 20 --avg 5
```

This writes ONNX files directly to `track2_starting_kit/sherpa_zipformer/weights/standard/`.

### 6. Validate

```bash
cd track2_starting_kit
python3 local_decode.py \
    --submission-dir ./sherpa_zipformer \
    --manifest-csv /workspace/data/manifest/Dev.csv \
    --streaming-manifest-csv /workspace/data/manifest/Dev_streaming.csv \
    --data-root /workspace/data \
    --out-csv ./Dev.predict.csv \
    --out-partial-json ./Dev.partial_results.json
```

### 7. Evaluate

```bash
cd ..
./evaluate.sh --start_stage 2 --stop_stage 2 \
    --split Dev --hyp-csv track2_starting_kit/Dev.predict.csv
```

Baseline (zero-shot): ~34.59% CER. Target after finetuning: ~15–25% CER.

### 8. Submit

```bash
cd track2_starting_kit/sherpa_zipformer
zip -r submission.zip model.py config.yaml setup.sh weights/
# Upload submission.zip to Codabench
```

---

## Key files

| File | Purpose |
|---|---|
| `setup_finetune.sh` | Install deps + download pretrained checkpoint |
| `prepare_sapc_lhotse.py` | CSV → lhotse CutSets |
| `finetune.py` | Main training script |
| `finetune_config.yaml` | All hyperparameters |
| `export_onnx.sh` | Export `.pt` → ONNX + copy to sherpa weights |

---

## Hyperparameter notes

Key settings in `finetune_config.yaml`:

- **`base_lr: 0.0005`** — much lower than pretraining (~0.045) to avoid destroying pretrained features
- **`max_duration: 300`** — total seconds of audio per batch (lhotse dynamic bucketing)
- **`accumulate_grad_steps: 4`** — effective batch = 4× `max_duration`
- **`num_epochs: 20`** — SAPC is much smaller than LibriSpeech; more epochs needed

If Dev CER plateaus or rises, try:
- Reduce `base_lr` to `0.0001`
- Increase `weight_decay` to `1e-5`
- Reduce `num_epochs` (early stopping at best-valid-loss checkpoint)

---

## Troubleshooting

**k2 import fails:**
```bash
python3 -c "import k2; print(k2.__version__)"
# If missing, re-run: bash finetuning/setup_finetune.sh
```

**CUDA OOM:**
Reduce `max_duration` in `finetune_config.yaml` (e.g. to `200` or `150`).

**ONNX export fails:**
Ensure the architecture args match exactly: `--causal True --chunk-size 16 --left-context-frames 128`.

**sherpa-onnx not installed on RunPod:**
```bash
pip install sherpa-onnx
```
The export script skips the sherpa verification step if it's not installed.
