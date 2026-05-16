# Plan: baseline-vs-finetuned benchmark on SAPC2 Dev

## Phase 1 — Environment

Runpod gpu: Ask me for the ssh information. You can edit files in this repo, push and pull them on the pod.

One conda env, NeMo + ONNX Runtime + ffmpeg/libsndfile. Two paths:

- **Path A (preferred for first benchmark)**: NeMo PyTorch on whatever hardware you have. CPU works, GPU is much faster — if you have any GPU at all, use it for this phase. The point of phase 1 is *accuracy comparison*, not latency.
- **Path B (for phase 5 only)**: ONNX Runtime CPU with the danielbodart int8-dynamic export. Skip until accuracy numbers look right.

```bash
conda create -n sapc2 python=3.10 -y
conda activate sapc2
pip install Cython packaging
pip install "nemo_toolkit[asr]"
pip install pandas tqdm soundfile librosa
# verify
python -c "import nemo.collections.asr as nemo_asr; print(nemo_asr.__file__)"
```

## Phase 2 — Inference harness (write once, reuse for all 3 models)

Write a single script `run_nemo_inference.py` that takes any `.nemo` checkpoint and a manifest CSV and produces a hypothesis CSV in the format `evaluate.sh` expects. Pseudocode:

```python
# args: --model (HF repo id or local .nemo path), --manifest, --audio-root, --out
# 1. Load model:
#    - if arg ends with .nemo: ASRModel.restore_from(path)
#    - else: ASRModel.from_pretrained(repo_id)
# 2. Read manifest CSV with pandas
# 3. Build list of audio paths from manifest['id'] + audio_root
# 4. Call model.transcribe(audio_paths, batch_size=16)
#    - returns list of Hypothesis objects; .text is what you want
# 5. Write CSV with columns ['id', 'raw_hypos']
# 6. Print WER preview on first 50 rows against manifest.norm_text_without_disfluency for fast feedback
```

Two specific things to get right:

- **Long-audio handling**. SAP utterances can be tens of seconds. NeMo's `.transcribe()` handles this, but if you OOM or get truncation, switch to batch_size=1 or chunk the audio first. Don't fight this — use whichever path works.
- **Output text normalization**. *Don't* normalize in the inference script. The eval script normalizes both refs and hyps with its own normalizer — if you pre-normalize you risk double-stripping things and getting fake-good numbers. Hand it raw model output.

Smoke test the harness on **just 5 manifest rows** with the baseline model before running anything full-size. Check that `id` values in the output CSV exactly match the manifest IDs (this trips people up because of stem-vs-full-path issues).

## Phase 3 — Run the three models on Dev

Same harness, three checkpoints:

| Run | Model | What it tells you |
|---|---|---|
| **R1** | `nvidia/nemotron-speech-streaming-en-0.6b` | Baseline streaming WER on SAP Dev — your starting point |
| **R2** | `Phaedrus33/nemotron-speech-streaming-children-v17` | Streaming finetune on the right base model — the closest existing analog to what your final system would look like |

For R1 and R2, use the default `att_context_size` (which is `[70,13]` = 1120ms chunks) for this first benchmark. Don't sweep latency yet — you want accuracy in the same configuration the leaderboard reports.

```bash
python run_nemo_inference.py \
  --model nvidia/nemotron-speech-streaming-en-0.6b \
  --manifest $DATA_ROOT/manifest/Dev.csv \
  --audio-root $DATA_ROOT/raw/Dev \
  --out hyps/R1_baseline_nemotron.csv
```

For each run, log: wall-clock time, batch size used, GPU memory peak (if GPU), number of utterances, average audio duration. You'll want this when you're sizing the actual fine-tuning run.

## Phase 4 — Evaluate and tabulate

```bash
./evaluate.sh --start_stage 1 --stop_stage 2 --split Dev \
  --hyp-csv hyps/R1_baseline_nemotron.csv
mv $DATA_ROOT/eval/metrics.Dev.json metrics/R1_metrics.json

# repeat for R2, R3
```

Then a tiny table-builder script that reads `metrics/R*_metrics.json` and prints:

```
Model                                     CER%    WER%
R1 baseline streaming nemotron            ?       ?
R2 phaedrus streaming nemotron finetune   ?       ?
```

For reference, Takahashi 2025 went from Whisper-large-v2 baseline 17.82 → 8.11 WER on SAP Test2 after a heavy Parakeet-TDT-1.1B fine-tune. The 2025 challenge used WER on the SAP-2024-04-30 dataset, SAPC2 has new and larger data and uses CER as the primary metric, so don't expect identical numbers — but as a sanity check: **R1 should be comfortably worse than R2**. If any of those orderings are wrong, something is broken.

## Phase 5 — Decide what's next based on the table

Three possible outcomes, each with a different next action:

- **R1 ≈ R2** (streaming finetune barely helps): suggests Phaedrus's children's-speech finetune doesn't transfer well to dysarthria. You'll need your own SAP finetune; the children's checkpoint is not a useful warm-start. Move to a fine-tuning recipe based on Takahashi 2025.
- **R3 noticeably beats R2**: the expected case. Streaming costs you accuracy, and atypical-speech finetuning helps. Quantify the streaming gap (R2−R3) — that's your "ceiling cost" for choosing streaming over offline. Then plan a SAP-specific finetune.

In all three cases, **Phase 5 ends with a decision to fine-tune on SAP**. The benchmark doesn't change *whether* you finetune; it tells you *what to expect* and *what your warm-start should be*.

## Phase 6 (deferred) — CPU + latency

Don't touch this until phases 0–5 are done and the table makes sense. When you're ready:

1. Pull `danielbodart/nemotron-speech-600m-onnx` (just `int8-dynamic/`, `shared/`, `config.json`)
2. Write a custom ONNX inference loop following the danielbodart `config.json` spec (the cache tensor shapes, mel filterbank, RNN-T greedy loop are all documented there)
3. Verify the ONNX int8 output matches NeMo PyTorch output on a handful of utterances within ~0.5 WER points
4. Then measure TTFT and TTLT on the SAPC2 streaming manifest using the competition's `compute_latency.py`

That's a multi-day task on its own, which is exactly why it goes after the accuracy benchmark — you don't want to find out your ONNX path is broken after you've already invested in fine-tuning.
