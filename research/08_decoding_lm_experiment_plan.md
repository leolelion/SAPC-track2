# 08 — Decoding + LM experiment plan (no retraining, size-free, fits budget)

Goal: lower CER below A1 (**21.61% 2k Dev / 23.44% Test1**) by improving **decoding only** — no GPU
retraining, ~MB size cost. This is our biggest untapped lever (we currently decode **greedy, no LM**).
See [[07_streaming_asr_techniques_deepdive]]. Run on pod; prep here.

## Assets (on pod /workspace, persist)
- Model: `finetune/onnx/a1/{encoder,decoder,joiner}.onnx` (+ int8), `tokens.txt`, `bpe.model` (avg-5 chunk16).
- Ruler: `finetune/eval/dev_eval_2k.csv` + refs; scorer: sapc-nemotron `utils/evaluate.py` (sclite min-two-refs,
  torchmetrics → /opt/pylibs). Streaming/latency: `Dev_streaming.csv` + `local_decode.py` + `compute_latency.py`.
- Decode harness: `finetune/eval/decode.py` (sherpa-onnx OnlineRecognizer, greedy). sherpa env: rebuild
  `/tmp/sherpa_env` (pip sherpa-onnx>=1.13.3) — ephemeral, lost on stop.

## Experiments (EV order)
### E1 — Beam search (no LM)  [cheapest, immediate]
- `decoding_method="modified_beam_search"`, `max_active_paths ∈ {4, 8, 16}`.
- Compare CER vs greedy 21.61% on 2k ruler. Expect ~0.5–2% rel.
- Watch: beam ↑ compute → re-check TTFT/TTLT on Dev_streaming stays within Pareto (we have headroom; CER primary).

### E2 — n-gram (KenLM) shallow fusion  [the main win]
1. **Build LM**: extract `norm_text_without_disfluency` (and/or with) from `SAPC2/manifest/Train.csv` →
   train KenLM (lmplz -o 3 and -o 4) → binarize. (Optionally add external English text, but in-domain SAPC2
   text likely best for disfluency/vocabulary match.)
2. **FIRST verify sherpa-onnx ONLINE LM support**: does `OnlineRecognizer.from_transducer` accept a KenLM
   n-gram for online modified_beam_search, or only an ONNX **RNN-LM** (`lm=`, `lm_scale=`)?
   - If RNN-LM only → train a small LSTM LM on SAPC2 text via icefall `rnn_lm`, export ONNX. (icefall is on pod.)
   - If n-gram supported → use directly.
3. Sweep `lm_scale ∈ {0.1, 0.2, 0.3, 0.4, 0.5}` on a dev subset → pick best → full 2k. Expect 5–15% rel.

### E3 — Internal-LM subtraction (ILME / Density Ratio)  [stretch, +~14% over E2]
- Subtract the transducer's internal LM before adding the external LM. sherpa-onnx likely lacks native ILME;
  icefall supports LODR / n-gram-rescoring with ILM subtraction. Evaluate via icefall decode path if needed.
- Defer until E1/E2 land; biggest extra gain but most work.

### E4 — Second-pass n-best rescoring  [optional, low latency]
- Emit n-best from beam search → rescore with the LM (n-gram or small neural) → re-pick 1-best. 6–19% rel.

## Eval protocol
- Primary: 2k ruler CER (min-two-refs, sclite). Always compare to greedy 21.61% control.
- Latency: re-run `Dev_streaming.csv` → `compute_latency.py` for the chosen config (beam/LM add compute;
  confirm mean(TTFT,TTLT) still beats field; CER is primary but don't regress latency off the Pareto).
- Pick the config with best CER at acceptable latency. Re-export/repackage submission (recipe:
  [[submission-offline-packaging]]) — n-gram/RNN-LM adds only MBs, well under size limit.

## Why this first (vs Nemotron / bigger model)
- Size-free, no retraining, upgrades the LIVE #1 submission, proven 5–15%+ rel in literature, especially for
  atypical speech (AM uncertain → LM disambiguates). Highest EV per effort within the Track-2 budget.
