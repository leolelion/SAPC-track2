# 21 — Flow audit + ranked next steps (independent agent + our synthesis, 2026-06-26)

## Audit of the full finetune flow — real issues found
- **L4 baseline-confound correction (47.5→43.4)**: strongly endorsed — best epistemic hygiene in the flow.
- **L5 target-text coherence gap**: doc 19 said "keep normalized", `run_fullrun.sh` used `--target text` (CASED).
  The reversal (per the config-review's "use NVIDIA-native cased") was real but undocumented. Likely CER-neutral
  (scorer normalizes) BUT cased emits punctuation tokens → small TTFT/emission cost on a LATENCY-scored track.
  Resolve empirically in the deployment loop (measure TTFT of the cased model).
- **L6 single-context [70,1] pin**: internally consistent but a debatable design call — NeMo multi-lookahead
  training is ≥ single-context accuracy AND preserves the latency lever. We threw the lever away. Next finetune
  should be multi-lookahead (`att_context_probs`).
- **L7 validation noise**: `limit_val_batches=30` + `shuffle=False` = same first ~480 utts every epoch → noisy/
  biased early-stop + ckpt ranking. FIXED → `limit_val_batches=1.0` (full internal-dev) for next run.
- **L8 checkpoint averaging**: (a) only 4 epochs → "top-5" averaged ALL incl. under-trained epoch-1 (can hurt;
  averaging benefits come from the CONVERGED tail). (b) bug: averaged integer buffers (num_batches_tracked).
  FIXED → average only float param tensors. Convergence/selection is the weakest link in the v1 run.
- **Encoder-only "wins" verdict**: directionally trustworthy (replicated 2 scales + theory + stronger
  internal-dev gap 8.01 vs 11.06 = full-unfrozen overfitting signature) BUT the 1.6-pt Dev_diag gap on 425
  speaker-correlated utts has NO CI, and full-unfrozen had FEWER empties (latency-relevant). Treat head-to-head
  as UNRESOLVED until speaker-level (blockwise) bootstrap CI + empty-rate latency consequence on the real harness.

## RANKED NEXT STEPS (measure before training more)
1. **[DOING] Deployment-CER loop** — build NeMo-ONNX streaming model.py (adapt to encoder-model.onnx +
   decoder_joint-model.onnx I/O) → validate on ZERO-SHOT export first → run encoder-only winner through
   `local_decode.py` on Dev → TRUE streaming CPU CER **+ TTFT/TTLT latency** + int8(encoder)-vs-fp32 delta.
   Decision-critical; everything else depends on the real metric. (Open issues a/c/d.)
2. **Representative-Dev eval + speaker-level bootstrap CI** (cheap) — converts "beats zipformer 23.44%?" from
   projection to measurement; resolves enc-vs-full ranking. Run with #1.
3. **int8 recipe**: quantize ENCODER int8, keep decoder+joint FP32 (lit: enc int8 ≈ fp32 WER) — measure on dysarthric.
4. **Fix-and-converge retrain**: full-dev val + clean tail averaging + 8–12 epochs (drop full-unfrozen arm) +
   **multi-lookahead** training (restore latency lever). Fix dataloader (pretokenize/workers; was 53% GPU-bound).
5. **Severity-aware sampling** for residual ALS/DS empties (lit: mixed-severity → 56–61% rel gain; high-sev helps low-sev).
6. **TTS dysarthric augmentation** — high ceiling (FastConformer+FastSpeech2 cut CER to ~7.3%) but heavy; defer.
7. **LoRA** — low priority (no forgetting observed, so its main benefit is moot).
8. **Beam / LM fusion — LAST / avoid.** CORRECTION to earlier ranking: for a latency-scored CPU-budget track,
   beam+LM RAISES TTFT/compute for small CER; greedy is the latency-correct default. (I had over-ranked this.)

## Status answers
- Finetuning = a REAL full v1 model exists (encoder-only winner), but likely UNDER-converged (4 epochs, noisy
  selection) → a v2 (converged + multi-lookahead) is expected after #1 measures the real metric.
- **Latency = UNMEASURED.** All CERs are NeMo transcribe() (offline-ish, no TTFT/TTLT). The deployment loop is
  what produces the first latency numbers. Model trained at [70,1] (160ms) + FastEmit → should be low, but unproven.
