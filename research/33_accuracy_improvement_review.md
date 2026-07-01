# 33 — ML-researcher review: where Nemotron accuracy stands and how to push it further (2026-06-28)

> Reviewer framing: treat the current int8 submission as v1 (shipped/validated). This doc audits the
> *training* process that produced it and ranks the next accuracy levers strictly by expected CER gain
> per GPU-hour, against the Track-2 constraints (streaming, CPU-only, latency-scored). Grounded in
> docs 01, 02, 06, 13, 14, 18–21, 28, 32. Numbers are directional with literature anchors — not forecasts
> (the repo's "projection not promise" rule holds).

## 1. Verified state (the v1 we just shipped)

- Model: Nemotron 0.6B cache-aware FastConformer-RNNT, **encoder-only finetune** (decoder/joint frozen).
- Train config (doc 20): all 309k utts / 683 h, cased targets, **pinned `att_context=[70,1]`**,
  SpecAugment, LR 1e-4 / 15% warmup, **4 epochs** + early-stop + top-5 ckpt avg.
- Official-scorer CER (doc 32, Dev-500, min-over-two-refs): **FP32 8.64% / int8 8.76%** (WER 12.77 / 13.07).
- Severe-enriched Dev_diag-425 (int8): **24.85%**, with the residual concentrated in
  **ALS 33.76% and Down Syndrome 35.06%**; Parkinson's already 5.97%.
- Latency (int8, pod): TTFT p50 1.075 s, TTLT p50 0.199 s.
- This is a large win over the A1 zipformer (Test1 23.44% CER). v1 is frontier-worthy on CER alone.

## 2. The binding constraint on accuracy right now

The errors are **not uniform** — they are concentrated in (a) the **severe tail** (ALS/DS, ~34% CER,
the recoverable-moderate empties) and (b) whatever was **left on the table by under-training**. Everything
below is ranked against those two facts. Levers that don't touch them (beam/LM, int4, three-graph split)
are explicitly out for accuracy.

## 3. The single biggest miss: the v1 model is UNDER-CONVERGED

Every training-side doc flags this, and the fixes were already written into the code but **never re-run**
because deployment/packaging/int8/unk-fix consumed the schedule. This is the highest-confidence, lowest-risk
CER gain available — the same recipe, executed properly:

| Issue (doc) | v1 reality | Fix (status) |
|---|---|---|
| Epochs (06, 20, 21) | **4 epochs**, loss not plateaued; SAPC1 winner used 20, icefall 10–20 | train **8–15 epochs** |
| Val/early-stop (21) | `limit_val_batches=30` + `shuffle=False` → same ~480 utts every epoch → biased ckpt pick | `limit_val_batches=1.0` (**fixed in code, not run**) |
| Ckpt averaging (21) | "top-5" over only 4 epochs averaged the under-trained epoch-1; also averaged integer buffers | average **converged-tail float tensors only** (**fixed in code, not run**) |
| Dataloader (20) | 53% GPU-bound, 13 h wall → epochs were "expensive", discouraged more | `pretokenize`/more workers → makes 8–15 epochs affordable |
| Wasted arm (20) | full-unfrozen trained in parallel and **lost** | drop it → ~half the cost |

**Verdict:** a converged encoder-only retrain is "free" accuracy — no new modeling risk, the code fixes
exist. **This is step 1.** Expectation: meaningful CER drop, weighted toward the severe tail where an
under-trained encoder still emits empties. (Direction high-confidence; magnitude unknown — measure it.)

## 4. Fold these THREE levers into the SAME retrain (not separate runs)

The retrain is one GPU session; bundle the cheap high-EV levers into it:

### 4a. Multi-lookahead context training (accuracy + restores the latency lever) — docs 19, 21
v1 **pinned `[70,1]`**, discarding NeMo's intended multi-lookahead. Train with `att_context_probs` over
`[[70,13],[70,6],[70,1],[70,0]]`. NeMo's own guidance: multi-lookahead is **≥ single-context accuracy** and
lets us pick the CER×TTFT operating point at *deploy* time instead of being locked. Also removes the latent
train/deploy context-mismatch risk. Cost: a config change + verifying `nemo_finetune.py` supports the list.

### 4b. Severity-aware oversampling (aimed straight at the ALS/DS residual) — docs 13, 14, 01
Residual CER is concentrated in ALS/DS. Literature is strong and specific:
severity-specific finetuning **~32% relative** (TORGO/UASpeech, Sapkota 2026); mixed-severity sampling
**56–61% relative**, and **high-severity data helps low-severity** (Geng 2305.10659). No test-time label is
needed — this is purely a **train-time sampling weight** (proxy severity by etiology + utterance error/empty
propensity, avoiding pure circularity per doc 14 §3). Low integration cost; folds into the dataloader.

### 4c. Verify + add 3-way speed perturbation (0.9/1.0/1.1) — docs 06, 01
The recipe (06) flags speed-perturb as "**Missing — high value**"; doc 20 mentions only SpecAugment.
**Verify whether it was in the v1 full run** (check `run_fullrun.sh` / `nemo_finetune.py` on the pod). If
absent, add it — dysarthric speaking-rate varies hugely; literature ~**7.9–9.3% relative WER** (UASpeech).
Near-zero integration cost via NeMo built-in.

> Anti-forgetting replay slice (`--use-mux` / typical-speech) is **lower priority**: v1 showed *no*
> forgetting (PD improved 11.2→4.8%). Keep a clean-English probe as a guard, but don't pay for replay yet.

## 5. Higher-ceiling, heavier — only if §3–4 leave the severe tail binding

### TTS-synthesized dysarthric augmentation (docs 13 §3, 01 §2)
Highest-ceiling **data** lever for ALS/DS specifically: multi-talker TTS + severity coefficient + pause
insertion cut CER to ~7.3% in one study (~18% relative CER reduction over real-only). BUT: heavy pipeline,
and at 683 h of *real* data we are past the low-data regime where synthetic pays most. **Defer** until a
converged + severity-weighted run shows the severe tail is still the binding constraint. High effort.

## 6. Explicitly OUT for accuracy on this track (don't spend here)

- **Beam search / LM fusion** (doc 21 #8, 01 §4): raises TTFT/compute for small CER on a *latency-scored CPU*
  track. Greedy is the latency-correct default. Avoid.
- **LLM post-ASR correction** (doc 01 §4, 13 §4): −14.5% WER offline but destroys TTLT / blows CPU budget.
  Out. (A *tiny* rule-based corrector is the only tolerable variant; low EV.)
- **int4 k-quant** (doc 02, 28-D): a size/speed play that is slightly *accuracy-negative* (Banfic int4 8.20%
  vs FP32 8.03% WER) and full-integer encoder paths *degraded* WER. Not an accuracy lever.
- **Three-graph split decoder/joiner** (doc 02): runtime/speed, not accuracy.

## 7. Recommended plan (ordered, EV-ranked)

1. **Ship v1 now.** The int8 zip (`597b8d95…`, Dev-500 8.76%) is validated and frontier-worthy. Submitting
   locks a Pareto point and satisfies "≥1 valid public submission," **de-risking the v2 gamble.** (Strategic;
   do this before spending GPU on v2.)
2. **Pre-retrain verification (no GPU yet):** restart pod only when ready; pull the v1 **loss curve +
   per-epoch internal-dev CER** to confirm under-convergence empirically (not just by doc), and confirm which
   aug flags v1 actually used. This is the Chesterton's-fence check before committing GPU.
3. **v2 converged retrain (one run):** encoder-only, drop full-unfrozen, **8–15 epochs**, fixed full-val +
   clean-tail float-only averaging, **multi-lookahead `att_context_probs`**, **severity-aware oversampling**,
   **+ speed-perturb if missing**, dataloader fixed. Set a hard GPU-budget cap + a smoke gate first.
4. **Faithful eval gate (mandatory, doc 32 pattern):** export → int8 → `local_decode.py` + official
   `evaluate.sh` on Dev-500 **and** Dev_diag-425 with speaker-block bootstrap CI; require official rc=0 and a
   CER improvement with non-overlapping/credible CI vs v1 **before** any v2 submission. Never submit on a proxy.
5. **Re-decide TTS augmentation** only if §3–4 leave ALS/DS as the binding constraint.

## 8. What would change my ranking (epistemic honesty)

- If the v1 loss curve is actually **flat by epoch 4**, step 3's "convergence" gain shrinks and severity-aware
  sampling + TTS become the lead levers. **Verify before assuming** (step 2).
- If multi-lookahead measurably *costs* CER at the deploy context (possible — it trades capacity across
  contexts), keep it as a deploy-flexibility tool but pick the single best context for the submitted point.
- All magnitude expectations are literature transfer across regime changes (offline≥1.1B → streaming 0.6B,
  WER → CER). Treat every gain as a hypothesis the faithful harness must confirm.

## 9. One-line bottom line

v1 is shipped-quality; the **highest-EV accuracy work is a single converged encoder-only retrain that folds in
multi-lookahead + severity-aware sampling + speed-perturb** — the fixes are already coded, it directly targets
the under-training and the ALS/DS tail, and it costs one gated GPU session. TTS augmentation is the
higher-ceiling follow-up; beam/LM/int4/LLM-GER are off-track for accuracy here.
