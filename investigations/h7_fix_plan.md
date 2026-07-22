# H7 Fix Plan — the Nemotron offline→streaming collapse

> **⚠️ RESOLVED 2026-07-22 — see `research/45`. The "our export bug" hypothesis below is FALSIFIED.**
> The finetuned v1/PEFT checkpoint was LOST (pod `3dwiczo41jeg1y` + local), so the experiment ran on the
> BASE Nemotron as a 3-arm A/B/C on Dev_diag @ [70,1], same proxy scorer: NeMo `transcribe()` **43.4%** ≈
> NeMo-native cache-aware streaming **43.8%** ≈ **our** ONNX export→`local_decode.py` **43.4%** (0.4-pt
> spread). ⇒ the streaming pipeline is FAITHFUL — *neither* "intrinsic chunked-cache loss" (H7 mech-3)
> *nor* "our export/cache-stepping bug" (H7 mech-2) holds. The historical finetuned 18→28 gap was
> therefore weights/training-specific (PEFT not surviving streaming cache-norm → Arm-3 re-finetune, no
> cheap fix) OR a confounded comparison (the 18.46% eval set was never documented); cannot separate —
> checkpoint gone. **Deflates, not revives, Nemotron → ship beam-4.** Arms 1–3 below are kept for provenance.

> **Self-contained handoff.** Assumes the reader has ZERO context from the conversation that
> produced it (2026-07-14). Everything needed to act is here or linked. Follow the global
> defensive protocol (`/Users/o/Downloads/CLAUDE.md`) and repo rules (`SAPC-template/CLAUDE.md`):
> prediction-before-action, stop-on-failure, and **never submit on a proxy** — ship only on the
> organizers' real `local_decode.py` + scorer (`validate-against-real-harness`).

---

## The issue in one paragraph
A SAP-finetuned NVIDIA Nemotron ASR RNNT (0.6B, `nemotron-speech-streaming-en-0.6b`, cache-aware
*pretrained*) transcribes dysarthric `Dev_diag` at **~18–24% CER via NeMo `transcribe()`** but
collapses to **28.19% through the faithful cache-aware chunked-streaming export** (F8 in
`investigations/nemotron_vs_zipformer.md`; primary source `research/41`). The challenge scores the
streaming path, so the collapse is what makes Nemotron lose to the finetuned zipformer. Fixing (or
ruling out a fix for) this ~10-point gap is the goal.

## The corrected understanding (READ THIS — it reorders the suspects)
Earlier framing called 18.46% "offline / full context." **Wrong.** Both numbers run at roughly the
**same limited context `[70,1]`** (70 left, **1** right). They differ only in HOW that context is
computed:
- **`transcribe()` (18–24%)** = ONE forward over the whole utterance, limited context via an
  **attention mask**. "Context-faithful but NOT chunk-boundary-faithful" (research/19:12). A PROXY.
- **Streaming export (28.19%)** = utterance **chopped into 100 ms chunks**, limited context via
  **caches** (conv left-context + attention KV) stitched across seams. The FAITHFUL deployment path.

So the ~10 points are **masked-single-pass vs chunked-with-caches**, NOT less context. This
downgrades "fundamental right-context loss" and elevates two fixable suspects. **Confirmed own-goal:**
we were "mixing **[70,1]-train with [70,6]-deploy**" (research/19:32) — a context-regime mismatch.

### Ranked hypotheses for the 10 points
1. **Train/deploy config mismatch (ours to fix, cheapest).** `[70,1]`-train vs `[70,6]`-deploy;
   possibly a finetune that de-adapted the base model's cache-aware behavior. No GPU to test.
2. **Chunk/cache computation discrepancy in OUR export (the "export bug" bucket).** Conv cache init,
   seam stitching, per-chunk normalization diverging from the masked path. An independent
   implementation (audio.cpp) can isolate this.
3. **Fundamental right-context info-loss.** WEAK — the proxy already runs at right-context = 1.

---

## Prerequisites to start (none of this runs on the dev mac)
- A **Linux x86_64 CPU host or the Docker image** `xiuwenz2/sapc2-runtime:latest` (baseline
  Linux-only wheels: k2/kaldifeat cp311 manylinux). GPU host ONLY for Arm 3 (finetuning).
- **SAP data** + manifests: `$DATA_ROOT/manifest/Dev.csv`, `Dev_streaming.csv`, and the
  severe-enriched **`Dev_diag`** split used by all prior numbers (speaker-disjoint).
- The **v1 finetuned Nemotron checkpoint** (best artifact; locate via `research/20`, `research/24`).
- For Arm 2: the **audio.cpp** repo (`https://github.com/0xShug0/audio.cpp`, Apache-2.0) + its
  `audiocpp_gguf` converter and CPU build (`scripts/build_linux.sh`). See
  `investigations/audio-cpp-fit.md` for the converter/streaming-session findings.

---

## The plan — three arms, cheapest first. STOP after each arm and report before the next.

### ARM 1 — Fix the config mismatch — ❌ FALSIFIED LOCALLY 2026-07-14 (do NOT run; no pod)
**Hypothesis 1 (train/deploy att_context mismatch) is DEAD for the 28.19% artifact.** Verified from
our own export/deploy scripts on the dev mac — train == eval == export == deploy == **`[70,1]`**:
- `scripts/export_deploy.sh:2,17` — export winner **at [70,1]**, `set_default_att_context_size([70,1])`.
- `scripts/run_v2_export_eval.sh:20` — "Export v2 .nemo → ONNX **at [70,1]**".
- `scripts/build_deploy_sub.sh:2` — "build **[70,1]** submission dirs → run the REAL local_decode.py".
- `scripts/nemo_finetune.py:18` — `--train-ctx` = "SINGLE pinned deploy context (train+eval+export
  same)"; the 28.19% run trained at `[70,1]` (research/41:13).
- The `[70,6]` in research/19:32 was the OLD **zero-shot** ONNX-harness baseline (research/18:25),
  NOT the finetuned deploy. The mismatch had already been resolved to `[70,1]` everywhere.

⇒ Re-running "at matched context" would just reproduce ~28% (it was already matched). **No pod, no
spend on Arm 1.** Mechanism 3 (context-loss) is also weak (both `[70,1]`). **Mechanism 2
(chunked-cache computation vs masked single-pass) is now the sole live explanation** for the ~10-pt
gap — note the concrete `CHUNK_NEW 56→16` cache-stepping knob in `build_deploy_sub.sh:21`.
**Next test = Arm 2.**

DOING / EXPECT / steps:
1. Confirm the att_context v1 was **trained** at (grep the finetune config; research says `[70,1]`).
2. Re-export v1 to ONNX cache-aware with `model.set_export_config({'cache_support':'True'})`
   (mandatory or cache I/O is missing — research/19:18) at the **matching** deploy context.
3. Run the faithful gate on `Dev_diag`:
   ```bash
   cd track2_starting_kit
   python3 local_decode.py --submission-dir ./<nemotron_matched_ctx> \
     --manifest-csv $DATA_ROOT/manifest/Dev_diag.csv \
     --streaming-manifest-csv $DATA_ROOT/manifest/Dev_diag_streaming.csv \
     --data-root $DATA_ROOT --out-csv ./nemo.predict.csv \
     --out-partial-json ./nemo.partial.json
   cd .. && ./evaluate.sh --split Dev_diag --hyp-csv track2_starting_kit/nemo.predict.csv \
     --start_stage 0 --stop_stage 2
   ```
**Decision rule (write it before running):**
- Matched-context CER **drops toward ~24%** → the mismatch was a real chunk of the gap; the fix is
  a config change, cheaply. Proceed to see how much residual remains → Arm 2.
- Matched-context CER **stays ~28%** → mismatch was NOT the driver; the gap is in the chunk/cache
  computation or the weights → Arm 2 to localize.

**Cost:** minutes–hours of CPU, zero GPU. **Stop-on-failure:** if export or `local_decode.py`
errors, STOP and report the exact error (do not silently retry).

---

### ARM 2 — audio.cpp independent-implementation diagnostic (isolates hypothesis 2)
**Full spec lives in `investigations/audio-cpp-fit.md` ("Cheap diagnostic experiment spec").**
Summary: convert v1 weights → GGUF, run audio.cpp streaming at **matched** `lookahead_tokens`, score
on `Dev_diag` with OUR scorer (not audio.cpp's).

Critical control: **audio.cpp `lookahead_tokens` MUST match the context Arm 1 deployed at**, or the
comparison is void (that is the [70,1]/[70,6] mistake again).

**Decision rule:**
- audio.cpp CER **≈ Arm-1 faithful (28%-ish)** → the collapse is intrinsic to correct chunked-cache
  streaming at this context → **audio.cpp does NOT rescue accuracy**; the only real fix is Arm 3.
- audio.cpp CER **≈ the `transcribe` proxy (18–22%)** → the loss was in OUR ONNX export, not the
  regime → **audio.cpp rescues the Nemotron path**; escalate to the real-time session rewire (its
  streaming session currently buffers audio and only emits at `finish_stream` — see
  audio-cpp-fit.md FACTS) and then **certify on the real `local_decode.py` + official scorer**.

**Cost gates (audio-cpp-fit.md):** (a) GGUF convert must succeed (name-map shim if loader rejects
keys); (b) CPU build + `tests/nemotron_asr` warm bench must transcribe correctly BEFORE trusting CER;
(c) per-100 ms-chunk CPU encode latency must be well below real-time on the 24-core worker
(`eval-worker-cpu-confirmed`) or the latency case is dead regardless of CER. STOP at any failed gate.

---

### ARM 3 — Cache-aware finetuning in NeMo (the real training fix; GPU; NEEDS EXPLICIT Q APPROVAL)
Only if Arms 1–2 show the loss is in the **weights/regime**, not our export/config. Hypothesis: v1
was finetuned in a context regime that de-adapted the base model's cache-aware streaming behavior.
Fix = finetune **in** the cache-aware limited-context regime so the weights match the deploy budget:
- Train multi-lookahead `[[70,13],[70,6],[70,1],[70,0]]` with `att_context_probs` (research/19:30,
  NeMo-intended — preserves the latency lever) OR fix ONE deploy context and train+deploy at exactly it.
- Add forgetting insurance (replay slice + clean-English probe), lower peak LR (~1e-4, ~10–20%
  warmup), wire checkpoint averaging (research/19:33-35).
- Gate on `Dev_diag` via the FAITHFUL harness at the matched context. Ship IFF real-harness CER
  beats the banked zipformer path.

**This is a fresh, explicitly-approved GPU bet** (roadblock memory + research/41 recommendation).
Do the full cost-gate ritual in `SAPC-template/CLAUDE.md` before any `runpodctl pod start`.

---

## Decision tree (updated 2026-07-14 after Arm 1 falsified)
0. ~~Arm 1 matched-context re-run~~ — **FALSIFIED locally**; train=deploy=`[70,1]` already. Skip.
1. **Arm 2 is now the first test:** audio.cpp GGUF stream at matched `[70,1]` → is the chunked-cache
   collapse intrinsic (~28%) or specific to OUR ONNX cache-stepping (~20%)?
2. If ~20% (our export) → fix ONNX cache stepping / adopt audio.cpp, then certify on the real harness.
3. If ~28% (intrinsic to chunked-cache @ [70,1]) → only Arm 3 (cache-aware finetune, approval-gated
   GPU) or a different operating point can help; audio.cpp does not rescue CER.
4. **Nothing here is a submission.** Shipping always requires `local_decode.py` + official scorer.

## EV reassessment 2026-07-14 (after Arm 1 falsified) — VERDICT: HOLD Nemotron, ship zipformer
Arm 1's falsification removed the only *cheap* rescue; what remains is low-probability and expensive.
For Nemotron to beat the zipformer on **Test1** (the only metric that counts), ALL must hold:
1. The 18→28 gap must be a **fixable bug, not proxy inflation** — but `transcribe()` is a known
   optimistic proxy (CLAUDE.md precedent: proxy 24.96% vs real 51.52%), and at the SAME `[70,1]`
   context masked-vs-cached should see similar left-context, so ~30% odds it's a real bug.
2. **Dev→Test transfer wall stands (F9):** Nemotron slope +18 vs zipformer +1.8. Recovered Dev
   accuracy need not transfer — int8 `_t1` already went Dev_diag→Test1 27.97%.
3. **CPU streaming latency at 0.6B unverified** (track is latency-scored, CPU-only).
4. Then packaging + real-harness certification (correlated-proxy-blind-spot risk).
Compounded P(Nemotron ships AND beats zipformer) ≈ **10–15%**, at multiple pod sessions.
The audio.cpp find only unblocks **deployment**; the binding constraint is **accuracy-under-transfer**,
which nothing here moves. **⇒ Hold Nemotron. Ship gated beam-4 zipformer (~21%).**
Sole exception: if permanently retiring Nemotron *with proof* is wanted, run **Arm 2a once**
(NeMo-native streaming @ [70,1], no build/conversion) to settle proxy-inflation vs bug — for
closure only, not because success would likely change the ship decision.

## Evidence-backed default if no one picks this up
Per the roadblock memory and research/39–41: **STOP the Nemotron rescue, bank A1 (Test1 23.44%),
ship the gated beam-4 zipformer (~21% expected).** This plan is the ONLY thing that would reopen
Nemotron, and only Arm 1 is cheap enough to run speculatively.

## Files / references
- `investigations/audio-cpp-fit.md` — audio.cpp fit, converter, streaming-session findings, Arm-2 spec.
- `investigations/nemotron_vs_zipformer.md` — full F1–F10 / H1–H7 post-mortem.
- `research/41_v2_peft_armC_FAILED.md` — source of 18.46 vs 28.19 (F8).
- `research/19_prefullrun_review.md` — transcribe≠deployment; `cache_support`; [70,1]/[70,6] mismatch.
- `research/25_nemotron_experiment_summary.md` — the `[70,1]` transcribe numbers; "proxy ≠ ship."
- Memory: `nemotron-vs-zipformer-roadblock`, `validate-against-real-harness`, `eval-worker-cpu-confirmed`.
