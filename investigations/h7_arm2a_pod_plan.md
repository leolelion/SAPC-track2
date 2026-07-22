# H7 Arm 2a — NeMo-native streaming diagnostic (pod run card)

> **⚠️ EXECUTED & RESOLVED 2026-07-22 — see `research/45`.** The finetuned v1 checkpoint was LOST (pod +
> local), so this ran on the BASE model as A/B/C instead. Result: transcribe **43.4%** ≈ NeMo-native
> streaming **43.8%** ≈ our ONNX export **43.4%** → the pipeline is FAITHFUL; the "C ≈ A ⟹ our export bug"
> rule below did NOT hold up — Arm B showed our export *also* matches transcribe, so nothing collapses on
> base. No cheap fix exists; Nemotron stays retired (ship beam-4). Card kept for the (unlikely) finetuned re-test.

> **Self-contained pod handoff, 2026-07-22.** Executes Step 1 of Q's Nemotron-reopen
> sequence: settle whether the ~10-pt streaming collapse is **our ONNX-export bug** or
> **intrinsic to chunked-cache streaming**, using NeMo's OWN authoritative streaming
> implementation as an independent third path. Closure/learning experiment — **not** a
> route to win, and **not** a submission. Follows `/Users/o/Downloads/CLAUDE.md` (predict
> before acting, stop on failure) and repo rules (`SAPC-template/CLAUDE.md`): never ship on
> a proxy; only `local_decode.py` + official scorer certify.

---

## The question (one line)
Same weights, same audio, `[70,1]`: `transcribe()` scores **~18–24%** but **our** ONNX
cache-aware streaming export scores **28.19%**. Is the collapse in *our export* or in
*chunked-cache streaming itself*?

## Why NeMo-native streaming answers it
Three decode paths of the SAME v1 checkpoint on the SAME set (`Dev_diag`, 425 severe utts,
speaker-disjoint), SAME hand-rolled two-ref CER (`nemo_eval_diag.py` methodology):

| Arm | Path | Pass structure | Context via | Status |
|---|---|---|---|---|
| **A** (ref) | NeMo `transcribe()` @ [70,1] | one forward / utterance | attention mask | ~23.58% (v1); **re-measure fresh this session** |
| **B** (ref) | our ONNX export → `local_decode.py` | 100 ms chunks | our cache stepping | 28.19% (research/41) |
| **C** (NEW) | NeMo `speech_to_text_cache_aware_streaming_infer.py` @ [70,1] | 100 ms chunks | **NeMo's own** caches | **never run** |

**The decisive comparison is C vs A**, because A and C differ in *only one variable* —
masked-single-pass vs chunked-with-caches — while sharing weights, eval set, and scorer.
NeMo's streaming impl is independent of our ONNX export, so:
- **C ≈ A (within ~2 pts):** chunked-cache streaming is fine → the 28.19% collapse is a bug
  in **OUR ONNX export/cache-stepping** → fixable in our code (then re-certify on the real
  harness). Nemotron path potentially revivable on the accuracy axis.
- **C ≈ 28% (collapsed vs A):** even NeMo's own reference streaming collapses at [70,1] →
  the loss is **intrinsic to correct chunked-cache streaming** → no export fix helps; only
  cache-aware finetuning (Arm 3) or a different latency operating point could → per EV, STOP.
- **C between A and 28%:** partial — decompose (re-check att_context strictly matched, then
  weigh residual).

**Write the success criterion BEFORE running:** "The collapse is our export bug IFF fresh
arm-C CER ≤ fresh arm-A CER + 2 pts."

---

## Cost gate — this is a paid GPU pod. Do ALL of this before `pod start`.
- [ ] Q's explicit go for pod spend (this doc is the pre-registered plan; spend is the only
      thing left to approve).
- [ ] Confirm pod `3dwiczo41jeg1y` still exists and its volume is intact (Stage 0 below) —
      **the single biggest unknown.** If the volume/checkpoint is gone, this plan is blocked
      until the v1 checkpoint is restored (separate, expensive) — STOP and report, do not
      improvise a re-finetune.
- [ ] Two script edits staged locally first (see "Local prep").
- Expected pod runtime once RUNNING: **~15–30 min** (model load + 425-utt streaming infer,
  batch 32, GPU, for both v1 and base). GPU is correct here — the decision metric is CER, not
  latency; latency is a *separate* gate not touched by this experiment.

## Local prep (do on the Mac; no pod)
1. **Harden `scripts/run_stream_infer.sh`** (currently downloads the infer script from NeMo
   `main` — API drift on `--asr_model/--manifest_file/--att_context_size/--output_path` would
   break the run):
   - Pin the download to a known-good NeMo ref/tag instead of `main`, OR keep the existing
     `grep add_argument` echo (line 21-22) as a pre-flight and **abort with a clear message**
     if any expected arg name is absent (currently it just prints them).
2. **Add arm-A reference in the SAME session** so A is re-measured fresh (not trusted from a
   3-week-old number) as the internal control. `scripts/nemo_eval_diag.py` already does exactly
   arm A: `python nemo_eval_diag.py <v1.nemo> <Dev_diag.csv> <data_root> "[70,1]"`. Run it right
   before arm C on the same checkpoint. If fresh-A ≠ ~23.58% (±2), STOP — the env/checkpoint
   drifted and no comparison is valid.
3. Confirm the checkpoint path in `run_stream_infer.sh:29`
   (`$OUT/full_enc/ft_smoke_encoder_only.nemo`) is the v1 BEST artifact (research/20/24). Stage 0
   validates this by reproducing arm A on it.

---

## Pod stages (stop after each; report before the next)

**Stage 0 — verify volume + env (cheap; ssh only, no compute).** After the pod is RUNNING
(reuse `autorun_stream.sh`'s start/endpoint logic), ssh in and confirm ALL exist, else STOP:
- `/workspace/finetune/nemo_ft/full_enc/ft_smoke_encoder_only.nemo` (v1 checkpoint)
- `/workspace/finetune/nemo_ft/nemotron-speech-streaming-en-0.6b.nemo` (base; else re-pull)
- `/workspace/nemoenv/` (venv), `/workspace/sapc-nemotron/utils/normalizer/` (scorer)
- `/workspace/SAPC2/manifest/Dev_diag.csv` (+ the referenced audio under `/workspace/SAPC2`)

**Stage 1 — arm A fresh reference (transcribe @ [70,1]).**
```bash
source /workspace/nemoenv/bin/activate
python3 /workspace/nemo_eval_diag.py \
  /workspace/finetune/nemo_ft/full_enc/ft_smoke_encoder_only.nemo \
  /workspace/SAPC2/manifest/Dev_diag.csv /workspace/SAPC2 "[70,1]"
```
EXPECT: ALL CER ≈ 23.6% (±2). IF NO → STOP (env/checkpoint drift; comparison void).

**Stage 2 — arm C NeMo-native streaming @ [70,1].** Run the hardened `run_stream_infer.sh`
(it already: fetches the NeMo cache-aware streaming infer script, builds the NeMo json
manifest from `Dev_diag.csv`, runs streaming for v1 + base at `--att_context_size 70 1`,
scores per-etiology two-ref CER with the same normalizer). Output: `stream_full70_1.json`,
`stream_base70_1.json`, printed CER, `stream_infer.log`.
EXPECT: a v1 streaming ALL-CER to compare against Stage-1 arm A.

**Stage 3 — decide + copy back, then STOP the pod immediately.**
- Copy back: `stream_infer.log`, `stream_full70_1.json`, `stream_base70_1.json`, and the
  Stage-1 transcribe output.
- Apply the decision rule (C vs fresh A) above. Record the verdict in `research/` +
  `EXPERIMENT_LOG.md` + update `investigations/h7_fix_plan.md` decision tree.
- `runpodctl pod stop 3dwiczo41jeg1y`.

## Stop-on-failure triggers (per Rule 0 — output WORDS, not a retry)
- Stage 0 missing checkpoint/env → STOP (blocked on artifact restore).
- Stage 1 fresh-A off by >2 pts from 23.58% → STOP (invalid baseline).
- NeMo streaming infer script arg mismatch / crash → STOP with exact error (NeMo main drift).
- Any export/decode exception → STOP, report exact error, do not silently retry.

---

## What a result does and does NOT change (honest EV framing)
This is **closure**, matching Q's framing. Even the favorable outcome (C ≈ A ⇒ our export bug)
does **not** by itself flip the ship decision:
- The binding constraint is the **Dev→Test transfer wall (F9)**: Nemotron slope +18 vs
  zipformer +1.8. Recovering Dev streaming accuracy need not transfer to Test1 (int8 `_t1`
  already went Dev_diag 23.58 → Test1 27.97).
- CPU streaming latency at 0.6B is still unverified (track is latency-scored, CPU-only).
- Any CER here is a **proxy** (hand-rolled two-ref, like arms A/B). Shipping still requires
  `local_decode.py` + official scorer.
So: C≈A reopens *an accuracy-fixable path* worth a follow-up; C≈28% *retires Nemotron with
proof*. Either way the default remains **ship gated beam-4 zipformer (~21%)** unless a later,
explicitly-approved bet clears the transfer wall too.

## Files
- Runner: `scripts/run_stream_infer.sh` (arm C) + `scripts/nemo_eval_diag.py` (arm A).
- Launcher: `scripts/autorun_stream.sh` (pod `3dwiczo41jeg1y`, start→run→pull-log→stop).
- Context: `investigations/h7_fix_plan.md`, `investigations/nemotron_vs_zipformer.md` (F8/H7),
  `investigations/audio-cpp-fit.md` (arm-2 audio.cpp alt), `research/41` (18.46 vs 28.19).
</content>
</invoke>
