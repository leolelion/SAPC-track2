# 29 - Big-picture audit: Nemotron Track 2 logic and assumptions (2026-06-28)

ML-scientist review of the whole Track 2 / Nemotron line of work. Goal: check that the
chain of claims is internally consistent and grounded in real, reproduced evidence, and flag
assumptions that are still unverified before any Codabench submission.

## The objective, restated from `Track2.md`

- Track 2 = streaming ASR, CPU-only, 15000 s per submission.
- Ranking is the Pareto frontier of accuracy (CER primary, min over two refs) and latency
  (`mean(TTFT_p50, TTLT_p50)`). Prize is split among all teams on the frontier of the
  sequestered test2.
- Strategic consequence (confirmed empirically by our own A1 zipformer result): the CER-extreme
  point is always on the Pareto frontier, so being the clear CER leader is sufficient for frontier
  membership even with poor latency. A1 ranked #1 with the worst TTFT on the board.

## The evidence chain (what is actually reproduced)

1. A1 streaming Zipformer-66M, finetuned on SAP, int8 ONNX, is our real, submitted, #1 system:
   Test1 CER 23.44% / WER 31.51%, TTFT p50 1437.87 ms, TTLT p50 94.64 ms. Dev->Test transfer was
   healthy (2k Dev 21.62% -> Test1 23.44%, ~+1.8 abs). This is the bar to beat and the only point
   with a real hidden-test number.
2. Zero-shot Nemotron failed on severe dysarthria (Dev_diag 43.4% CER, 18% empty via NeMo
   transcribe at `[70,1]`). Diagnosed as domain failure, not packaging/speed.
3. Full-data finetune (309k utt / 683 h, cased, pinned `[70,1]`) cut Dev_diag to 23.6% CER
   (encoder-only) vs 25.2% (full-unfrozen), no catastrophic forgetting. Dev speakers verified
   disjoint from Train.
4. A deployment artifact ("Wh ") made encoder-only look broken (Dev_100 11.7%). Root cause traced
   to RNNT SOS init (`_last_token=0` vs blank id 1024). With the SOS fix, encoder-only streams
   clean: Dev_100 1.3% CER, 0/100 "Wh ", and is the best deploy arm on both offline and streaming.
5. Faithful `local_decode.py` deploy-path CER for encoder-only + SOS (FP32), reproduced from both
   the deploy dir and the extracted submission zip:
   - representative Dev-500 (seed 23, speaker-disjoint): 10.49% CER, speaker CI 7.92-13.55%, 6 empty.
   - severe-enriched Dev_diag-425: 24.49% CER, CI 16.06-33.16%, 30 empty.
   - latency on Dev_streaming-123: TTFT p50 1.15 s / p90 2.36 s; TTLT p50 0.31 s / p90 0.35 s.
6. Encoder-int8 candidate: representative Dev-500 10.60% (paired delta +0.12%, CI -0.23..+0.50),
   Dev_diag 24.85% (+0.36), 0 "Wh ", size 839M vs FP32 2.2G. Structural preflight clean
   (MatMulInteger/DynamicQuantizeLinear present, decoder SHA unchanged, IO names preserved).

This chain is coherent and, where it matters for model selection, reproduced on the real harness.
The SOS-fix and encoder-only decisions are well supported.

## Assumptions that are TRUE / well-grounded

- Streaming-native model is required (offline SOTA loses CER when chunked). Verified in lit and our
  own pipeline; Nemotron cache-aware is correct for this track.
- CER is the dominant ranking lever for frontier membership. Verified by A1 (#1 with worst latency).
- Encoder-only beats full-unfrozen: supported by two independent streams (offline transcribe 23.6 vs
  25.2; faithful streaming Dev_100 1.3 vs 5.9). The streaming gap is large, not marginal.
- The SOS bug was a harness bug, not a model failure: discriminating test, 0/100 "Wh " after fix.
- Encoder int8 is near-lossless here: paired Dev-500 delta CI includes ~0, severe set +0.36 only.

## Assumptions still UNVERIFIED (the real risks before submission)

1. SCORER ROBUSTNESS (highest priority). The official `evaluate.sh` Dev-500 gate fails with
   `preds from sgml-ref1 and sgml-ref2 are not identical`. `compute_metrics.compute_from_sgml` is
   shared by the organizers' `scoring.py`. If our Nemotron hypotheses can trip this assert, a real
   Codabench submission (FP32 or int8) could be scored as a failure on Test1. This must be
   understood and made robust on the hyp/wrapper side before submitting anything.
   - Missing control: official `evaluate.sh` was never run on the FP32 Dev-500 hyp, only int8.
     We do not yet know whether this is int8-specific or a general hyp/scorer-path issue.
   - Code analysis (this session): the official normalizer never emits `,`/`:`/`"`, so the SGML
     parser cannot be garbled by ordinary hyp text; and identical hyp tokens reconstruct identically
     even when the alignment op differs (verified with a synthetic case). Therefore the only
     plausible non-unk triggers are (a) `unk`-drop asymmetry that `_is_unk_only_mismatch` should
     already absorb, or (b) a hyp token with a kept non-ASCII symbol (`¢ € £`) or other rare
     character that sclite/the parser handle asymmetrically. The offending rows must be inspected.
   - Tooling: `scripts/diagnose_official_eval.sh` (pod-side, no pod stop) reruns FP32 and int8
     official Dev-500, preserves SGML, and runs `scripts/debug_sgml_pred_mismatch.py` to print the
     exact rows and whether each is unk-only.
2. LATENCY ON A CODABENCH-LIKE CPU. All Nemotron latency (TTFT p50 1.15 s) was measured on the
   H200 pod with 192 vCPUs, not the slower Codabench worker. A 600M encoder will be slower there;
   the A1 66M model already hit 1437 ms TTFT on the real worker. Frontier membership likely still
   holds via CER leadership, but TTFT must be measured on a representative CPU, not assumed.
3. CPU TIME BUDGET. The 20-worker throughput/thread sweep for the finetuned package was skipped.
   The old 0.017 wall-RTF number is from the previous already-int8 zero-shot package, not this
   FP32/finetuned one. Whether the package finishes Test1 (10521 utts) under 15000 s is unverified.
   int8 encoder reduces this risk but it is still unmeasured.
4. UPLOAD SIZE. The FP32 package is 2.2G; the only size that is known to upload is the ~820M int8
   class. This favors the int8 package as the primary submission if its official gate passes.
5. DEV-500 -> TEST1 REPRESENTATIVENESS. Dev-500 is speaker-disjoint and a strong signal, but its
   etiology mix vs hidden Test1 is unknown. 10.49% Dev-500 is not a promise of a Test1 number; the
   honest projection (using A1's ~+1.8 Dev->Test gap as a prior) is low-to-mid teens CER, which
   would still be a large improvement over A1's 23.44%.
6. CONVERGENCE. The v1 finetune was ~4 epochs with noisy early-stop and all-epoch checkpoint
   averaging (`research/21`). Real CER headroom likely remains from a converged + multi-lookahead v2.
   This is upside, not a blocker.

## Recommended decision order

1. Resolve the scorer assert first (control: does FP32 also fail?). Do not submit until the
   organizers' `local_decode.py` + `evaluate.sh` path passes on the extracted package, because the
   same scorer runs on Test1.
2. Prefer the int8 package as the primary submission (size + speed), conditional on (1) and the
   already-acceptable paired CER deltas.
3. Measure the int8 throughput/thread sweep and a Codabench-like latency point.
4. Keep the FP32 package as the correctness fallback; keep A1 as the always-valid public LB entry.
5. Defer the v2 converged/multi-lookahead retrain until the int8 submission is validated and live.

## One-line bottom line

The modeling story (finetune + encoder-only + SOS fix + encoder int8) is sound and reproduced. The
remaining gates are engineering/eval-integrity, not model quality, and the scorer-assert is the one
that can silently fail a real submission, so it is the correct next thing to fix.
