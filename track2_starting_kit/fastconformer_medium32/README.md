# fastconformer_medium32 — low-latency Pareto-corner candidate

~32M cache-aware streaming FastConformer-Hybrid (`stt_en_fastconformer_hybrid_medium_streaming_80ms`),
run at **80 ms lookahead** (`att_context_size=[70,1]`), RNN-T head. Rationale + experiment gate:
**`research/46_fastconformer_medium_fastlane_plan.md`**.

## Verification status
- **Verified locally** (no NeMo): 5-method contract, 100 ms→model-chunk buffering, callback firing,
  reset semantics, NeMo-return text extraction. `python3 -m py_compile model.py` + bare `import` pass.
- **NOT verified** (needs pod): the two `# >>> VERIFY-ON-POD <<<` blocks in `model.py` — streaming-param
  setup and `conformer_stream_step`'s exact return tuple (the version-sensitive spot that broke before).
- **No number from this dir is trustworthy** until it runs through the real `local_decode.py` (both passes)
  + `evaluate.sh` on Dev. House rule: `validate-against-real-harness`.

## Run (E1, on pod)
```bash
bash setup.sh                      # installs NeMo, downloads .nemo into weights/
cd .. # track2_starting_kit
python3 local_decode.py \
  --submission-dir ./fastconformer_medium32 \
  --manifest-csv  $DATA_ROOT/manifest/Dev_streaming.csv \
  --streaming-manifest-csv $DATA_ROOT/manifest/Dev_streaming.csv \
  --data-root $DATA_ROOT \
  --out-csv ./Dev_streaming.medium32.csv \
  --out-partial-json ./Dev_streaming.medium32.partial.json
```
Then `evaluate.sh` stages 0–2 (accuracy) and 3 (latency). Decision metrics: research/46 §2.
