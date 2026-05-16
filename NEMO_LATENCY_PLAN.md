Strong result. Zero-shot R1 at 17.81% CER with TTFT P50 1.9s on CPU, 4 threads, is already a competitive Track 2 submission on its own. Before deciding, a few observations on what you have:

**The RTF number matters more than it looks.** RTF 1.117 on a 4-thread CPU with no batching means you're *barely* keeping up with real time, and the tail is structurally the last chunk's compute. If the eval environment is slower than your pod by even 20%, streaming falls behind real-time and TTLT explodes. **Check what the SAPC2 eval environment actually provides** (core count, CPU generation) before optimizing anything else — if it's a 2-core VM, none of your numbers transfer. This is the biggest risk to the whole plan and it's cheap to resolve.

**The 1.895s TTFT decomposition is informative.** The structural floor is roughly "1120ms first chunk + ~700ms compute + ~80ms decode = ~1.9s". That matches what you measured, which means there's basically no room at this chunk size. The only TTFT wins are architectural (smaller chunk) or compute (int8). You haven't tried int8 yet — and the danielbodart export exists precisely for this. **At 4 threads on the same CPU, int8 should cut compute roughly in half**, dropping TTFT toward ~1.5s with no accuracy change. That's a free Pareto improvement.

**The Phaedrus R3 failure is now fully explained.** Children's speech overfit + no improvement on SAP = 25.48 CER. Drop it from the plan permanently.

## My recommendation: do (a) and add (d), skip (b) for now

**(a) Multi-lookahead sweep — yes, but verify first.** The Nemotron-Speech-Streaming model card documents four chunk presets (`[70,0]`, `[70,1]`, `[70,6]`, `[70,13]`) and explicitly says *"at inference time, without requiring any re-training"* — so it *does* support runtime switching. The line in NeMo is:

```python
asr_model.encoder.set_default_att_context_size([70, 1])
```

Then re-run the streaming harness. You get three new Pareto points (80ms, 160ms, 560ms chunks) from the same checkpoint. Published WER at these settings is 8.43 / 7.67 / 7.07 on LibriSpeech-averaged — so on Dev_streaming I'd ballpark +1–2 CER per step down. Worth ~30 minutes of compute per setting and gives you **four Pareto points instead of one** from zero-shot.

**(d) int8 on CPU — the one thing you haven't tried.** Pull `danielbodart/nemotron-speech-600m-onnx` int8-dynamic, write a small ONNX streaming harness (same sleep-until-arrival logic, just swapping NeMo PyTorch for two ONNX Runtime sessions). Use the same `Dev_streaming.csv`. This gives you the Pareto point *you'd actually deploy*, and tests whether the int8 calibration breaks CER. Expected: CER within 0.5 points of FP32, TTFT dropping from 1.9s toward ~1.3–1.5s. If it works, this is the single biggest Pareto improvement available to you without touching the model weights.

**(b) Bigger streaming subset — defer.** 123 utterances is enough for stable P50 (sampling error ~±50ms). P90/P95 are noisier but not critical for the Pareto frontier which uses P50. Skip until after fine-tuning — at that point expanding the subset is a one-time cost that validates the final number.

**(c) SAP finetune — defer one step.** You want to finetune *after* the int8 test because your deployment target is int8 Nemotron. If int8 breaks accuracy badly (unlikely, given danielbodart's cache-aware calibration), you'd need to finetune with quantization-aware training instead of post-hoc quantization, and that's a very different pipeline. One day of int8 smoke-testing saves a week of wasted fine-tuning.

## Concrete next session

1. **15 min**: confirm SAPC2 eval environment specs. If it's <4 cores or much slower per-core, everything downstream changes.
2. **1 hr**: multi-lookahead sweep — re-run the harness at `[70,0]`, `[70,1]`, `[70,6]` on Dev_streaming, add three rows to the table.
3. **Half day**: download danielbodart int8-dynamic, adapt the streaming harness to ONNX Runtime (same sleep loop, swap model calls), re-run at `[70,13]`. Verify CER stays within 0.5 of 17.81.
4. **If int8 holds**: run the full multi-lookahead sweep on int8 too. Now you have 4–8 Pareto points, all on CPU, and you know which chunk size you'd finetune toward.
5. **Then**: SAP finetune on the Nemotron PyTorch checkpoint, starting from the Takahashi recipe (full finetune, no freezing, SpecAugment + speed perturb, 10–20 epochs, 4-checkpoint averaging). Target the chunk size that won the Pareto analysis.

One caveat on step 2: the multi-lookahead runtime switching is documented for the *production* checkpoint but I'd smoke-test it on 5 utterances before the full 123-utt run. If `set_default_att_context_size` silently no-ops, you'd get identical TTFT numbers and waste the run. Cheap to verify: at `[70,0]` the TTFT should drop dramatically (<500ms). If it doesn't, something's wrong.

Which do you want to run first — the multi-lookahead verification, or the int8 port? I'd do multi-lookahead first because it's cheaper and tells you whether you even need to bother with int8 at 1120ms (if `[70,1]` gives 17.9 CER + 400ms TTFT zero-shot, you might just ship that).