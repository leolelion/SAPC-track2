# 04 - Experiment Plan

Executive summary:
1. The ablation order starts with measurement and thread/warmup controls, then touches mel, then structural runtime changes.
2. Every experiment has a single variable, a latency distribution, and an accuracy guardrail.
3. The primary speed metrics are per-stream RTF, max per-chunk compute, callback timing, and official TTFT/TTLT when manifests are available.
4. The primary accuracy guardrail is official min-over-two-refs CER/WER on the fixed 2k Dev ruler.
5. A speed win is rejected if it makes Test1/Dev-style truncation less likely but worsens CER beyond the guardrail.

## Fixed Controls

Use these controls for every runtime experiment:
- Same model checkpoint/export unless the experiment explicitly changes export.
- Same fixed audio set.
- Same host, CPU affinity, and thread settings within an experiment.
- Warm session unless measuring cold-start.
- Model and audio staged on local disk or `/dev/shm`; do not benchmark MooseFS random I/O.
- Record exact git hash, submission directory, zip hash, ORT version, Python version, CPU diagnostic, and environment variables.
- Track total ingestion wall time as well as latency. Track 2's latency timestamps exclude `__init__()` startup in the local harness, but the competition has a 15000 s per-submission time limit.

## Fixed Audio Sets

Minimum sets once SAP data is accessible:
- `sap_short`: 50 utterances, 0-5 s.
- `sap_medium`: 50 utterances, 5-15 s.
- `sap_long`: 50 utterances, 15+ s.
- `sap_ruler_2k`: existing speaker/etiology-balanced Dev ruler for official CER/WER.

Until SAP data is accessible:
- `/tmp/nemo_say_sample.aiff` can be used only as a smoke/pilot sample, never as a conclusion.

## Metrics

Latency/runtime:
- Per-utterance compute RTF.
- Wall RTF for batch-style decode.
- Real-time-paced first non-empty partial visible time.
- Real-time-paced final visible time.
- Max, p50, p90, p99 per-chunk compute time.
- Feature/encoder/decoder time split.
- Encoder calls, decoder calls, feature calls.
- ORT op profile grouped by op type and provider.

Accuracy:
- Official CER/WER through `evaluate.sh`/organizer scorer.
- Transcript parity for exact-runtime rewrites before scorer runs.
- Empty-output count and truncation-like output length anomalies.

Reporting:
- Use p50/p90/p99 distributions. >=100 utterances/runs for acceptance, small pilots allowed only for debugging.

## Experiment E0 - Worker Diagnostic

Question:
- What CPU and thread environment are we optimizing for?

Variable:
- None; diagnostic only.

Procedure:
- Extract `[CPU_DIAGNOSTIC]` from Test1 ingestion log or run a diagnostic submission.
- Record CPU model, physical/logical cores, memory, matmul benchmark, load average.

Decision:
- Define the thread sweep range and target-like pod affinity from this data.

## Experiment E1 - Baseline Linux Single-Stream Profile

Question:
- What actually dominates on real SAP audio?

Variable:
- None; current exact submission.

Procedure:
- Run current `nemo_submission` on `sap_short`, `sap_medium`, and `sap_long`.
- Use `scripts/bench_nemotron_single_stream.py` for instrumentation wrappers around `_ensure_features`, encoder session, and decoder session.
- Enable ORT profiling for one representative utterance per duration bucket.

Decision:
- Rank blockers by measured contribution to RTF and p90 chunk compute.

Accept/reject:
- No code accepted; this is baseline evidence.

## Experiment E2 - Thread Sweep

Question:
- Is `SAPC2_THREADS=4` optimal for the worker?

Variable:
- `SAPC2_THREADS` in `{1,2,3,4,6,8}` adjusted to real core count.

Procedure:
- Use same warm model and fixed audio.
- Measure p50/p90/p99 RTF and max per-chunk compute.
- Repeat with real-time pacing for callback visibility.

Decision:
- Pick the lowest thread count that keeps p99 chunk compute below the 100 ms arrival cadence when possible, or minimizes lag when not possible.

Accuracy guardrail:
- Not needed; thread count should not change output. Still spot-check transcript parity.

## Experiment E3 - Warmup

Question:
- Does bounded startup warmup reduce first-utterance and p90/p99 latency without materially spending the Codabench time budget?

Variable:
- Warmup disabled vs enabled in `Model.__init__()` only.

Procedure:
- Add warmup in a new submission copy.
- Warm up with synthetic/zero input, then call `reset()`.
- Run cold model load plus first N utterances.
- Compare first partial timing, first-utterance RTF, and total startup wall time.
- If possible, test under the same process count as the official batch accuracy pass, because startup warmup may repeat per worker.

Decision:
- Keep if it improves tail latency and model-load cost remains acceptable under the 15000 s submission limit.

Accuracy guardrail:
- Transcript parity after `reset()`.

No-go:
- No warmup in `reset()`, `accept_chunk()`, or `input_finished()`.
- No warmup using evaluation audio, manifest-specific information, or network downloads.

## Experiment E4 - Incremental Mel

Question:
- Does rolling feature extraction remove tail cost while preserving exact outputs?

Variable:
- Full-prefix recompute vs incremental mel.

Procedure:
1. Unit parity: for many chunk prefixes, compare mel matrices from current `_ensure_features()` and incremental implementation.
2. Transcript parity: compare final/partial transcripts on fixed audio.
3. Runtime: profile real SAP duration buckets.
4. Accuracy: official scorer on `sap_ruler_2k`.

Decision:
- Accept only if official CER is unchanged within <=0.1 absolute and p90/p99 runtime improves.

## Experiment E5 - ORT Optimized Model / Profiling Options

Question:
- Can graph optimization reduce encoder time or startup without changing outputs?

Variable:
- Current online optimization vs pre-optimized ORT model, same provider and thread settings.

Procedure:
- Serialize optimized graph with target-compatible options.
- Run transcript parity and profile.
- Compare startup and steady-state runtime separately.

Decision:
- Keep only if runtime/startup improves with no output changes.

Accuracy guardrail:
- Transcript parity first; official scorer if any discrepancy.

## Experiment E6 - Split Decoder/Joiner Export

Question:
- Does three-graph RNNT runtime materially reduce decoder overhead or enable better native runtime integration?

Variable:
- Combined decoder+joint graph vs split prediction-network and joiner graphs from same checkpoint.

Procedure:
- Obtain or export split ONNX files.
- Build a new sibling submission directory.
- Compare logits on fixed encoder frames against current combined graph.
- Compare transcripts on fixed audio.
- Profile decoder/joiner call counts and wall time.
- Run official scorer.

Decision:
- Continue only if parity is proven and measured runtime improves on real SAP/worker-like host.

## Experiment E7 - sherpa-onnx Native Runtime

Question:
- Can sherpa-onnx run this model family faster with C++ streaming features/decoder?

Variable:
- Current Python/ORT loop vs sherpa-onnx runtime.

Procedure:
- First check latest sherpa-onnx model config support on Linux.
- If it still requires encoder/decoder/joiner, combine with E6 split export.
- Bundle wheel offline without replacing NumPy/Torch.
- Compare transcripts and official scorer.

Decision:
- Keep only if packaging is stable, CER guardrail passes, and runtime gain is large enough to justify higher integration risk.

## Experiment E8 - Quantization Variants

Question:
- Is the current dynamic-int8 encoder the best CPU point?

Variable:
- fp32, current dynamic-int8, static/QDQ int8, any available int4 weight-only variant.

Procedure:
- Use same thread setting and fixed audio.
- Measure ORT op profile and RTF.
- Run official scorer for each candidate.

Decision:
- Accept only if CER is within guardrail and runtime improves enough to matter.

## Staged Rollout

1. Measurement: E0, E1.
2. Zero-accuracy-risk controls: E2, E3.
3. Local code cleanup with parity: E4.
4. Encoder/runtime optimization: E5, E8.
5. Structural/native runtime: E6, E7.
6. Only after speed gate: Nemotron fine-tuning and emission-latency training.

## Definition Of Done For Speed Gate

- Current or modified Nemotron decodes the streaming set at worker RTF < 1 with margin.
- Re-upload no longer collapses on Test1: Test1 CER should be close to Dev-style CER, not roughly 2x.
- Official CER/WER guardrail passes.
- TTFT/TTLT are reported from official latency scripts, not only proxy timers.
