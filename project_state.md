# SAPC2 Track 2 — Project State Briefing

Generated 2026-05-27 from `/Users/o/Development/SAPC-template` on macOS (Apple
M3, 8 GB RAM, Python 3.13). Local dev only — no GPU on this machine; all real
training/eval was run on RunPod. The repo's `Track2 (Streaming ASR Track).md`
has the official task description; SAPC2 Track 2 is streaming ASR for dysarthric
speech, CPU-only evaluation, hard RTF<1.0 gate, scored on CER (primary) + WER
(secondary) + TTFT/TTLT P50.

---

## 1. Repo layout

```text
.
|-- BENCHMARK_PLAN.md              # Plan: baseline-vs-finetuned benchmark on SAPC2 Dev (5 phases, NeMo path)
|-- Claude.pdf                     # 3.2 MB; SAPC challenge background dump
|-- NEMO_LATENCY_PLAN.md           # Follow-up plan after first Nemotron run (multi-lookahead + int8)
|-- README.md                      # Pipeline doc for preprocess.sh + evaluate.sh
|-- Track2\ (Streaming\ ASR\ Track).md   # Official SAPC2 task spec (interface + scoring)
|-- dev100_bundle/                 # Local Dev set, 100 dysarthric utts, all ALS speakers, ~19 min audio
|-- dev_streaming_bundle/          # Local Dev_streaming subset, 123 utts, multi-etiology, ~13 min audio
|-- eval_per_etiology.sh           # Helper: split metrics by etiology column
|-- evaluate.sh                    # SCTK-based eval (CER/WER/latency)
|-- finetuning/                    # icefall Zipformer finetune scripts + README
|-- inference_debug.md             # Notes on a broken finetune-export pipeline (Apr 5)
|-- local_results/                 # One run output: zipformer_standard on Dev_streaming (CSV + latency JSON + log)
|-- preprocess.sh                  # 4-stage SAP data preprocessing
|-- run_nemo_inference.py          # NeMo `.transcribe()` harness (Phase 2 of BENCHMARK_PLAN)
|-- run_nemo_streaming_latency.py  # NeMo cache-aware streaming harness with real-time sleep loop
|-- run_onnx_streaming_latency.py  # ONNX Runtime streaming harness (danielbodart int8 layout)
|-- start-pod-watcher.sh           # RunPod helper script
|-- steps/                         # Sub-scripts called by preprocess.sh & evaluate.sh
|-- track1_starting_kit/           # Track 1 baselines (canary_qwen, parakeet, whisper) — not relevant to Track 2
|-- track2_starting_kit/           # All Track 2 model kits (see below)
|-- utils/                         # Metric, normalizer, manifest helpers
|-- watcher.log                    # RunPod watcher log (small)
```

Top-level dirs explained:
- `track2_starting_kit/` — 7.1 GB. One subdir per candidate streaming model:
  `emformer_rnnt/`, `faster_whisper/`, `gigaam/`, `moonshine/`, `nemotron_streaming/`,
  `parakeet_ctc/`, `qwen3_asr/`, `sherpa_zipformer/`, `streaming_zipformer/`, `wenet_u2pp/`.
  Plus `local_results/` (eval CSVs/JSONs for each model), `local_decode.py`, `run_score.py`.
- `track1_starting_kit/` — Track 1 (non-streaming) baselines, mostly untouched.
- `finetuning/` — Stand-alone icefall Zipformer SAP finetuning recipe (LibriSpeech epoch-30.pt warm-start);
  pretrained/ and data/ subdirs are empty locally (filled on RunPod).
- `dev100_bundle/` (70 MB) — 100-utt subset, ALL ALS speakers, used as quick accuracy probe (Dev_100.csv).
- `dev_streaming_bundle/` (24 MB) — 123-utt streaming subset with MFA timing columns, used for TTFT/TTLT (Dev_streaming.csv).
- `steps/` — bash subroutines: env setup, unzip, preprocess, streaming subset generation, evaluate.sh stages.
- `utils/` — normalizer (HF ASR leaderboard style), `compute_latency.py`, `compute_metrics.py`, manifest builders.
- `local_results/` (728 KB at repo root) — single local run output for `zipformer_standard` on Dev_streaming.

Disk usage > 100 MB:
```text
 16K	steps/
 44K	track1_starting_kit/
 76K	finetuning/
252K	utils/
728K	local_results/
 24M	dev_streaming_bundle/
 70M	dev100_bundle/
7.1G	track2_starting_kit/
```

Inside `track2_starting_kit/`:
```text
168M	track2_starting_kit/moonshine/         (venv + ONNX)
203M	track2_starting_kit/wenet_u2pp/weights
321M	track2_starting_kit/sherpa_zipformer/weights  (standard + kroko ONNX)
528M	track2_starting_kit/wenet_u2pp/venv
583M	track2_starting_kit/gigaam/venv
1.4G	track2_starting_kit/qwen3_asr/venv
4.0G	track2_starting_kit/parakeet_ctc/weights      (single final.nemo, 4.06 GB)
```

---

## 2. Git state

Current branch: `main` (HEAD = `b40d653 Chnages`)

All branches (latest commit first):
```text
claude/amazing-edison-c7d764 | 2026-05-16 | Add NEMOTRON_PLAN.md: Nemotron streaming replication + fine-tune plan
main                         | 2026-05-16 | Chnages
my-work                      | 2026-03-19 | save my work before syncing with upstream
claude/dazzling-kalam        | 2026-03-12 | Update README with submission guidelines and model details
```

Last 30 commits on main:
```text
b40d653 Chnages
02aab0f remove old files
064cfd4 Fix parakeet_ctc setup: detect nested venv, pin torch version
5d4b19a Fix venv path search to find nested venv dir
28ad9b5 Skip NeMo venv reinstall if already installed in parakeet_ctc/setup.sh
dd9f264 Add parakeet_ctc and streaming_pruned_stateless7 starting kits with scoring scripts
32c1965 Add finetuning test suites (force-add past gitignore)
d573834 delete old files
58fdbe6 Add finetuning/ directory: scripts, tests, and setup for streaming Zipformer SAPC finetuning
e96d385 script fixes
200121c kroko extraction
8deeef6 Update gitignore
b22dda3 changes
cfcfd0a fix
caf20d6 wenetu2 fit
008ed37 Update setup.sh dependencies
cb45172 add new files
0b39b56 update gitignore
2433782 model fix for net2
4b56e5b fixes
0aea250 wenet fix
ddf6d8d fixes
7738926 FIx bug
7b588eb Add wenet u2
666218a Add nemo fast conformer
df8ff87 Remove moonshine v1 code
32441fb Moonshine v2
6d28dff Add hugging face dependency
4000bb9 fix model weights
b358078 Update gitignore
```

Uncommitted changes:
```text
?? .claude/worktrees/
```
Only an untracked worktrees directory under `.claude/`. Nothing else dirty.

Branch notes:
- `claude/amazing-edison-c7d764` (May 16) lives in `.claude/worktrees/amazing-edison-c7d764/`.
  This is the active Nemotron-streaming exploration branch and **contains content not in main**:
  `NEMOTRON_PLAN.md`, `CLAUDE.md`, and the entire `track2_starting_kit/experiments/` ledger
  (`results.jsonl`, `LEADERBOARD.md`, `experiments.py`, `README.md`).
- `my-work` (Mar 19) is two months stale — looks abandoned.
- `claude/dazzling-kalam` (Mar 12) is the very first claude branch — abandoned.
- `streaming_pruned_stateless7/` kit was added in `dd9f264` then deleted in `02aab0f`
  ("remove old files"). Result CSVs are still in `track2_starting_kit/local_results/`
  (see Section 4) but the model code is gone.

---

## 3. Current best submission

There are two "best" answers depending on what you mean:

**(A) Best on the actual Codabench leaderboard (Test1):** the SAPC2 official
baseline `sherpa_zipformer/standard` (zero-shot LibriSpeech streaming Zipformer).
This is the only submission with a Test1 number recorded in the experiments ledger
([.claude/worktrees/amazing-edison-c7d764/track2_starting_kit/experiments/results.jsonl:1](.claude/worktrees/amazing-edison-c7d764/track2_starting_kit/experiments/results.jsonl#L1)):
WER 52.77 / CER 34.59 / TTFT 1025 ms / TTLT 423 ms, dated 2026-03-01.

**(B) Best result obtained anywhere (local Dev100, zero-shot ONNX):**
`sherpa_zipformer/kroko` — Zipformer2 streaming, CER 5.87% / WER 12.54% on
dev100 (100 ALS utts), no finetuning. Same kit code, different ONNX weights.

Both share the same model.py: [track2_starting_kit/sherpa_zipformer/model.py](track2_starting_kit/sherpa_zipformer/model.py).

### model.py (sherpa_zipformer)

```python
#!/usr/bin/env python3
"""
Sherpa-ONNX Streaming Zipformer — SAPC2 Track 2
================================================

Wraps sherpa-onnx's OnlineRecognizer (Zipformer-Transducer) with the
5-method streaming interface required by the ingestion program.

Unlike the icefall-based streaming_zipformer/, this implementation:
  - Uses pre-exported ONNX models (no PyTorch at inference time)
  - Has a single clean dependency: `pip install sherpa-onnx`
  - Runs natively on CPU with ONNX Runtime (faster than PyTorch CPU)
  - Supports all three Zipformer size variants via config.yaml

This is the recommended submission vehicle for the competition.

Supported variants (set model.variant in config.yaml):
  standard  — ~70M params, LibriSpeech-trained (official baseline)
  kroko     — Zipformer2 architecture, edge-optimised
  small     — ~20M params, Pareto latency anchor

How it streams:
  The Zipformer-Transducer is a true streaming model. Each call to
  accept_chunk() feeds audio directly to the encoder and fires the
  partial callback as soon as new tokens are decoded. No buffering.

Required interface (called by ingestion program):
  __init__()                       — Load model weights (once)
  set_partial_callback(fn) -> None — Register partial result callback
  reset()             -> None      — Reset state per audio file
  accept_chunk(buf)   -> str       — Feed 100 ms audio chunk
  input_finished()    -> str       — Signal end of audio, return text

To change settings, edit config.yaml (not this file).

Directory layout after running setup.sh:
  sherpa_zipformer/
  ├── model.py       ← this file
  ├── config.yaml    ← all tunable settings
  ├── setup.sh       ← installs sherpa-onnx and downloads ONNX weights
  └── weights/
      └── <variant>/
          ├── encoder.onnx
          ├── decoder.onnx
          ├── joiner.onnx
          └── tokens.txt
"""

# =====================================================================
# Section 1: Imports
# =====================================================================
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from omegaconf import OmegaConf

try:
    import sherpa_onnx
except ImportError as exc:
    raise ImportError(
        "sherpa-onnx is not installed. Run setup.sh first, or:\n"
        "  pip install sherpa-onnx"
    ) from exc

# =====================================================================
# Section 2: Config
# =====================================================================
_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_config = OmegaConf.load(_DIR / "config.yaml")


# =====================================================================
# Section 3: Model — Public Interface for the Ingestion Program
# =====================================================================
class Model:
    """Streaming ASR model wrapping sherpa-onnx Zipformer-Transducer.

    True streaming: audio is processed frame-by-frame as chunks arrive;
    partial callbacks fire as soon as new tokens are decoded.

    Lifecycle (called by ingestion program):
      model.set_partial_callback(fn)           # register callback (once)
      model.reset()                             # prepare for new file
      for chunk in audio_chunks:
          partial = model.accept_chunk(chunk)   # returns partial text
      final = model.input_finished()            # returns final text
    """

    def __init__(self):
        variant = _config.model.variant
        weights_dir = _DIR / "weights" / variant
        print(f"Loading sherpa-onnx Zipformer ({variant}) from {weights_dir} …")

        self._partial_callback: Optional[Callable[[str], None]] = None
        self._recognizer = self._build_recognizer(weights_dir)
        self._stream = None  # created in reset()

        print(f"sherpa-onnx Zipformer ({variant}) loaded (cpu)")

    # -----------------------------------------------------------------
    # Streaming Interface (called by the ingestion program)
    # -----------------------------------------------------------------

    def set_partial_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback for partial results: callback(text: str)."""
        self._partial_callback = callback

    def reset(self) -> None:
        """Reset state for a new audio file. Call once before each file."""
        self._stream = self._recognizer.create_stream()

    def accept_chunk(self, audio_chunk: np.ndarray) -> str:
        """Feed one 100 ms audio chunk (float32, 16 kHz) and return partial text.

        Feeds the chunk to the encoder and decodes any newly available
        frames. Fires the partial callback if the hypothesis changed.
        """
        self._stream.accept_waveform(
            sample_rate=_config.audio.sample_rate,
            waveform=audio_chunk,
        )
        self._drain()
        text = self._recognizer.get_result(self._stream).strip()
        if text and self._partial_callback is not None:
            self._partial_callback(text)
        return text

    def input_finished(self) -> str:
        """Signal end of audio. Flushes encoder tail, returns final text."""
        # Append 0.3 s of silence to flush the encoder's right-context frames.
        tail = np.zeros(
            int(_config.audio.sample_rate * 0.3), dtype=np.float32
        )
        self._stream.accept_waveform(
            sample_rate=_config.audio.sample_rate,
            waveform=tail,
        )
        self._stream.input_finished()
        self._drain()
        final_text = self._recognizer.get_result(self._stream).strip()
        if self._partial_callback is not None:
            self._partial_callback(final_text)
        return final_text

    # -----------------------------------------------------------------
    # Private Helpers
    # -----------------------------------------------------------------

    def _drain(self) -> None:
        """Decode all frames that the recognizer has ready."""
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

    def _build_recognizer(self, weights_dir: Path) -> sherpa_onnx.OnlineRecognizer:
        """Construct and return an OnlineRecognizer from config + weights."""
        cfg = _config
        w = weights_dir

        return sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=str(w / cfg.model.encoder_file),
            decoder=str(w / cfg.model.decoder_file),
            joiner=str(w / cfg.model.joiner_file),
            tokens=str(w / cfg.model.tokens_file),
            num_threads=cfg.model.num_threads,
            provider="cpu",
            sample_rate=cfg.audio.sample_rate,
            feature_dim=cfg.audio.feature_dim,
            decoding_method=cfg.decoding.method,
            max_active_paths=cfg.decoding.max_active_paths,
            enable_endpoint_detection=cfg.endpoint.enable,
            rule1_min_trailing_silence=cfg.endpoint.rule1_min_trailing_silence,
            rule2_min_trailing_silence=cfg.endpoint.rule2_min_trailing_silence,
            rule3_min_utterance_length=cfg.endpoint.rule3_min_utterance_length,
        )
```

### config.yaml (sherpa_zipformer)

```yaml
# =====================================================================
#  Sherpa-ONNX Zipformer Streaming — Configuration
#
#  All tunable settings live here. Edit values below to experiment,
#  then re-run. No need to modify model.py.
#
#  Variant guide (model.variant):
#    standard  ~70M params  LibriSpeech-trained, official baseline
#              Expected: CER ~34% zero-shot, RTF < 0.3× on CPU
#    kroko     Zipformer2 architecture, edge-optimised variant
#              Expected: similar or better accuracy vs. standard at
#              similar or lower latency
#    small     ~20M params  Pareto latency anchor (embedded/Cortex-A7)
#              Expected: worse zero-shot accuracy, much lower RTF
#
#  After running setup.sh for a variant, weights land in:
#    sherpa_zipformer/weights/<variant>/
#
#  To switch variants:
#    1. Change model.variant below
#    2. Re-run setup.sh (it reads this file to pick the download target)
# =====================================================================

# --- Model ---
model:
  variant: "standard"

  # ONNX file names within weights/<variant>/ (usually unchanged)
  encoder_file: "encoder.onnx"
  decoder_file: "decoder.onnx"
  joiner_file:  "joiner.onnx"
  tokens_file:  "tokens.txt"

  # Number of CPU threads for ONNX Runtime inference.
  # On the evaluation server, 4 is a safe default.
  num_threads: 4

# --- Audio Input ---
audio:
  sample_rate: 16000              # Hz — must match the competition's 16 kHz stream
  feature_dim: 80                 # Mel filterbank bins (standard for Zipformer)

# --- Decoding Strategy ---
decoding:
  method: "greedy_search"         # "greedy_search" | "modified_beam_search"
                                  # greedy_search: fastest, ~1% worse WER
                                  # modified_beam_search: max_active_paths beams
  max_active_paths: 4             # Beam width (only for modified_beam_search)

# --- Endpoint Detection ---
# When enabled, the recognizer fires an endpoint after a configurable
# silence duration. Disabled by default — the competition ingestion
# program controls utterance boundaries via input_finished().
endpoint:
  enable: false
  rule1_min_trailing_silence: 2.4 # s — silence after voice (safe default)
  rule2_min_trailing_silence: 1.2 # s — silence after non-blank tokens
  rule3_min_utterance_length: 20  # s — force endpoint for long utterances
```

Note: on this checkout `model.variant: "standard"` is committed, but the local
best Dev100 numbers came from the same kit with `variant: kroko` (kroko ONNX
weights are present on disk at [track2_starting_kit/sherpa_zipformer/weights/kroko/](track2_starting_kit/sherpa_zipformer/weights/kroko/),
67 MB encoder.onnx). UNKNOWN — need user input — which variant is actually
packaged into the submitted zip for the official Test1 baseline; I assumed
"standard" based on the leaderboard entry's `model_source` field saying "icefall
librispeech streaming zipformer".

### Streaming strategy & per-chunk cost (sherpa_zipformer)

**True cache-aware streaming**, not batch-accumulation or sliding-window. Each
`accept_chunk(100ms)` call feeds the waveform into sherpa-onnx's
`OnlineRecognizer`; the underlying ONNX encoder advances by `decode_chunk_len`
fbank frames (~320 ms model chunk per `inference_debug.md`), with left-context
caches `[128, 64, 32, 16, 32, 64]` carried frame-to-frame. Token IDs decode
incrementally and the partial callback fires whenever the hypothesis text
changes. No re-encoding of past audio.

Per-chunk compute: one ONNX encoder step per ~320 ms model chunk (so ~1 step
per 3 input chunks), one joiner pass per emitted token, no GPU. ONNX Runtime
CPU with 4 threads is what config.yaml currently sets. RTF on the competition
runtime UNKNOWN — need user input (the only RTF measurement in the ledger is
for `nemotron_streaming`, not sherpa_zipformer).

---

## 4. Leaderboard / results so far

### Test1 (official Codabench)

```text
| Model                       | WER%  | CER%  | TTFT P50 ms | TTLT P50 ms | RTF | Verified |
|-----------------------------|-------|-------|-------------|-------------|-----|----------|
| sherpa_zipformer/standard   | 52.77 | 34.59 |     1025    |     423     |  —  |   NO     |
```

Source: [.claude/worktrees/amazing-edison-c7d764/track2_starting_kit/experiments/results.jsonl:1](.claude/worktrees/amazing-edison-c7d764/track2_starting_kit/experiments/results.jsonl#L1).
This entry is backfilled and unverified per the experiments ledger schema. No
other Test1 submission is recorded. UNKNOWN — need user input — whether
anything else has been actually submitted to Codabench since.

### Full ledger (worktree branch, 9 runs)

Verbatim from [.claude/worktrees/amazing-edison-c7d764/track2_starting_kit/experiments/LEADERBOARD.md](.claude/worktrees/amazing-edison-c7d764/track2_starting_kit/experiments/LEADERBOARD.md):

```text
| Dataset       | Model                                  | WER%  | CER%  | TTFT P50 ms | TTLT P50 ms | RTF   | Gate  | Date       | Commit  | Verified |
|---------------|----------------------------------------|-------|-------|-------------|-------------|-------|-------|------------|---------|----------|
| Dev           | qwen3_asr                              | 23.34 | 14.23 |     —       |     —       |   —   | FAIL  | 2026-03-01 |   —     |   NO     |
| Dev           | whisper-large-v2                       | 26.08 | 20.36 |     —       |     —       |   —   |  —    | 2026-03-01 |   —     |   NO     |
| Dev           | parakeet-tdt-0.6b-v2                   | 51.64 | 44.64 |     —       |     —       |   —   |  —    | 2026-03-01 |   —     |   NO     |
| Dev100        | sherpa_zipformer/standard-finetuned-1ep| 11.80 |  7.75 |     —       |     —       |   —   |  —    | 2026-04-07 |   —     |   NO     |
| Dev100        | sherpa_zipformer/kroko                 | 12.54 |  5.87 |     —       |     —       |   —   |  —    | 2026-04-07 |   —     |   NO     |
| Dev100        | sherpa_zipformer/standard              | 16.84 |  7.95 |     —       |     —       |   —   |  —    | 2026-04-07 |   —     |   NO     |
| Dev_streaming | nemotron_streaming                     |  —    | 17.81 |    1895.0   |     —       | 1.117 | FAIL  | 2026-05-16 | b40d653 |  yes     |
| Dev_streaming | nemotron_streaming/phaedrus-children-v17|  —   | 25.48 |     —       |     —       |   —   |  —    | 2026-05-16 | b40d653 |  yes     |
| Test1         | sherpa_zipformer/standard              | 52.77 | 34.59 |    1025     |    423      |   —   |  —    | 2026-03-01 |   —     |   NO     |
```

Notes:
- All `qwen3_asr`, `whisper-large-v2`, `parakeet-tdt-0.6b-v2` rows are GPU
  batch eval on full Dev (47,929 utts), NOT streaming/CPU — accuracy reference only.
- `sherpa_zipformer/standard-finetuned-1ep` finetuned 1 epoch on SAP from
  LibriSpeech epoch-30.pt. Per `inference_debug.md` the export pipeline broke
  (sherpa-onnx returned empty strings; PyTorch direct also empty due to a
  missed `encoder_embed` subsampling step). The 7.75 CER number is suspicious
  and the entry is marked `verified: false`.
- `sherpa_zipformer/kroko` 5.87 CER zero-shot is the current local high score
  on Dev100. Memory file `project_kroko_arch.md` (per MEMORY.md) holds the
  Zipformer2 architecture details extracted from the ONNX.
- `nemotron_streaming` zero-shot on Dev_streaming: **RTF 1.117 FAILS the
  Track 2 gate** (RTF < 1.0). att_context_size [70,13] ≈ 1120 ms chunk.

### Additional dev100 results (not yet in ledger)

From [track2_starting_kit/local_results/pruned_stateless7_dev100_comparison.md](track2_starting_kit/local_results/pruned_stateless7_dev100_comparison.md):

```text
| Model                            | Variant                  | N   | WER%  |
|----------------------------------|--------------------------|-----|-------|
| streaming_ps7_libri_giga         | LibriSpeech + GigaSpeech | 100 | 13.22 |
| zipformer_standard               |  —                       | 100 | 21.17 |
| zipformer_kroko                  |  —                       | 100 | 21.38 |
| wenet_u2pp                       |  —                       | 100 | 24.21 |
| moonshine_small                  |  —                       | 100 | 29.27 |
| emformer_rnnt                    |  —                       | 100 | 33.78 |
| moonshine_tiny                   |  —                       | 100 | 44.10 |
```

These are dev100 batch-mode CSV results, ranked by WER. The
streaming_ps7_libri_giga number (13.22% WER) is interesting because the kit
code was deleted in commit `02aab0f` — only the result CSV remains, and the
weights file UNKNOWN — need user input — likely lives on the RunPod pod.

### Local eval metric JSONs (current main, repo root)

`dev100_bundle/eval/metrics.Dev_100_local.json` (best on dev100, finetuned):
```json
{"n_utts": 100, "wer": 0.11800404638051987, "cer": 0.0775466114282608}
```
`dev100_bundle/eval/metrics.Dev_100_local.kroko.json` (kroko zero-shot):
```json
{"wer": 0.12542144302090358, "cer": 0.058667109855409676, "n_utts": 100}
```
`dev100_bundle/eval/metrics.Dev_100_local.baseline.json` (standard zero-shot):
```json
{"n_utts": 100, "wer": 0.16835017502307892, "cer": 0.07947645336389542}
```
`dev_streaming_bundle/eval/metrics.Dev_streaming_local.json` (best on Dev_streaming, finetuned):
```json
{"n_utts": 123, "wer": 0.2306777685880661, "cer": 0.16938775777816772}
```
`dev_streaming_bundle/eval/metrics.Dev_streaming_local.baseline.json`:
```json
{"n_utts": 123, "wer": 0.4617210626602173, "cer": 0.28451788425445557}
```

### Latency JSON (local zipformer_standard on Dev_streaming)

[local_results/zipformer_standard_dev_streaming.latency.json](local_results/zipformer_standard_dev_streaming.latency.json):
```json
{
  "n_utts_total": 123,
  "ttft_sec": {"p50": 1.048, "p90": 1.430, "p95": 1.703},
  "ttlt_sec": {"p50": 0.0707, "p90": 0.1046, "p95": 0.1106}
}
```
So `zipformer_standard` (sherpa-onnx, standard variant) on Dev_streaming locally:
**TTFT P50 ≈ 1048 ms, TTLT P50 ≈ 71 ms**. Note this is the local pod, not the
eval server CPU.

---

## 5. Dataset state

**Local data on this machine is dev-only.** The full SAP corpus is NOT on
this Macbook — it lives on RunPod under `/workspace/SAPC2/` (per
`run_nemo_streaming_latency.py` example commands and `inference_debug.md`).

`preprocess.sh` (lines 40–55) has placeholders left in place:
```bash
CONDA_ENV_NAME="sapc2"                       ### TODO: change to your conda env name
DATA_ROOT="/path/to/data"                    ### TODO: change to your data root
PROJ_ROOT="/path/to/SAPC-template"           ### TODO: change to your project root
```
Confirms preprocess.sh has not been run on this machine.

### Local manifests

```text
       101 dev100_bundle/Dev_100.csv
       101 dev100_bundle/Dev_100_local.csv
       124 dev100_bundle/Dev_streaming.csv         (same as dev_streaming_bundle/Dev_streaming.csv)
       124 dev_streaming_bundle/Dev_streaming.csv
       124 dev_streaming_bundle/Dev_streaming_local.csv
```
Each is N+1 lines (header + rows). Effective utterance counts:
- Dev_100: **100 utts** (all ALS speakers, 1 unique `speaker` value)
- Dev_streaming: **123 utts** across multiple etiologies

### Audio duration totals (computed from manifest `duration` column)

```text
Dev_100         : 1138.59 s  ≈  18.98 min
Dev_streaming   :  778.79 s  ≈  12.98 min
```

### Etiology breakdown

Dev_100 (100 utts):
```text
 100 ALS
```
Single etiology — careful, this is NOT representative of the real test set.

Dev_streaming (123 utts):
```text
  28 ALS
  28 Cerebral Palsy
  20 Down Syndrome
  35 Parkinson's Disease
  12 Stroke
```

### Speaker counts

Dev_100: UNKNOWN exact count — column 2 is `speaker` (UUID); a quick count
shows the manifest column suggests `1` unique value but the audio filenames
contain many distinct UUIDs, so the column may actually be a single
representative speaker UUID. UNKNOWN — need user input — the actual SAPC1
test1.tsv / test2.tsv may have many more speakers. Dev_streaming row 1 shows
multiple unique speaker UUIDs (123 rows / many UUIDs).

### Train / Test manifests on disk

None present locally. The official `Train.csv`, `Test1.csv`, `Test2.csv` do
NOT exist anywhere under this repo's working directory.
- `inference_debug.md:140` references `/workspace/SAPC2/manifest/Train.csv` —
  "336k utterances" mentioned, "743h dysarthric speech" mentioned in same file.
- `finetuning/finetune.py` expects `<data-root>/manifest/Train.csv` and
  `<data-root>/manifest/Dev.csv` (see prepare_sapc_lhotse.py).

**Train: ~336k utts / ~743 h** per `inference_debug.md`; **Dev: 47,929 utts**
per the ledger entry for whisper-large-v2; **Test1: 10,500 utts** per the
sherpa_zipformer Test1 entry. (These last two numbers may be off by one row
because of headers/footers.)

### Preprocessing artifacts present locally

- Normalized refs as SCTK `.trn`: e.g. `dev100_bundle/manifest/ref1.Dev_100_local.norm.trn`,
  `ref2.Dev_100_local.norm.trn` (two refs per utt — with and without disfluencies).
- SCTK alignment SGML in `dev100_bundle/eval/sctk/` and `dev_streaming_bundle/eval/sctk/`.
- The Dev_streaming manifest has MFA columns: `mfa_speech_start`,
  `mfa_speech_end`, `vad_speech_start`, `vad_speech_end`, `abs_vad_end_minus_duration`.
  This is the streaming subset built by `preprocess.sh` stage 4.

---

## 6. Models on disk

### Inside the repo

```text
4.0 GB   track2_starting_kit/parakeet_ctc/weights/final.nemo
                — Fine-tuned NeMo Parakeet-CTC 1.06B (Track 1 SAPC2 submission)
                  source: scott-morgan-foundation/sapc2-track1-parakeet-ctc
                  Architecture: FastConformer encoder + ConvASRDecoder (CTC),
                  trained with att_context_size=[-1,-1] (full attention, non-streaming).
                  Used in track2_starting_kit/parakeet_ctc/ via batch-accumulation streaming.
 250 MB   track2_starting_kit/sherpa_zipformer/weights/standard/encoder.onnx
                — sherpa-onnx Zipformer "standard" (~70M params, LibriSpeech)
                  csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26
 2.0 MB   track2_starting_kit/sherpa_zipformer/weights/standard/decoder.onnx
 1.0 MB   track2_starting_kit/sherpa_zipformer/weights/standard/joiner.onnx
 245 KB   track2_starting_kit/sherpa_zipformer/weights/standard/bpe.model
 5.0 KB   track2_starting_kit/sherpa_zipformer/weights/standard/tokens.txt
  67 MB   track2_starting_kit/sherpa_zipformer/weights/kroko/encoder.onnx
                — Kroko quantized streaming Zipformer2 (edge-optimised)
                  csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06
 617 KB   track2_starting_kit/sherpa_zipformer/weights/kroko/decoder.onnx
 337 KB   track2_starting_kit/sherpa_zipformer/weights/kroko/joiner.onnx
 6.3 KB   track2_starting_kit/sherpa_zipformer/weights/kroko/tokens.txt
  91 KB   track2_starting_kit/sherpa_zipformer/weights/kroko/kroko_onnx_weights.json
 203 MB   track2_starting_kit/wenet_u2pp/weights/final.zip
                — WeNet U2++ Conformer (Wenet 2.0 pretrained)
                  contains training yaml + units.txt
```

Other kits with venvs (the model is cached inside the venv when
`from_pretrained` runs, not in `weights/`):
```text
1.4 GB   track2_starting_kit/qwen3_asr/venv          (Qwen3-ASR-1.7B inside the venv's HF cache)
 583 MB   track2_starting_kit/gigaam/venv
 528 MB   track2_starting_kit/wenet_u2pp/venv
 168 MB   track2_starting_kit/moonshine/venv         (moonshine-tiny/small)
```

`nemotron_streaming/` has no committed weights and no venv yet on this checkout;
its setup.sh downloads `nvidia/nemotron-speech-streaming-en-0.6b` revision
`ac0580bb7d3d6e39c4361db6afe28db9211793e4` into a venv-local HF cache when
run on the pod.

### Nemotron Speech Streaming presence

**`nvidia/nemotron-speech-streaming-en-0.6b` is NOT present on this Macbook.**
- No `.nemo` file in the repo (only one .nemo file exists: `parakeet_ctc/weights/final.nemo`).
- No ONNX export of Nemotron in the repo.
- No HF cache for it under this user's home.
- The setup script ([track2_starting_kit/nemotron_streaming/setup.sh:14](track2_starting_kit/nemotron_streaming/setup.sh#L14))
  downloads it into a kit-local venv on first run (RunPod).
- The danielbodart int8-dynamic ONNX export referenced in `NEMO_LATENCY_PLAN.md`
  and `NEMOTRON_PLAN.md` has NOT been downloaded locally.

### Parakeet / Zipformer / Whisper / Canary catalog

- **Parakeet**: 1× fine-tuned `parakeet_ctc/weights/final.nemo` (4.06 GB, Track 1
  SAPC submission). Phaedrus children's-speech checkpoint
  (`Phaedrus33/nemotron-speech-streaming-children-v17`) tested zero-shot on
  RunPod (CER 25.48), not on disk locally.
- **Zipformer**: sherpa-onnx ONNX exports for `standard` (250 MB enc) and
  `kroko` (67 MB enc) under `track2_starting_kit/sherpa_zipformer/weights/`.
  `streaming_zipformer/` (icefall, PyTorch) has NO `weights/` dir locally —
  the icefall epoch-30.pt is downloaded on the pod by `finetuning/setup_finetune.sh`.
- **Whisper**: not on disk locally. `faster_whisper/` kit has model code but
  no checkpoint; whisper-large-v2 was run zero-shot on RunPod (GPU).
- **Canary**: only in `track1_starting_kit/canary_qwen/` as a 1.6 KB zip
  scaffold; no weights.

UNKNOWN — need user input — what is on the RunPod persistent volume. The
`inference_debug.md` listing implies the pod has the icefall `epoch-30.pt`,
fine-tuned `epoch-1.pt`, and exported ONNX files under `/workspace/finetune/`.

---

## 7. Training / fine-tuning runs

### Training code in the repo

- [finetuning/finetune.py](finetuning/finetune.py) (18,160 bytes) — icefall
  Zipformer-Transducer finetuning, single-GPU, lhotse data pipeline.
- [finetuning/prepare_sapc_lhotse.py](finetuning/prepare_sapc_lhotse.py) —
  CSV → lhotse CutSet converter.
- [finetuning/setup_finetune.sh](finetuning/setup_finetune.sh) — installs
  k2, kaldifeat, icefall, downloads pretrained `epoch-30.pt`.
- [finetuning/export_onnx.sh](finetuning/export_onnx.sh) — averages last N
  checkpoints, exports encoder/decoder/joiner ONNX files into
  `track2_starting_kit/sherpa_zipformer/weights/standard/`.
- [finetuning/finetune_config.yaml](finetuning/finetune_config.yaml) —
  hyperparams (base_lr 5e-4, num_epochs 20, max_duration 300s, chunk_size 16,
  left_context_frames 128, causal=true).
- [finetuning/tests/](finetuning/tests/) — `test_data.py`, `test_env.py`,
  `test_model.py` + `run_tests.sh`.
- [run_nemo_inference.py](run_nemo_inference.py) — NeMo `.transcribe()`
  batch harness for the BENCHMARK_PLAN Phase 2.
- [run_nemo_streaming_latency.py](run_nemo_streaming_latency.py) — cache-aware
  streaming harness with real-time sleep loop.
- [run_onnx_streaming_latency.py](run_onnx_streaming_latency.py) — same loop
  for the danielbodart Nemotron ONNX layout.

### What's been trained

Per [inference_debug.md](inference_debug.md) and the ledger:

1. **icefall streaming Zipformer2** (66M params, causal, chunk-size 16,
   left-context 128) was finetuned **1 epoch** on the SAP train set
   (336k utts / 743 h) starting from LibriSpeech `epoch-30.pt`. Training
   loss: 7.37 → 0.035 (suspiciously fast). The finetuned dev100 numbers
   (CER 7.75 / WER 11.80) are recorded but UNVERIFIED — the export
   pipeline produced empty outputs through sherpa-onnx AND through direct
   PyTorch/onnxruntime inference; root cause was identified
   (`model.encoder_embed` subsampling skipped) but the doc ends "Identified
   but not yet tested". Status: **pipeline is broken / numbers should not be
   trusted**.

2. **Nemotron Speech Streaming 0.6B zero-shot** on Dev_streaming (RunPod CPU,
   4 threads, att_context_size=[70,13]): CER 17.81, TTFT P50 1895 ms,
   RTF 1.117 — **fails the RTF gate**. (Logged 2026-05-16, commit b40d653.)

3. **Phaedrus children's-speech Nemotron** zero-shot: CER 25.48 on
   Dev_streaming — no transfer to dysarthric speech, dropped from plan.

No SAP-specific finetuning of Nemotron has run yet. Per `NEMOTRON_PLAN.md` (on
the `claude/amazing-edison-c7d764` branch) that is the next major work item.

### Run outputs / logs

- [local_results/zipformer_standard_dev_streaming.log](local_results/zipformer_standard_dev_streaming.log)
  (22 lines) — single local sherpa-onnx zipformer run on a RunPod pod, output
  copied here. Mentions manifest path `/workspace/SAPC2/manifest/Dev_streaming.csv`.
- `track2_starting_kit/local_results/*.partial.json` + `.csv` — predictions
  for all 8 kits on dev100 batch + streaming modes (Mar 22 timestamps).
- No W&B artifact directories, no `wandb/` folder, no tensorboard
  `events.out.tfevents.*` checked in. UNKNOWN — need user input — whether any
  W&B run IDs exist; if they do, they're on the pod.

### Last training run

The most recent thing logged as a training/eval is the 2026-05-16 Nemotron
zero-shot benchmarks (commit b40d653). The **finetune** itself (1 epoch icefall
Zipformer2 on SAP) is dated by `inference_debug.md` to early April 2026 (file
mtime Apr 5). No training-time loss curves are checked in, just the inference
debug doc.

---

## 8. Hardware & environment

**This local machine (not the eval target):**

```text
GPU:    none — Apple Silicon (M3)
CPU:    Apple M3 — ncpu=8, physicalcpu=8, logicalcpu=8
RAM:    8 GB (hw.memsize=8589934592)
Disk:   460 GB total, 55 GB available  (/dev/disk3s1s1)
Python: 3.13.0 — /Library/Frameworks/Python.framework/Versions/3.13/bin/python3
Active env: none (no conda or venv active in shell)
```

Library versions on the system Python:
```text
torch         2.6.0
torchaudio    2.6.0
onnxruntime   1.24.4
onnx          1.20.1
```
`nemo_toolkit` is NOT installed locally — `pip show nemo_toolkit` returned no
output, and there's no global conda env named `sapc2`.

**The eval target (per the official spec, `Track2 (Streaming ASR Track).md`):**
- Docker: `xiuwenz2/sapc2-runtime:latest`
- Pre-installed: PyTorch 2.5.0+cu124, torchaudio, torchvision, Python 3.11
- GPU: **none — Track 2 is evaluated exclusively on CPUs**
- Time limit: 15,000 seconds per submission
- `setup.sh` runs before model load; `requirements.txt` auto-installed after.

UNKNOWN — need user input — the actual core count and CPU generation of the
SAPC2 eval VM. `NEMO_LATENCY_PLAN.md:3` flags this as the single biggest risk
to the project: if the eval CPU is <4 cores or significantly slower than the
RunPod pod, none of the local TTFT/RTF numbers transfer.

**The RunPod pod used for training/eval:** UNKNOWN — need user input — exact
CPU/GPU SKU. Past entries mention "RunPod CPU 4 threads" and a separate GPU
pod for finetuning.

---

## 9. Submission history

UNKNOWN — no local submission history beyond the experiments ledger.

There is no `submission_log.txt`, no dated `submission-v*` tags, and the
`*.zip` files in the repo are tiny (1–7 KB starter-kit scaffolds, not packed
submissions):
```text
1605   track1_starting_kit/canary_qwen.zip
1665   track1_starting_kit/parakeet.zip
1077   track1_starting_kit/whisper.zip
7331   track2_starting_kit/streaming_zipformer.zip
```

The only "submission" recorded anywhere is the single Test1 entry for
`sherpa_zipformer/standard` (Section 4 above) on 2026-03-01. There may be
more submissions on Codabench that never made it into the ledger.

UNKNOWN — need user input — how many submissions have been made, when, and
what code revision each contained.

---

## 10. Open questions and known issues

### TODO/FIXME/XXX in code

```text
(no matches outside of venvs/__pycache__ and worktree copies)
```
The repo's Python code itself is free of TODO comments. The shell scripts
have a few `### TODO: change to your X` markers in `preprocess.sh:43-45`
(see Section 5 — those `DATA_ROOT`/`PROJ_ROOT`/`CONDA_ENV_NAME` placeholders
were never customized).

### Known broken / suspicious

1. **icefall Zipformer2 finetune export is broken** — see Section 7. The
   1-epoch finetuned checkpoint produces empty strings through sherpa-onnx
   and through direct ONNX/PyTorch inference. Root cause identified but never
   re-tested. The 7.75 CER / 11.80 WER finetuned dev100 numbers may be
   fabrications of a buggy decode path. [inference_debug.md](inference_debug.md).

2. **Nemotron zero-shot RTF 1.117 fails the Track 2 gate.** This is the
   currently-most-recent run. The remediation plan (`NEMO_LATENCY_PLAN.md`,
   `NEMOTRON_PLAN.md`) is to switch to ONNX int8/int4 and/or shrink
   att_context_size from [70,13] to [70,1] (160 ms chunk). Neither has been
   attempted yet.

3. **streaming_pruned_stateless7 kit was deleted** in commit 02aab0f
   ("remove old files") but its dev100 result is still the second-best
   recorded WER (13.22% libri_giga). UNKNOWN — need user input — whether
   the deletion was intentional and whether the model is worth resurrecting.

4. **`preprocess.sh` placeholders never customised** — implies no one has
   run preprocess from this checkout. The dev bundles must have been
   produced elsewhere (RunPod) and copied here.

5. **inference_debug.md ends mid-investigation** — "Identified but not yet
   tested". UNKNOWN — need user input — whether the encoder_embed fix was
   tried later and what the result was.

### From commit messages / notes

- `b40d653 Chnages` (May 16) — empty-ish commit message, content per `git
  show` is the Nemotron benchmark runs.
- No CI files (`.github/workflows/` absent), no Makefile, no lint config.
- `inference_debug.md` is the closest thing to a known-issues doc.

---

## 11. Dead-code / abandoned-experiment candidates

These are guesses for the user to review — **nothing has been deleted**.

1. **`track2_starting_kit/streaming_zipformer/`** (icefall PyTorch path).
   Code is intact (model.py 16 KB, last touched Mar 12) but no `weights/`
   dir present locally and no `icefall/` clone. Its purpose (PyTorch
   inference) is dominated by `sherpa_zipformer/` (ONNX, faster on CPU).
   Likely superseded; kept around because `finetuning/export_onnx.sh` reuses
   the same checkpoint format. Evidence: dir mtime Apr 7, no result CSV
   under that exact name in local_results, README in `finetuning/` says
   "ONNX is the recommended submission vehicle".

2. **`track2_starting_kit/local_results/streaming_pruned_stateless7_*`**.
   The matching kit was deleted (commit 02aab0f, Apr 10) but the result
   CSVs and `pruned_stateless7_dev100_comparison.md` remain. The
   `streaming_pruned_stateless7_librispeech/` dir is empty (placeholder for
   a planned librispeech-only decode that never ran).

3. **`track2_starting_kit/test_moonshine*.csv` + `test_moonshine*_partial.json`,
   `test_output.csv`, `test_partial.json`** in `track2_starting_kit/` root
   (not under `local_results/`). All mtime 2026-03-21. Look like leftover
   smoke-test outputs from the moonshine experiments. Five of these files
   total ~640 KB.

4. **Track-1 zips** (`track1_starting_kit/*.zip`, 1–2 KB each). The
   Track 1 starting kit is unrelated to your Track 2 work and may be stale.

5. **`my-work` branch** (Mar 19, 2 months stale) and **`claude/dazzling-kalam`**
   (Mar 12, 2.5 months stale, only modifies README). Both look abandoned.

6. **`watcher.log`** at repo root (3.9 KB, Mar 21) — RunPod watcher log,
   non-functional artifact.

7. **`Claude.pdf`** at repo root (3.2 MB) — background PDF; probably
   superseded by the markdown plans.

8. **`emformer_rnnt/`, `faster_whisper/`, `gigaam/`, `moonshine/`,
   `wenet_u2pp/`, `qwen3_asr/`** — six other model kits, all benchmarked
   in March/April and none of which are competitive on dev100 (WER
   24–44%, see Section 4). Disk cost ~3 GB combined (venvs + wenet weights).
   The Nemotron and sherpa_zipformer/kroko paths are the clear winners.
   Candidates for archival once a path is committed.

9. **`inference_debug.md`** — half-finished investigation. Useful as a
   warning but stale (Apr 5).

10. **`finetuning/data/` and `finetuning/pretrained/`** — both are empty
    dirs locally; would only be populated on RunPod by `setup_finetune.sh`
    and `prepare_sapc_lhotse.py`.

---

## 12. Questions for the user

These are the things I couldn't determine from the repo and that the other
Claude instance is likely to need before strategizing.

1. **Codabench leaderboard state.** What is currently submitted? The ledger
   only has the 2026-03-01 `sherpa_zipformer/standard` Test1 entry (52.77 WER /
   34.59 CER). Has anything been submitted since? Where do we sit in the
   public rankings, and how many submissions are left in the budget?

2. **Eval CPU spec.** `NEMO_LATENCY_PLAN.md` flags this as the biggest
   unknown. What CPU does the SAPC2 eval VM actually have — core count,
   generation (does it have AVX2 / VNNI / AMX)? Without this, no
   RTF/TTFT number from RunPod transfers.

3. **Strategic direction — Nemotron or icefall Zipformer2 (kroko)?**
   - Kroko (Zipformer2 ONNX) is the local best on Dev100 (CER 5.87 zero-shot)
     but has zero verified Test1 number and no TTFT/RTF measurement.
   - Nemotron has a working streaming harness but fails the RTF gate by 11.7%
     at the default chunk size, and needs SAP finetuning before competitive
     accuracy. The plan exists; nothing has been executed beyond zero-shot.

4. **Finetuned icefall checkpoint — is the export-broken status resolved?**
   `inference_debug.md` ends mid-debug. Was the `encoder_embed` subsampling
   fix tried, and does the 1-epoch finetuned model actually produce text?
   If yes, what are the real CER/WER numbers (the 7.75/11.80 in the ledger
   may be from a broken decode and are flagged unverified).

5. **`streaming_pruned_stateless7` libri_giga model** — the dev100 WER 13.22%
   number is the second-best of any model on dev100 but the kit code was
   deleted (commit 02aab0f). Where are the weights, and is this path worth
   resurrecting?

6. **RunPod persistent state.** What lives on the pod's persistent volume?
   Specifically:
   - `/workspace/SAPC2/manifest/Train.csv` and `Dev.csv`, `Test1.csv`,
     `Test2.csv` — what are the exact row counts?
   - `/workspace/finetune/exp/standard/epoch-1.pt` (finetuned Zipformer2) —
     still there? With its ONNX export?
   - Any W&B run IDs or tensorboard event files from the finetune?

7. **Submission size limit.** `NEMOTRON_PLAN.md:88` mentions checking it
   before Phase 5 ("int8 ~876 MB, int4 ~0.7 GB"). What's the actual cap?

8. **Branch merging plan.** `claude/amazing-edison-c7d764` has the
   `NEMOTRON_PLAN.md`, `CLAUDE.md`, and the entire
   `track2_starting_kit/experiments/` ledger; none of that is on `main`.
   Is the intent to merge that branch, or treat it as exploratory?

9. **Submission deadline.** When is the SAPC2 Track 2 submission window
   closing? What's the practical time budget remaining for finetuning runs
   on RunPod GPU?

10. **The `sherpa_zipformer` variant in the actual Test1 submission** — was
    it `standard` (LibriSpeech) or `kroko` (Zipformer2)? The ledger says
    "icefall librispeech streaming zipformer" but `kroko` was also
    available by Apr; the difference matters for what improvement we
    expect from finetuning.
