# nemotron_streaming_nemo_free_safesetup — SAPC2 Track 2 submission

**Derivative of `nemotron_streaming_nemo_free/` (v6 / SHA `afd93afe…`)
with exactly one change:** the setup.sh smoke test is wrapped in
`set +e` so a smoke-test failure is non-fatal and setup.sh always
exits 0. Rationale: v6 returned the same `Detected splits: []`
failure as prior submissions and Codabench produced no visible
ingestion logs — one plausible reason is that setup.sh exited
non-zero (from a smoke-test failure we can't reproduce locally) and
ingestion was never spawned. Making the smoke test non-fatal removes
that failure path.

---

# Original kit documentation (model.py, config.yaml, mel-equivalence
# notes — all unchanged from v6)


**Nemo-free production candidate for the int8-static Nemotron ASR.**
Replaces NeMo's `AudioToMelSpectrogramPreprocessor` with a hand-rolled
torch + numpy reimplementation (`MinimalMelPreprocessor` in
[model.py](model.py)) that was validated byte-equivalent to NeMo at
max abs diff `1.9e-6` (well under the `1e-4` acceptance threshold).

## Why no NeMo

Codabench ingestion logs showed:

```
ValueError: numpy.dtype size changed, may indicate binary
incompatibility. Expected 96 from C header, got 88 from PyObject
```

This is a numpy ABI conflict. The lightning wheel (transitively
pulled in via `nemo_toolkit[asr]`) was compiled against numpy 2.x
(dtype size 96), but pip resolved numpy to 1.26.4 (dtype size 88)
because numba 0.65.1 (transitively pulled in via librosa via nemo)
has a strict `numpy<2` constraint. The mismatch crashes the C-level
import.

Removing the `numpy<2` pin from `setup.sh` (v5 attempt) didn't help —
the transitive numba constraint still forced the downgrade. The only
fix that closes the ABI mismatch entirely is to **drop NeMo**, which
removes lightning + numba + librosa from the dependency tree at the
same time.

## Files

| File | Purpose |
|---|---|
| `model.py` | 5-method `Model` class. SETUP_VERIFY probes torch + onnxruntime only (no nemo). Mel preprocessor inline (`MinimalMelPreprocessor`). int8-static ONNX encoder + fp32 decoder. |
| `config.yaml` | num_threads + weights paths. |
| `setup.sh` | Pip-installs `omegaconf` + `huggingface_hub` + `onnxruntime` + `onnx` globally into base Python. Pins torch to the system version via constraint file. Downloads danielbodart's prebuilt ONNX. End-of-script: logs `pip list`, runs an import check, and **explicitly verifies that nemo / lightning / numba / librosa are NOT installed**. |
| `weights/` (created by `setup.sh`) | encoder_model.onnx (int8), decoder_model.onnx (fp32), tokens.txt — ~880 MB total. |

## Validated against NeMo

A one-off test in
[/tmp/preproc_dev/test_equiv.py](../../scripts/audit/preproc_byte_equivalence/test_equiv.py)
(also archived in this repo under `scripts/audit/preproc_byte_equivalence/`)
ran identical audio through both preprocessors:

| File | shape match | max abs diff | mean abs diff | verdict |
|---|---|---|---|---|
| `0a71bb2c-…_1052_4303.wav` (3.49 s) | ✓ (1, 128, 350) | 1.9e-6 | 1.8e-8 | PASS |
| `0a71bb2c-…_1053_4303.wav` (4.73 s) | ✓ (1, 128, 474) | 1.9e-6 | 1.8e-8 | PASS |
| `0a71bb2c-…_1054_4303.wav` (3.57 s) | ✓ (1, 128, 358) | 1.9e-6 | 1.3e-8 | PASS |

Implementation notes:

- The mel filterbank is computed in numpy fp64 at module load to
  match librosa's precision exactly. torchaudio's
  `melscale_fbanks(mel_scale='slaney', norm='slaney')` computes
  everything in fp32 and diverges by ~1.4e-4 in the high mel bins
  (just over the 1e-4 threshold).
- All NeMo internal defaults are preserved: `preemph=0.97`,
  `mag_power=2.0`, `log_zero_guard_value=2**-24`, slaney mel scale +
  slaney area normalization, hann window with `periodic=False`,
  STFT `center=True` and `pad_mode="constant"`.
- The seq_len output formula `floor(L / hop)` matches NeMo's
  `get_seq_len` for the `exact_pad=False` + `center=True` case
  (NeMo's general formula reduces to this when `n_fft` is even).

## Reference numbers (unchanged — same model, same decode)

| Set | Scorer | CER% | WER% | Source |
|---|---|---|---|---|
| Dev_10k (10,521 utts) | sclite, dual-ref MIN | **21.59** | **27.46** | `f160e39` |

The mel preprocessor is byte-equivalent to NeMo at fp32-noise scale,
so these numbers carry forward. Expected on Test1: ~20–25 CER.
