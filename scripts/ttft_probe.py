#!/usr/bin/env python3
"""Per-chunk wrapper compute probe (single-process, controlled torch threads).

Answers: is the parakeet_realtime_ft wrapper's per-100ms-chunk CPU compute under the
100ms real-time budget on a *properly threaded* host? The pod visibly has nproc=128 but
a ~13.6-CPU cgroup quota, so torch's default 128 threads oversubscribe → inflates the
streaming pass's wall clock (and thus TTFT). Here we fix torch threads and measure the
true per-chunk cost, decomposed into feature-extract vs model-step.

    python3 ttft_probe.py <N_THREADS>
"""
import sys, time, importlib.util
import numpy as np, soundfile as sf, torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
torch.set_num_threads(N)
torch.set_num_interop_threads(1)

MP = "/workspace/SAPC-template/track2_starting_kit/parakeet_realtime_ft/model.py"
spec = importlib.util.spec_from_file_location("pk", MP)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
m = mod.Model()

# time-decompose: wrap _extract_features and _stream_step
tf = {"feat": 0.0, "step": 0.0}
_of, _os = m._extract_features, m._stream_step
def tfeat(raw):
    s = time.perf_counter(); r = _of(raw); tf["feat"] += time.perf_counter() - s; return r
def tstep(*a, **k):
    s = time.perf_counter(); r = _os(*a, **k); tf["step"] += time.perf_counter() - s; return r
m._extract_features = tfeat
m._stream_step = tstep

import csv
rows = list(csv.DictReader(open("/workspace/parakeet_ft/dev20_streaming.csv")))
# warm-up utterance (first pass triggers any lazy init), then measure on the next 3
def stream(rec, measure):
    wav = "/workspace/SAPC2/" + rec["audio_filepath"]
    audio, _ = sf.read(wav)
    if audio.ndim > 1: audio = audio.mean(1)
    audio = audio.astype(np.float32)
    m.reset()
    per_chunk = []
    for off in range(0, len(audio), 1600):
        s = time.perf_counter()
        m.accept_chunk(audio[off:off + 1600])
        per_chunk.append((time.perf_counter() - s) * 1000)
    s = time.perf_counter(); m.input_finished(); fin_ms = (time.perf_counter() - s) * 1000
    return per_chunk, fin_ms, len(audio) / 16000.0

stream(rows[0], False)  # warm-up
allc = []
for r in rows[1:4]:
    tf["feat"] = tf["step"] = 0.0
    pc, fin, dur = stream(r, True)
    allc += pc
    arr = np.array(pc)
    print(f"utt dur={dur:.1f}s chunks={len(pc)} | per-chunk ms: median={np.median(arr):.1f} "
          f"p90={np.percentile(arr,90):.1f} max={arr.max():.1f} | >100ms: {int((arr>100).sum())}/{len(pc)} "
          f"| feat={tf['feat']*1000:.0f}ms step={tf['step']*1000:.0f}ms final={fin:.0f}ms")
a = np.array(allc)
print(f"THREADS={N}  ALL per-chunk ms: median={np.median(a):.1f} p90={np.percentile(a,90):.1f} "
      f"mean={a.mean():.1f} | frac>100ms={100*(a>100).mean():.0f}%")
print("PROBE_DONE_N%d" % N)
