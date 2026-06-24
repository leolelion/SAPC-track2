#!/usr/bin/env python3
# "Beam/blank-recovery on empties": re-decode utterances with a BLANK-LOGIT PENALTY sweep.
# Greedy RNN-T picks blank every frame -> empty. If subtracting a penalty from the blank logit
# makes the CORRECT text emerge, the fix is a cheap decode change (no finetuning). If only garbage
# emerges, the acoustics carry no usable signal for this model => domain failure.
# Works by intercepting the decoder ONNX output and lowering the blank logit before the model's
# own argmax/break logic runs (faithful: reuses the real decode loop).
# Usage (pod): cd /workspace/finetune/nemo_submission
#   python3 /workspace/nemotron_altdecode_empties.py <manifest.csv> <data_root> <id> [<id> ...]
import sys, os, csv, wave
import numpy as np
SUB = "/workspace/finetune/nemo_submission"
sys.path.insert(0, SUB); os.chdir(SUB)
import model as M

man, data_root = sys.argv[1], sys.argv[2]
ids = sys.argv[3:]
rows = {r["id"]: r for r in csv.DictReader(open(man))}
BLANK = M.BLANK_ID
PENALTIES = [0.0, 3.0, 6.0, 10.0]

def read_wave(p):
    with wave.open(p, "rb") as f:
        s = f.readframes(f.getnframes())
    return np.frombuffer(s, dtype=np.int16).astype(np.float32) / 32768.0

def cer(h, r):
    h, r = list(h), list(r)
    if not r: return 0.0 if not h else 1.0
    dp = list(range(len(r)+1))
    for i in range(1, len(h)+1):
        p = dp[0]; dp[0] = i
        for j in range(1, len(r)+1):
            c = dp[j]; dp[j] = min(dp[j]+1, dp[j-1]+1, p+(h[i-1]!=r[j-1])); p = c
    return min(1.0, dp[len(r)]/len(r))

m = M.Model()
orig = m._dec_sess.run
PEN = [0.0]                       # mutable closure for current penalty
def wrapped(outnames, feed, _orig=orig):
    out = _orig(outnames, feed)
    if PEN[0]:
        out[0][..., BLANK] -= PEN[0]
    return out
m._dec_sess.run = wrapped

for uid in ids:
    r = rows[uid]; path = os.path.join(data_root, r["audio_filepath"])
    ref = r.get("norm_text_without_disfluency", "")
    samples = read_wave(path)
    print(f"\nID {uid[:12]} dur={r['duration']}s eti={r['etiology']}  ref={ref[:55]!r}")
    for pen in PENALTIES:
        PEN[0] = pen
        m.reset()
        for i in range(0, len(samples), 1600):
            m.accept_chunk(samples[i:i+1600])
        hyp = m.input_finished()
        c = cer(hyp.lower(), ref.lower())
        print(f"  blank_penalty={pen:4.1f}: CER={100*c:5.1f}%  hyp={hyp[:55]!r}")
