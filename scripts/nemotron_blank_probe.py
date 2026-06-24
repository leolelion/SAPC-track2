#!/usr/bin/env python3
# (B) Blank-probability probe. Wraps the submission's decoder ONNX call to log, per decode
# step, the blank logit vs the best NON-blank logit. Tells us WHY an utterance is empty:
#   - blank >> best_nonblank on every frame, large stable margin  => model CONFIDENTLY predicts
#     blank = acoustic-domain "give up" (finetuning is the fix).
#   - small/erratic margins, NaNs, or saturated values            => degenerate / quantization
#     / decode-path artifact (potentially fixable without finetuning).
# Usage (pod):
#   cd /workspace/finetune/nemo_submission
#   python3 /workspace/nemotron_blank_probe.py <manifest.csv> <data_root> <id> [<id> ...]
import sys, os, csv, wave
import numpy as np
SUB = "/workspace/finetune/nemo_submission"
sys.path.insert(0, SUB)
os.chdir(SUB)
import model as M           # the submission module (defines Model, BLANK_ID, etc.)

man, data_root = sys.argv[1], sys.argv[2]
ids = sys.argv[3:]
rows = {r["id"]: r for r in csv.DictReader(open(man))}

def read_wave(p):
    with wave.open(p, "rb") as f:
        s = f.readframes(f.getnframes())
    return np.frombuffer(s, dtype=np.int16).astype(np.float32) / 32768.0

m = M.Model()
BLANK = M.BLANK_ID
for uid in ids:
    r = rows[uid]
    path = os.path.join(data_root, r["audio_filepath"])
    trace = []                       # (blank_logit, best_nonblank_logit, argmax)
    orig = m._dec_sess.run
    def wrapped(outnames, feed, _orig=orig, _tr=trace):
        out = _orig(outnames, feed)
        lg = np.asarray(out[0]).reshape(-1)          # [vocab]
        b = float(lg[BLANK]); nb = np.delete(lg, BLANK)
        _tr.append((b, float(nb.max()), int(lg.argmax())))
        return out
    m._dec_sess.run = wrapped
    m.reset()
    samples = read_wave(path)
    for i in range(0, len(samples), 1600):
        m.accept_chunk(samples[i:i+1600])
    final = m.input_finished()
    m._dec_sess.run = orig
    n = len(trace)
    if n == 0:
        print(f"\nID {uid[:12]} dur={r['duration']}s eti={r['etiology']}: NO decode steps (n=0)"); continue
    blanks = sum(1 for b, nb, a in trace if a == BLANK)
    margins = [b - nb for b, nb, a in trace]
    finite = np.isfinite([b for b,_,_ in trace] + [nb for _,nb,_ in trace]).all()
    arr = np.array(margins)
    print(f"\nID {uid[:12]} dur={r['duration']}s eti={r['etiology']} | final={final[:40]!r}")
    print(f"  decode_steps={n}  argmax==blank: {blanks}/{n} ({100*blanks/n:.0f}%)  finite_logits={finite}")
    print(f"  (blank - best_nonblank) margin: mean={arr.mean():+.2f} min={arr.min():+.2f} "
          f"max={arr.max():+.2f}  (>0 => blank wins)")
    print(f"  ref: {r.get('norm_text_without_disfluency','')[:60]!r}")
