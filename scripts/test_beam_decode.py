"""Local correctness test for the RNN-T beam added to parakeet_onnx/model.py.

No weights, no onnxruntime, no SAP data: a fake decoder session returns scripted logits
keyed by (encoder frame, last emitted token), so both decode paths are driven over a
known joint and their outputs can be compared against hand-computed expectations.
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/o/Downloads/SAPC-template/track2_starting_kit/parakeet_onnx")
import model as M

BLANK = 4
VOCAB = ["▁a", "▁b", "▁c", "▁d", "<b>"]


class FakeDec:
    """Returns LOGITS[(frame, last_token)], else a blank-dominant row. The state output
    is a 1-element array carrying the number of emissions consumed so far, which lets the
    test assert that the beam threads prediction state per hypothesis and never leaks it."""

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def run(self, _out_names, feeds):
        self.calls += 1
        f = int(feeds["encoder_outputs"][0, 0, 0])
        t = int(feeds["targets"][0, 0])
        row = self.table.get((f, t), [-10.0, -10.0, -10.0, -10.0, 5.0])
        logits = np.array(row, dtype=np.float32).reshape(1, 1, 1, 5)
        state = feeds["state_in"] + 1.0
        return [logits, state]


def make_model(table, **over):
    m = object.__new__(M.Model)
    m._blank_id = BLANK
    m._max_symbols = 10
    m._blank_penalty = 0.0
    m._vocab = VOCAB
    m._dec = FakeDec(table)
    m._dec_out_names = ["outputs", "state"]
    m._dec_state_in = ["state_in"]
    m._dec_state_out = ["state"]
    m._target_length = np.array([1], dtype=np.int32)
    m._beam = 1
    m._beam_max_symbols = 2
    m._beam_expand = 2
    m._beam_prune_logp = -12.0
    m._beam_length_norm = "none"
    for k, v in over.items():
        setattr(m, "_" + k, v)
    m._raw_buf = np.zeros(0, dtype=np.float32)
    m._tokens = []
    m._hyp_text = ""
    m._last_emitted = ""
    m._dec_states = [np.zeros((1,), dtype=np.float32)]
    m._last_token = np.array([[BLANK]], dtype=np.int32)
    m._beam_hyps = (
        [M._BeamHyp(0.0, (), tuple(m._dec_states), m._last_token)] if m._beam > 1 else None
    )
    return m


def encoded(n_frames):
    e = np.zeros((1, 1, n_frames), dtype=np.float32)
    e[0, 0, :] = np.arange(n_frames)
    return e


def run(m, n_frames):
    enc = encoded(n_frames)
    if m._beam > 1:
        m._decode_frames_beam(enc, 0, n_frames)
    else:
        m._decode_frames(enc, 0, n_frames)
    return m._hyp_text


# ---------------------------------------------------------------- test 1
# Sharply peaked joint: there is one dominant path, so a beam must find exactly what
# greedy finds. This is the regression guard -- if the beam's state threading or blank
# bookkeeping is wrong, it diverges here even though no search is required.
PEAKED = {
    (0, BLANK): [8.0, -10, -10, -10, -10],   # emit 'a'
    (0, 0): [-10, -10, -10, -10, 8.0],       # then blank -> frame 1
    (1, 0): [-10, 9.0, -10, -10, -10],       # emit 'b'
    (1, 1): [-10, -10, -10, -10, 8.0],
    (2, 1): [-10, -10, 9.0, -10, -10],       # emit 'c'
    (2, 2): [-10, -10, -10, -10, 8.0],
}
g = make_model(PEAKED)
b = make_model(PEAKED, beam=3)
gt, bt = run(g, 3), run(b, 3)
print(f"test1 peaked   greedy={gt!r}  beam3={bt!r}  dec_calls greedy={g._dec.calls} beam={b._dec.calls}")
assert gt == "a b c", gt
assert bt == gt, f"beam diverged from greedy on a peaked joint: {bt!r} != {gt!r}"
print("test1 PASS: beam reproduces greedy when no search is needed")

# ---------------------------------------------------------------- test 2
# The deletion case. At frame 0 blank narrowly beats 'a', so greedy commits to blank and
# can never revisit it. The emitting path is then rewarded at frame 1 by more than the
# margin it gave up, so the higher-probability path is 'a b' and only a beam can reach it.
DELETION = {
    (0, BLANK): [0.9, -10, -10, -10, 1.0],   # blank wins by 0.1 -> greedy deletes here
    (0, 0): [-10, -10, -10, -10, 5.0],
    (1, BLANK): [1.5, -10, -10, -10, 2.0],   # staying empty keeps costing
    (1, 0): [-10, 8.0, -10, -10, 0.0],       # having emitted 'a' pays off big
    (1, 1): [-10, -10, -10, -10, 5.0],
}
g = make_model(DELETION)
b = make_model(DELETION, beam=2)
gt, bt = run(g, 2), run(b, 2)
print(f"test2 deletion greedy={gt!r}  beam2={bt!r}  dec_calls greedy={g._dec.calls} beam={b._dec.calls}")
assert gt == "", f"expected greedy to delete both words, got {gt!r}"
assert bt == "a b", f"expected beam to recover 'a b', got {bt!r}"
print("test2 PASS: beam recovers a word greedy deleted")

# ---------------------------------------------------------------- test 3
# Emission budget: beam_max_symbols caps emissions PER FRAME, and the final depth is a
# blank-only pass so every surviving hypothesis has paid for its blank.
# NB blank cost must DIFFER by state. Every path ends a frame with exactly one blank, so
# a table with a uniform blank logit makes all path scores tie and the assertion below
# would be testing tie-break order, not the budget.
BURST = {
    (0, BLANK): [9.0, -10, -10, -10, -19.0],   # emit 'a' free, blank ~ -28
    (0, 0): [-10, 9.0, -10, -10, 4.0],         # emit 'b' free, blank ~ -5.0
    (0, 1): [-10, -10, 9.0, -10, 8.9],         # blank ~ -0.74
    (0, 2): [-10, -10, -10, -10, 9.0],         # blank ~  0
}
one = make_model(BURST, beam=2, beam_max_symbols=1)
two = make_model(BURST, beam=2, beam_max_symbols=2)
t1, t2 = run(one, 1), run(two, 1)
print(f"test3 budget   max_symbols=1 -> {t1!r}   max_symbols=2 -> {t2!r}")
assert len(t1.split()) == 1, t1
assert len(t2.split()) == 2, t2
print("test3 PASS: beam_max_symbols caps emissions per frame")

# ---------------------------------------------------------------- test 4
# Length normalisation changes the FINAL ranking only. Two paths of different length are
# made near-equal in raw log-prob; 'mean' must prefer the longer one.
# Hand-computed: path 'a'   scores -0.4870 (len 1) -> mean -0.4870
#                path 'a b' scores -0.8870 (len 2) -> mean -0.4435
# so raw log-prob must pick the SHORT one and mean-normalisation the LONG one. This is
# the deletion bias in miniature: the longer path is per-token better and still loses.
LN = {
    (0, BLANK): [5.0, -20, -20, -20, -20],     # emit 'a', blank ~ -25
    (0, 0): [-20, -20, -20, -20, 5.0],         # blank ~ 0 -> frame 1
    (1, 0): [-20, -0.9, -20, -20, -0.5],       # blank -0.487 vs emit 'b' -0.887
    (1, 1): [-20, -20, -20, -20, 5.0],         # blank ~ 0
}
raw = make_model(LN, beam=3)
mean = make_model(LN, beam=3, beam_length_norm="mean")
rt, mt = run(raw, 2), run(mean, 2)
print(f"test4 lengthnorm none -> {rt!r}   mean -> {mt!r}")
assert rt == "a", f"raw log-prob should prefer the short path, got {rt!r}"
assert mt == "a b", f"mean normalisation should prefer the long path, got {mt!r}"
print("test4 PASS: length_norm=mean flips the ranking toward the longer path")

print("\nALL 4 PASS")
