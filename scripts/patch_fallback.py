#!/usr/bin/env python3
"""Patch a COPY of model.py to add a no-empty blank-penalty fallback.

When the normal RNN-T greedy emits nothing for an utterance (blank won at every
frame), input_finished() re-decodes the cached encoder frames with an escalating
blank penalty until a non-empty hypothesis appears. Zero effect on utts that
already produce output. Idempotent-ish: refuses to double-patch.
"""
import sys

path = sys.argv[1]
src = open(path).read()
if "_fallback_decode" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)

# 1) cache encoder frames in reset
a1 = "        self._drain_done = False\n"
assert src.count(a1) == 1, "anchor1"
src = src.replace(a1, a1 + "        self._enc_frames = []  # cached for no-empty fallback\n")

# 2) append per-chunk encoder frames (valid frames only, non-drain)
a2 = '        encoder_out = named["outputs"]\n        n_enc = int(named["encoded_lengths"][0])\n'
assert src.count(a2) == 1, "anchor2"
src = src.replace(
    a2,
    a2 + "        if not is_drain and n_enc > 0:\n"
         "            self._enc_frames.append(encoder_out[:, :, :n_enc].copy())\n",
)

# 3) input_finished: run fallback when empty
a3 = (
    '        self._run_steps(is_final=True)\n'
    '        self._run_drain()\n'
    '        self.compute_time_sec += (time.perf_counter() - t0)\n'
    '        return _strip_unk(self._last_emitted)\n'
)
assert src.count(a3) == 1, "anchor3"
src = src.replace(
    a3,
    '        self._run_steps(is_final=True)\n'
    '        self._run_drain()\n'
    '        if not self._emitted_tokens:\n'
    '            for _p in (5.0, 10.0, 20.0, 50.0, 1e9):\n'
    '                _t = self._fallback_decode(_p)\n'
    '                if _t.strip():\n'
    '                    self._last_emitted = _t\n'
    '                    break\n'
    '        self.compute_time_sec += (time.perf_counter() - t0)\n'
    '        return _strip_unk(self._last_emitted)\n'
)

# 4) add the fallback method (append before final _detokenize-of-emitted at class end:
#    insert right after _encode_and_decode by anchoring on its last line)
a4 = (
    '        if not is_drain and text != self._last_emitted:\n'
    '            self._partial_callback(text)\n'
    '        self._last_emitted = text\n'
)
assert src.count(a4) == 1, "anchor4"
method = (
    a4 +
    "\n"
    "    def _fallback_decode(self, blank_penalty: float) -> str:\n"
    '        """No-empty fallback: greedy over all cached encoder frames with a\n'
    "        penalty subtracted from the blank logit, from the (still-initial)\n"
    '        decoder state. Used only when the normal pass emitted nothing."""\n'
    "        if not self._enc_frames:\n"
    "            return \"\"\n"
    "        enc = np.concatenate(self._enc_frames, axis=2)  # [1, D, T]\n"
    "        Ttot = enc.shape[2]\n"
    "        dec_h = self._dec_h.copy()\n"
    "        dec_c = self._dec_c.copy()\n"
    "        last_token = self._last_token.copy()\n"
    "        toks = []\n"
    "        for f_idx in range(Ttot):\n"
    "            enc_frame = enc[:, :, f_idx:f_idx + 1]\n"
    "            for _sym in range(MAX_SYMBOLS_PER_FRAME):\n"
    "                dec_outs = self._dec_sess.run(\n"
    "                    None,\n"
    "                    {\n"
    '                        "encoder_outputs": enc_frame,\n'
    '                        "targets": last_token,\n'
    '                        "target_length": self._target_length,\n'
    '                        "input_states_1": dec_h,\n'
    '                        "input_states_2": dec_c,\n'
    "                    },\n"
    "                )\n"
    "                dnamed = dict(zip(self._dec_out_names, dec_outs))\n"
    '                logits = dnamed["outputs"][0, 0, 0].copy()\n'
    "                logits[BLANK_ID] -= blank_penalty\n"
    "                token = int(np.argmax(logits))\n"
    "                if token == BLANK_ID:\n"
    "                    break\n"
    "                toks.append(token)\n"
    "                last_token = np.array([[token]], dtype=np.int32)\n"
    '                dec_h = dnamed["output_states_1"]\n'
    '                dec_c = dnamed["output_states_2"]\n'
    "        return \" \".join(_detokenize(toks, self._vocab).split())\n"
)
src = src.replace(a4, method)

open(path, "w").write(src)
print("PATCHED_OK")
