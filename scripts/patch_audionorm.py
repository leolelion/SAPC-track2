#!/usr/bin/env python3
"""Patch model.py to add env-gated input gain normalization before the mel.
SAPC2_AUDIO_NORM = none(default) | rms | peak. Targets via SAPC2_RMS_TARGET / SAPC2_PEAK_TARGET.
Default 'none' => identical to shipped behavior. Offline/accuracy pass uses full-utterance gain
(non-causal); a causal AGC would be a follow-up if this helps."""
import sys
path = sys.argv[1]
src = open(path).read()
if "SAPC2_AUDIO_NORM" in src:
    print("ALREADY_PATCHED"); sys.exit(0)
anchor = "        all_audio = np.concatenate(self._raw_chunks)\n        audio_t = torch.from_numpy(all_audio).unsqueeze(0)\n"
assert src.count(anchor) == 1, "anchor"
ins = (
    "        all_audio = np.concatenate(self._raw_chunks)\n"
    "        _nm = os.environ.get(\"SAPC2_AUDIO_NORM\", \"none\")\n"
    "        if _nm == \"rms\":\n"
    "            _r = float(np.sqrt((all_audio.astype(np.float64) ** 2).mean())) + 1e-9\n"
    "            all_audio = (all_audio * (float(os.environ.get(\"SAPC2_RMS_TARGET\", \"0.05\")) / _r)).astype(np.float32)\n"
    "        elif _nm == \"peak\":\n"
    "            _p = float(np.abs(all_audio).max()) + 1e-9\n"
    "            all_audio = (all_audio * (float(os.environ.get(\"SAPC2_PEAK_TARGET\", \"0.9\")) / _p)).astype(np.float32)\n"
    "        elif _nm == \"rms_cond\":\n"
    "            _t = float(os.environ.get(\"SAPC2_RMS_TARGET\", \"0.05\"))\n"
    "            _r = float(np.sqrt((all_audio.astype(np.float64) ** 2).mean())) + 1e-9\n"
    "            if _r < _t:\n"
    "                all_audio = (all_audio * (_t / _r)).astype(np.float32)\n"
    "        audio_t = torch.from_numpy(all_audio).unsqueeze(0)\n"
)
open(path, "w").write(src.replace(anchor, ins))
print("PATCHED_OK")
