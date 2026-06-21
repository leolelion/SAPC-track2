#!/bin/bash
# Build an n-gram LM from SAPC2 transcripts for shallow fusion (research/08, experiment E2).
# Run on the pod. Produces a KenLM ARPA/binary; convert to whatever sherpa-onnx ONLINE decoding accepts.
#
# IMPORTANT (verify on pod): sherpa-onnx ONLINE transducer modified_beam_search may require an ONNX RNN-LM
# rather than a KenLM n-gram. Two paths:
#   (A) n-gram supported online -> use the KenLM binary directly.
#   (B) only RNN-LM online -> train a small LSTM LM on this same text via icefall egs/.../rnn_lm and
#       export to ONNX, then pass via decode_lm.py --lm rnnlm.onnx.
# Decide A vs B first; this script prepares the TEXT + a KenLM (useful for either: B trains the RNN-LM on it,
# and the n-gram is also usable for offline/rescoring paths).
set -e
DATA=/workspace/SAPC2/manifest/Train.csv
OUT=/workspace/finetune/lm
mkdir -p $OUT

echo "=== extract normalized transcripts (no-disfluency ref; the training target) ==="
python3 - <<PY
import csv
n=0
with open("$OUT/corpus.txt","w") as o:
    for r in csv.DictReader(open("$DATA")):
        t=r.get("norm_text_without_disfluency","").strip()
        if t:
            o.write(t+"\n"); n+=1
print("lines:", n)
PY

echo "=== train KenLM (3-gram and 4-gram) ==="
# requires kenlm (lmplz, build_binary). Install if missing: pip install https://github.com/kpu/kenlm/archive/master.zip
# or build from source. Adjust path to lmplz/build_binary as needed.
for O in 3 4; do
  lmplz -o $O --text $OUT/corpus.txt --arpa $OUT/sapc2_${O}gram.arpa --discount_fallback
  build_binary $OUT/sapc2_${O}gram.arpa $OUT/sapc2_${O}gram.bin
done
echo "=== done: $OUT ==="; ls -la $OUT
