#!/usr/bin/env python3
# Gate 0 (dependency-free): a .nemo is just a tar archive containing model_config.yaml +
# tokenizer + weights. Read the config WITHOUT importing nemo/torch (avoids the torch/torchvision
# ABI hell). Pins decoder type, att_context_size(+probs), d_model, vocab, language head.
#   python3 nemo_characterize.py [/path/to/model.nemo]
import sys, os, tarfile, io

NEMO = sys.argv[1] if len(sys.argv) > 1 else "/workspace/finetune/nemo_ft/nemotron-speech-streaming-en-0.6b.nemo"

def main():
    if not os.path.exists(NEMO):
        print(f"NOT FOUND: {NEMO}"); sys.exit(2)
    print(f"=== reading {NEMO} ({os.path.getsize(NEMO)/1e9:.2f} GB) ===")
    with tarfile.open(NEMO, "r:*") as tar:
        members = tar.getnames()
        cfgname = next((m for m in members if m.endswith("model_config.yaml")), None) \
            or next((m for m in members if m.endswith(".yaml")), None)
        print("=== archive members (non-weight) ===")
        for m in members:
            if not m.endswith((".ckpt", ".pt", ".bin")):
                print("  ", m)
        if not cfgname:
            print("NO model_config.yaml found"); sys.exit(3)
        raw = tar.extractfile(cfgname).read().decode("utf-8", "replace")
    # save full
    out = os.path.join(os.path.dirname(NEMO), "cfg_dump.yaml")
    open(out, "w").write(raw); print(f"\nwrote {out}\n")
    # print key lines without needing a yaml parser
    print("=== KEY CONFIG LINES (decoder/att_context/d_model/vocab/lang/prompt) ===")
    for ln in raw.splitlines():
        low = ln.lower()
        if any(k in low for k in ["att_context", "_target_", "d_model", "n_layers", "num_classes",
                                  "vocab", "subsampling", "model_type", "tdt", "durations",
                                  "prompt", "lang", "joint", "decoder", "tokenizer", "blank"]):
            print(ln.rstrip())

if __name__ == "__main__":
    main()
