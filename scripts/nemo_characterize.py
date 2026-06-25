#!/usr/bin/env python3
# Gate 0: characterize the actual Cache-Aware FastConformer-RNNT checkpoint so finetuning
# hyperparameters are MEASURED, not borrowed from Parakeet. Run on the pod in a NeMo env (GPU
# not required just to load). Downloads the .nemo if absent.
#
#   pip install -q "nemo_toolkit[asr]" huggingface_hub
#   python3 nemo_characterize.py
import os, sys, json

MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
LOCAL = "/workspace/finetune/nemo_ft/nemotron-speech-streaming-en-0.6b.nemo"

def get_nemo_path():
    if os.path.exists(LOCAL):
        return LOCAL
    os.makedirs(os.path.dirname(LOCAL), exist_ok=True)
    from huggingface_hub import hf_hub_download
    # filename may vary; try the obvious one, else list repo
    try:
        return hf_hub_download(MODEL_ID, filename=os.path.basename(LOCAL), local_dir=os.path.dirname(LOCAL))
    except Exception as e:
        from huggingface_hub import list_repo_files
        files = [f for f in list_repo_files(MODEL_ID) if f.endswith(".nemo")]
        print("repo .nemo files:", files)
        if not files:
            raise
        return hf_hub_download(MODEL_ID, filename=files[0], local_dir=os.path.dirname(LOCAL))

def dig(cfg, *keys, default="<absent>"):
    cur = cfg
    for k in keys:
        try:
            cur = cur[k]
        except Exception:
            try:
                cur = getattr(cur, k)
            except Exception:
                return default
    return cur

def main():
    path = get_nemo_path()
    print(f"=== loading {path} ===")
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf
    m = nemo_asr.models.ASRModel.restore_from(path, map_location="cpu")
    cfg = m.cfg
    print("=== MODEL CLASS ===", type(m).__name__)

    print("\n=== DECODER / LOSS (TDT vs RNNT?) ===")
    print("decoder._target_:", dig(cfg, "decoder", "_target_"))
    print("joint._target_:  ", dig(cfg, "joint", "_target_"))
    print("loss:            ", dig(cfg, "loss"))
    print("model_defaults:  ", dig(cfg, "model_defaults"))

    print("\n=== ENCODER / STREAMING (att_context) ===")
    for k in ["_target_", "n_layers", "d_model", "att_context_size", "att_context_style",
              "att_context_probs", "conv_context_size", "subsampling_factor"]:
        print(f"encoder.{k}:", dig(cfg, "encoder", k))

    print("\n=== TOKENIZER / VOCAB ===")
    try:
        print("vocab_size:", m.tokenizer.vocab_size)
    except Exception as e:
        print("tokenizer:", dig(cfg, "tokenizer"))
    print("labels/blank present in decoder cfg:", dig(cfg, "decoder", "vocab_size"))

    print("\n=== LANGUAGE / PROMPT HEAD (the 'prompt' question) ===")
    for probe in ["prompt", "lang", "language", "target_lang", "aux", "adapter"]:
        # shallow scan of top-level cfg keys
        hits = [kk for kk in OmegaConf.to_container(cfg, resolve=False).keys() if probe in kk.lower()]
        if hits:
            print(f"  cfg keys matching '{probe}':", hits)
    print("  (if none above, en-0.6b likely omits the language head -> simplest case)")

    print("\n=== FULL CFG DUMP (saved) ===")
    out = "/workspace/finetune/nemo_ft/cfg_dump.yaml"
    with open(out, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print("wrote", out)

if __name__ == "__main__":
    main()
