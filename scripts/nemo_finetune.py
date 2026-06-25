#!/usr/bin/env python3
# Finetune Cache-Aware FastConformer-RNNT on SAP dysarthric speech.
# Modes: overfit (Gate-1 wiring), smoke/train (val+ckpt+early-stop+top5-avg).
# v3 (post full-review): pinned single deploy context, re-enabled validation + top-5 checkpointing +
# early-stop, LR 1e-4 + warmup, dysarthria-tuned SpecAugment, post-fit checkpoint averaging.
#   source /workspace/nemoenv/bin/activate
#   python nemo_finetune.py --mode smoke --epochs 4 --freeze encoder_only --train-json train.json ...
import argparse, os, json, re, ast, glob
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["overfit", "smoke"], required=True)
ap.add_argument("--base-nemo", default="/workspace/finetune/nemo_ft/nemotron-speech-streaming-en-0.6b.nemo")
ap.add_argument("--train-json", required=True)
ap.add_argument("--val-json", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--freeze", choices=["full", "encoder_only"], default="encoder_only")
ap.add_argument("--train-ctx", default="[70,1]", help="SINGLE pinned deploy context (train+eval+export same)")
ap.add_argument("--lr", type=float, default=None)
ap.add_argument("--epochs", type=int, default=None, help="if set, train by epochs + early-stop (else max_steps)")
ap.add_argument("--max-steps", type=int, default=None)
ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf, open_dict
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
except Exception:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

TRAIN_CTX = ast.literal_eval(a.train_ctx)
LR = a.lr if a.lr is not None else (1e-3 if a.mode == "overfit" else 1e-4)
MAX_STEPS = a.max_steps if a.max_steps is not None else (400 if a.mode == "overfit" else 20000)

print(f"=== restore {a.base_nemo} ===")
m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(a.base_nemo, map_location="cpu")
print("[preflight] base optim:", OmegaConf.to_container(m.cfg.optim, resolve=True).get("name"),
      "| sched:", (m.cfg.optim.get("sched") or {}).get("name"))

# --- data: clean CLASSIC manifest dataloader (use_lhotse=False) ---
with open_dict(m.cfg):
    base_ds = {"sample_rate": 16000, "batch_size": a.bs, "num_workers": 8, "pin_memory": True,
               "use_lhotse": False, "is_tarred": False, "max_duration": 40.0, "min_duration": 0.1,
               "trim_silence": False, "shuffle_n": 2048}
    m.cfg.train_ds = OmegaConf.create({**base_ds, "manifest_filepath": a.train_json,
                                       "shuffle": (a.mode != "overfit")})
    m.cfg.validation_ds = OmegaConf.create({**base_ds, "manifest_filepath": a.val_json,
                                            "shuffle": False, "num_workers": 4})
m.setup_training_data(m.cfg.train_ds)
m.setup_validation_data(m.cfg.validation_ds)
nb = MAX_STEPS
try:
    nb = len(m._train_dl); print(f"[guard] train batches/epoch = {nb} (bs={a.bs})")
    if a.mode == "smoke":
        nl = sum(1 for _ in open(a.train_json))
        assert nb >= 0.8 * (nl / a.bs), f"dataloader capped! {nb} vs ~{nl//a.bs}"
except Exception as e:
    print("[guard] skipped:", e)

# pinned deploy context (train + eval + export all at TRAIN_CTX)
try:
    m.encoder.set_default_att_context_size(TRAIN_CTX)
    print(f"[ctx] pinned att_context = {m.encoder.att_context_size}")
except Exception as e:
    print("[ctx] set failed:", e)

# SpecAugment: overfit=off; else dysarthria-tuned (less freq masking, more time masking)
if a.mode == "overfit":
    m.spec_augmentation = None; print("[overfit] SpecAugment off")
else:
    try:
        sa = m.spec_augmentation
        for attr, val in [("freq_masks", 1), ("freq_width", 10), ("time_masks", 10)]:
            if hasattr(sa, attr): setattr(sa, attr, val)
        print("[specaug] dysarthria-tuned:",
              {k: getattr(sa, k, "?") for k in ["freq_masks", "freq_width", "time_masks", "time_width"]})
    except Exception as e:
        print("[specaug] tune skipped:", e)

if a.freeze == "encoder_only":
    for p in m.decoder.parameters(): p.requires_grad = False
    for p in m.joint.parameters():   p.requires_grad = False
    print("[freeze] decoder+joint frozen -> ENCODER-ONLY")

# optimizer: explicit AdamW + Cosine, LR + ~15% warmup (scaled to real total steps)
TOTAL = (a.epochs * nb) if (a.epochs and a.mode != "overfit") else MAX_STEPS
WARMUP = 0 if a.mode == "overfit" else max(200, int(0.15 * TOTAL))
with open_dict(m.cfg):
    m.cfg.optim = OmegaConf.create({"name": "adamw", "lr": LR, "weight_decay": 1e-3, "betas": [0.9, 0.98],
                                    "sched": {"name": "CosineAnnealing", "warmup_steps": WARMUP, "min_lr": 1e-6}})
m.setup_optimization(m.cfg.optim)
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
print(f"[optim] adamw lr={LR} cosine warmup={WARMUP} | trainable {trainable:.1f}M")

torch.set_float32_matmul_precision("high")
tk = dict(accelerator="gpu", devices=1, precision="bf16-mixed",
          num_sanity_val_steps=0, logger=False, default_root_dir=a.out_dir)
cbs = []
if a.mode == "overfit":
    tk.update(max_steps=MAX_STEPS, limit_train_batches=1, limit_val_batches=0.0, enable_checkpointing=False)
else:
    # val once per epoch (always valid) + top-5 ckpt on val_wer + early-stop
    tk.update(limit_val_batches=30, check_val_every_n_epoch=1, enable_checkpointing=True)
    if a.epochs: tk.update(max_epochs=a.epochs)
    else:        tk.update(max_steps=MAX_STEPS)
    ckpt = ModelCheckpoint(dirpath=a.out_dir, save_top_k=5, monitor="val_wer", mode="min",
                           filename="ft-{epoch}-{val_wer:.4f}")
    cbs = [ckpt, EarlyStopping(monitor="val_wer", mode="min", patience=2)]
trainer = pl.Trainer(callbacks=cbs, **tk)
m.set_trainer(trainer)
print(f"=== fit ({a.mode}, freeze={a.freeze}, epochs={a.epochs}, max_steps={MAX_STEPS}) ===")
trainer.fit(m)

# --- post-fit: average top-5 checkpoints (by val_wer) if present ---
out_nemo = os.path.join(a.out_dir, f"ft_{a.mode}_{a.freeze}.nemo")
ckpts = sorted(glob.glob(os.path.join(a.out_dir, "ft-*.ckpt")))
if len(ckpts) >= 2:
    print(f"[avg] averaging {len(ckpts)} checkpoints")
    avg = None; n = 0
    for c in ckpts:
        sd = torch.load(c, map_location="cpu").get("state_dict", {})
        if not sd: continue
        n += 1
        if avg is None: avg = {k: v.float().clone() for k, v in sd.items()}
        else:
            for k in avg:
                if k in sd: avg[k] += sd[k].float()
    if avg and n:
        for k in avg: avg[k] /= n
        m.load_state_dict(avg, strict=False)
        print(f"[avg] loaded average of {n} ckpts")
m.save_to(out_nemo); print("saved", out_nemo)

# --- post-train check at the pinned context, punctuation-stripped CER ---
def norm(s): return re.sub(r"[^\w\s]", "", s.lower()).strip()
def cer(h, r):
    h, r = list(h), list(r)
    if not r: return 0.0 if not h else 1.0
    dp = list(range(len(r)+1))
    for i in range(1, len(h)+1):
        p = dp[0]; dp[0] = i
        for j in range(1, len(r)+1):
            c = dp[j]; dp[j] = min(dp[j]+1, dp[j-1]+1, p+(h[i-1]!=r[j-1])); p = c
    return min(1.0, dp[len(r)]/len(r))
check = a.train_json if a.mode == "overfit" else a.val_json
recs = [json.loads(l) for l in open(check)][:60]
m.eval(); m.encoder.set_default_att_context_size(TRAIN_CTX)
with torch.no_grad():
    hyps = m.transcribe([r["audio_filepath"] for r in recs], batch_size=16)
hyps = [(h.text if hasattr(h, "text") else h) for h in hyps]
cers = [cer(norm(h), norm(r["text"])) for h, r in zip(hyps, recs)]
print(f"mean CER (punct-stripped) on {len(cers)} from {os.path.basename(check)} = {100*sum(cers)/len(cers):.2f}%")
print("FINETUNE_DONE")
