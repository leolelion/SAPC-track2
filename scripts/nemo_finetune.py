#!/usr/bin/env python3
# Finetune Cache-Aware FastConformer-RNNT (EncDecRNNTBPEModel) on SAP dysarthric data.
# Gate 1 (overfit-a-batch) and Gate 2 (smoke). Run inside the NeMo venv:
#   source /workspace/nemoenv/bin/activate && python nemo_finetune.py --mode overfit ...
# v2: fixes from independent review — explicit optimizer (no inherited Noam), setup_optimization,
# single flat train context via set_default_att_context_size (multi-lookahead deferred to full run),
# punctuation-stripped overfit CER, smoke dataloader-length guard, set ctx before transcribe.
import argparse, os, json, re, ast
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["overfit", "smoke"], required=True)
ap.add_argument("--base-nemo", default="/workspace/finetune/nemo_ft/nemotron-speech-streaming-en-0.6b.nemo")
ap.add_argument("--train-json", required=True)
ap.add_argument("--val-json", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--freeze", choices=["full", "encoder_only"], default="full")
ap.add_argument("--train-ctx", default="[70,1]",
                help="SINGLE flat att_context for training (e.g. [70,1] low-latency, [70,6] current deploy)")
ap.add_argument("--lr", type=float, default=None)
ap.add_argument("--max-steps", type=int, default=None)
ap.add_argument("--bs", type=int, default=8)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf, open_dict
try:
    import lightning.pytorch as pl
except Exception:
    import pytorch_lightning as pl

TRAIN_CTX = ast.literal_eval(a.train_ctx)                       # real python list, e.g. [70,1]
LR = a.lr if a.lr is not None else (1e-3 if a.mode == "overfit" else 2e-4)
MAX_STEPS = a.max_steps if a.max_steps is not None else (400 if a.mode == "overfit" else 3000)
WARMUP = 0 if a.mode == "overfit" else 200

print(f"=== restore {a.base_nemo} ===")
m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(a.base_nemo, map_location="cpu")
print("[preflight] base optim cfg:", OmegaConf.to_container(m.cfg.optim, resolve=True))
print("[preflight] base att_context_size:", m.cfg.encoder.get("att_context_size"))

# --- data: REPLACE the inherited Lhotse/tarred config with a clean CLASSIC manifest dataloader
# (base used use_lhotse + text_field=answer + tarred bucketing -> num_buckets=None crash). ---
with open_dict(m.cfg):
    base_ds = {"sample_rate": 16000, "batch_size": a.bs, "num_workers": 4, "pin_memory": True,
               "use_lhotse": False, "is_tarred": False, "max_duration": 40.0, "min_duration": 0.1,
               "trim_silence": False, "shuffle_n": 0}
    m.cfg.train_ds = OmegaConf.create({**base_ds, "manifest_filepath": a.train_json,
                                       "shuffle": (a.mode != "overfit")})
    m.cfg.validation_ds = OmegaConf.create({**base_ds, "manifest_filepath": a.val_json,
                                            "shuffle": False, "num_workers": 2})
m.setup_training_data(m.cfg.train_ds)
m.setup_validation_data(m.cfg.validation_ds)

# --- guard the ~1000-samples/epoch trap (NeMo #15782) for the smoke run ---
try:
    n_batches = len(m._train_dl)
    print(f"[guard] train dataloader batches/epoch = {n_batches} (bs={a.bs})")
    if a.mode == "smoke":
        n_lines = sum(1 for _ in open(a.train_json))
        assert n_batches >= 0.8 * (n_lines / a.bs), \
            f"dataloader capped! {n_batches} batches vs expected ~{n_lines//a.bs}; check limit_train_batches/sampler"
except Exception as e:
    print("[guard] dataloader length check skipped/failed:", e)

# --- train context (single flat; set_default_att_context_size actually re-takes on the live encoder) ---
try:
    m.encoder.set_default_att_context_size(TRAIN_CTX)
    print(f"[ctx] training/eval att_context set to {TRAIN_CTX}; now = {m.encoder.att_context_size}")
except Exception as e:
    print("[ctx] set_default_att_context_size failed:", e)

if a.mode == "overfit":
    m.spec_augmentation = None
    print("[overfit] SpecAugment disabled")

if a.freeze == "encoder_only":
    for p in m.decoder.parameters(): p.requires_grad = False
    for p in m.joint.parameters():   p.requires_grad = False
    print("[freeze] decoder+joint frozen -> adapting ENCODER only")

# --- optimizer: set EXPLICITLY (avoid inheriting an unknown Noam scale) + actually build it ---
with open_dict(m.cfg):
    m.cfg.optim = OmegaConf.create({
        "name": "adamw", "lr": LR, "weight_decay": 1e-3, "betas": [0.9, 0.98],
        "sched": {"name": "CosineAnnealing", "warmup_steps": WARMUP, "min_lr": 1e-6},
    })
m.setup_optimization(m.cfg.optim)
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
print(f"[optim] adamw lr={LR} cosine warmup={WARMUP} | trainable {trainable:.1f}M | max_steps={MAX_STEPS}")

torch.set_float32_matmul_precision("high")                     # use H200 tensor cores
# No in-training validation for the gates (avoids all val_check_interval-vs-batch-count crashes);
# the post-train transcribe + held-out Dev_diag eval are the real signals.
tk = dict(accelerator="gpu", devices=1, precision="bf16-mixed", max_steps=MAX_STEPS,
          num_sanity_val_steps=0, logger=False, enable_checkpointing=False, default_root_dir=a.out_dir,
          limit_val_batches=0.0)
if a.mode == "overfit":
    tk["limit_train_batches"] = 1
trainer = pl.Trainer(**tk)
m.set_trainer(trainer)
print(f"=== fit ({a.mode}, freeze={a.freeze}) ===")
trainer.fit(m)
out_nemo = os.path.join(a.out_dir, f"ft_{a.mode}_{a.freeze}.nemo")
m.save_to(out_nemo); print("saved", out_nemo)

# --- post-train check at the SAME training context, punctuation-stripped CER ---
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
    hyps = m.transcribe([r["audio_filepath"] for r in recs], batch_size=8)
hyps = [(h.text if hasattr(h, "text") else h) for h in hyps]
cers = [cer(norm(h), norm(r["text"])) for h, r in zip(hyps, recs)]
print(f"mean CER (punct-stripped) on {len(cers)} from {os.path.basename(check)} = "
      f"{100*sum(cers)/len(cers):.2f}%  (overfit PASS target: ~0)")
for h, r, c in list(zip(hyps, recs, cers))[:8]:
    print(f"  CER={100*c:5.1f}  HYP={h[:48]!r}  REF={r['text'][:48]!r}")
print("FINETUNE_DONE")
