#!/usr/bin/env python3
# Finetune the Cache-Aware FastConformer-RNNT (EncDecRNNTBPEModel) on SAP dysarthric data.
# Parameterized for Gate 1 (overfit a batch) and Gate 2 (smoke). Run inside the NeMo venv:
#   source /workspace/nemoenv/bin/activate
#   python nemo_finetune.py --mode overfit ...    # Gate 1
#   python nemo_finetune.py --mode smoke   ...    # Gate 2
#
# NOTE: NeMo's high-level training API is version-sensitive. The FIRST overfit run is expected to
# shake out any API specifics — that is exactly what Gate 1 is for. Marked [VERIFY] where most likely.
import argparse, os, json
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["overfit", "smoke"], required=True)
ap.add_argument("--base-nemo", default="/workspace/finetune/nemo_ft/nemotron-speech-streaming-en-0.6b.nemo")
ap.add_argument("--train-json", required=True)
ap.add_argument("--val-json", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--freeze", choices=["full", "encoder_only"], default="full",
                help="full = all unfrozen (Parakeet prior); encoder_only = freeze decoder+joint (anti-forgetting)")
ap.add_argument("--att-context", default="[[70,6],[70,1],[70,0]]",
                help="multi-lookahead training list; include [70,0] to keep the low-latency option strong")
ap.add_argument("--lr", type=float, default=None)            # default set per mode below
ap.add_argument("--max-steps", type=int, default=None)
ap.add_argument("--bs", type=int, default=8)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

import nemo.collections.asr as nemo_asr
from omegaconf import open_dict, OmegaConf
try:
    import lightning.pytorch as pl
except Exception:
    import pytorch_lightning as pl

ATT = OmegaConf.create(a.att_context)            # e.g. [[70,6],[70,1],[70,0]]
LR = a.lr if a.lr is not None else (1e-3 if a.mode == "overfit" else 2e-4)
MAX_STEPS = a.max_steps if a.max_steps is not None else (400 if a.mode == "overfit" else 3000)

print(f"=== restore {a.base_nemo} ===")
m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(a.base_nemo, map_location="cpu")

with open_dict(m.cfg):
    m.cfg.train_ds.manifest_filepath = a.train_json
    m.cfg.train_ds.batch_size = a.bs
    m.cfg.train_ds.shuffle = (a.mode != "overfit")          # overfit: no shuffle, hammer the same batch
    m.cfg.train_ds.is_tarred = False
    m.cfg.train_ds.max_duration = 40.0
    m.cfg.train_ds.num_workers = 4
    m.cfg.validation_ds.manifest_filepath = a.val_json
    m.cfg.validation_ds.batch_size = a.bs
    m.cfg.validation_ds.num_workers = 2
    m.cfg.encoder.att_context_size = ATT                    # train cache-aware multi-lookahead [VERIFY accepts list-of-lists]
    # optimizer
    m.cfg.optim.lr = LR
    if "sched" in m.cfg.optim and m.cfg.optim.sched is not None:
        m.cfg.optim.sched.warmup_steps = 0 if a.mode == "overfit" else 200

m.setup_training_data(m.cfg.train_ds)
m.setup_validation_data(m.cfg.validation_ds)

if a.mode == "overfit":
    m.spec_augmentation = None                              # AUG OFF for the wiring test
    print("[overfit] SpecAugment disabled")

if a.freeze == "encoder_only":
    for p in m.decoder.parameters(): p.requires_grad = False
    for p in m.joint.parameters():   p.requires_grad = False
    print("[freeze] decoder+joint frozen -> adapting ENCODER only (anti-forgetting arm)")

trainable = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
print(f"trainable params: {trainable:.1f}M ; lr={LR} ; max_steps={MAX_STEPS} ; att_context={ATT}")

trainer = pl.Trainer(
    accelerator="gpu", devices=1, precision="bf16-mixed",
    max_steps=MAX_STEPS,
    limit_train_batches=(1 if a.mode == "overfit" else 1.0),  # overfit: a single batch, repeated
    val_check_interval=(50 if a.mode == "overfit" else 500),
    num_sanity_val_steps=0, logger=False, enable_checkpointing=True,
    default_root_dir=a.out_dir,
)
m.set_trainer(trainer)
print(f"=== fit ({a.mode}) ===")
trainer.fit(m)

out_nemo = os.path.join(a.out_dir, f"ft_{a.mode}_{a.freeze}.nemo")
m.save_to(out_nemo); print("saved", out_nemo)

# ---- post-train check: transcribe the train(=overfit) / val set and report CER ----
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
recs = [json.loads(l) for l in open(check)]
wavs = [r["audio_filepath"] for r in recs][:60]
refs = [r["text"] for r in recs][:60]
print(f"=== transcribe {len(wavs)} from {check} (att={ATT[0]}) ===")
m.eval()
with torch.no_grad():
    hyps = m.transcribe(wavs, batch_size=8)                # [VERIFY] returns list of str or objects
hyps = [h.text if hasattr(h, "text") else (h[0] if isinstance(h, (list, tuple)) else h) for h in hyps]
cers = [cer(h.lower(), r.lower()) for h, r in zip(hyps, refs)]
print(f"mean CER on {len(cers)} = {100*sum(cers)/len(cers):.2f}%  (overfit target: ~0)")
for h, r, c in list(zip(hyps, refs, cers))[:8]:
    print(f"  CER={100*c:5.1f}  HYP={h[:50]!r}  REF={r[:50]!r}")
print("FINETUNE_DONE")
