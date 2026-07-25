#!/usr/bin/env python3
# AdaLoRA finetune of the cache-aware streaming parakeet_realtime RNNT on SAP dysarthric speech.
# Q-requested "parallel arm" (2026-07-25). Sibling of nemo_finetune_v2.py — SAME data/aug/[70,1]/CER
# scaffolding; the ONLY change is the trainable-parameter strategy: base FROZEN, low-rank AdaLoRA
# adapters on the encoder attention Linear layers, MERGED into the weights before save so the deployed
# .nemo is a plain streaming model (no adapter modules at inference -> sidesteps the research/45
# adapter-vs-streaming-cache-norm collapse; a merged LoRA is a pure weight delta, no extra norm).
#
# WHY this over full/encoder-only FT: fewer trainable params -> less Dev->Test overfit (the Nemotron
# failure mode). Gate is IDENTICAL to Arm A and MUST run through the real streaming harness.
#
# Honesty note (peft 0.19.1 + NeMo integration, verify on pod):
#  * True AdaLoRA rank reallocation needs PeftModel.update_and_allocate(step) called each step — wired
#    here via a Lightning callback. If that machinery can't be driven in-NeMo, the arm degrades to
#    fixed-rank LoRA-equivalent; the script LOGS which happened (never silently substitutes).
#  * Orthogonal-reg loss (an AdaLoRA refinement) is omitted unless trivially reachable; update_and_allocate
#    (the core adaptive-rank mechanism) is the part we drive.
#
#   python nemo_finetune_adalora.py --base-nemo /workspace/sweep/parakeet120.nemo \
#       --train-json train.json --val-json val.json --out-dir OUT --epochs 4 --train-ctx "[70,1]" \
#       --init-r 12 --target-r 4 --smoke   # add --smoke for a tiny shakeout
import argparse, os, json, re, ast, glob, math
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--base-nemo", required=True)
ap.add_argument("--train-json", required=True)
ap.add_argument("--val-json", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--train-ctx", default="[70,1]")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--max-steps", type=int, default=None)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=5e-4, help="AdaLoRA adapters tolerate higher LR than full FT")
ap.add_argument("--init-r", type=int, default=12, help="AdaLoRA initial rank (pruned toward target-r)")
ap.add_argument("--target-r", type=int, default=4, help="AdaLoRA final average rank")
ap.add_argument("--lora-alpha", type=int, default=32)
ap.add_argument("--lora-dropout", type=float, default=0.0)
ap.add_argument("--targets", default="self_attn",
                help="which encoder Linear groups to adapt: 'self_attn' (linear_q/k/v/out) or 'attn+ff'")
ap.add_argument("--no-augment", action="store_true")
ap.add_argument("--no-speed", action="store_true")
ap.add_argument("--gain-min", type=float, default=-20.0)
ap.add_argument("--gain-max", type=float, default=10.0)
ap.add_argument("--smoke", action="store_true", help="tiny shakeout: cap steps, skip early-stop patience")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf, open_dict
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
except Exception:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback

TRAIN_CTX = ast.literal_eval(a.train_ctx)
LR = a.lr
MAX_STEPS = a.max_steps if a.max_steps is not None else (60 if a.smoke else 20000)

print(f"=== restore {a.base_nemo} ===")
# parakeet_realtime_eou_120m restores as EncDecRNNTBPEModel (verified GATE0 2026-07-25).
m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(a.base_nemo, map_location="cpu")

# --- data pipeline: identical to nemo_finetune_v2.py (classic manifest + gain/speed aug) ---
with open_dict(m.cfg):
    base_ds = {"sample_rate": 16000, "batch_size": a.bs, "num_workers": 8, "pin_memory": False,
               "use_lhotse": False, "is_tarred": False, "max_duration": 40.0, "min_duration": 0.1,
               "trim_silence": False, "shuffle_n": 2048}
    train_ds = {**base_ds, "manifest_filepath": a.train_json, "shuffle": True}
    if not a.no_augment:
        augmentor = {"gain": {"prob": 0.5, "min_gain_dbfs": a.gain_min, "max_gain_dbfs": a.gain_max}}
        if not a.no_speed:
            augmentor["speed"] = {"prob": 0.5, "sr": 16000, "resample_type": "kaiser_best",
                                  "min_speed_rate": 0.9, "max_speed_rate": 1.1}
        train_ds["augmentor"] = augmentor
    m.cfg.train_ds = OmegaConf.create(train_ds)
    m.cfg.validation_ds = OmegaConf.create({**base_ds, "manifest_filepath": a.val_json,
                                            "shuffle": False, "num_workers": 4})
m.setup_training_data(m.cfg.train_ds)
m.setup_validation_data(m.cfg.validation_ds)
print("[aug]", "OFF" if a.no_augment else m.cfg.train_ds.get("augmentor"))

nb = MAX_STEPS
try:
    nb = len(m._train_dl); print(f"[guard] train batches/epoch = {nb} (bs={a.bs})")
except Exception as e:
    print("[guard] skipped:", e)

# pinned deploy context (train == eval == export == deploy)
try:
    m.encoder.set_default_att_context_size(TRAIN_CTX)
    print(f"[ctx] pinned att_context = {m.encoder.att_context_size}")
except Exception as e:
    print("[ctx] set failed:", e)

# CER-based selection (val_wer logs CER)
try:
    m.wer.use_cer = True; print("[select] use_cer=True -> val_wer == CER")
except Exception as e:
    print("[select] use_cer failed:", e)

# dysarthria-tuned SpecAugment (heavier time masking, lighter freq) — same as v2
try:
    sa = m.spec_augmentation
    for attr, val in [("freq_masks", 1), ("freq_width", 10), ("time_masks", 10)]:
        if hasattr(sa, attr): setattr(sa, attr, val)
    print("[specaug] tuned:", {k: getattr(sa, k, "?") for k in ["freq_masks","freq_width","time_masks","time_width"]})
except Exception as e:
    print("[specaug] tune skipped:", e)

# ------------------------------------------------------------------------------------------
# AdaLoRA injection on the ENCODER attention Linear layers via HF peft.
# We wrap the encoder SUBMODULE (not the whole model) so NeMo's LightningModule/training_step
# stays intact; RNNT loss is computed as usual and grads flow only to the adapters.
# ------------------------------------------------------------------------------------------
from peft import AdaLoraConfig, get_peft_model

# Discover target Linear module names by suffix (read the graph, don't hardcode layer counts).
attn_suffixes = ["linear_q", "linear_k", "linear_v", "linear_out"]
ff_suffixes = ["linear_1", "linear_2"]  # conformer FFN
want = attn_suffixes + (ff_suffixes if a.targets == "attn+ff" else [])
target_names = sorted({
    name.split(".")[-1]
    for name, mod in m.encoder.named_modules()
    if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in want
})
assert target_names, f"no encoder Linear targets matched {want} — inspect encoder.named_modules()"
print(f"[adalora] target_modules = {target_names}")

TOTAL = (a.epochs * nb) if (a.epochs and not a.smoke) else MAX_STEPS
# AdaLoRA rank schedule: warm up (tinit), then prune init_r -> target_r between tinit..tfinal.
tinit = max(1, int(0.1 * TOTAL))
tfinal = max(tinit + 1, int(0.7 * TOTAL))
ada_cfg = AdaLoraConfig(
    init_r=a.init_r, target_r=a.target_r, tinit=tinit, tfinal=tfinal, total_step=TOTAL,
    lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, target_modules=target_names,
    bias="none", task_type=None,
)
m.encoder = get_peft_model(m.encoder, ada_cfg)
# get_peft_model froze the ENCODER base + added adapters, but decoder/joint live OUTSIDE
# m.encoder and stay trainable (~5.3M) — freeze them so ONLY adapters train (the low-param,
# less-overfit AdaLoRA arm; base FULLY frozen). Verified needed by the CPU probe.
for p in m.decoder.parameters(): p.requires_grad = False
for p in m.joint.parameters():   p.requires_grad = False
# get_peft_model can move the wrapped encoder to CUDA (initializing a CUDA context in the
# parent), leaving a split-device model -> pin_memory dataloader workers then crash with
# "CUDA error: initialization error" at fork. Force uniform CPU placement; Lightning moves
# the whole model to GPU together at fit. (Verified fix 2026-07-25.)
m = m.to("cpu")
n_train = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
n_adapter = sum(p.numel() for n, p in m.named_parameters() if p.requires_grad and "lora" in n.lower()) / 1e6
n_other = sum(p.numel() for n, p in m.named_parameters() if p.requires_grad and "lora" not in n.lower()) / 1e6
print(f"[adalora] trainable {n_train:.3f}M (adapter {n_adapter:.3f}M / non-adapter {n_other:.3f}M — non-adapter MUST be ~0)")
assert n_other < 0.01, f"non-adapter params still trainable ({n_other:.3f}M) — base not fully frozen"
_HAS_ALLOC = hasattr(m.encoder, "update_and_allocate")
print(f"[adalora] update_and_allocate present on peft encoder: {_HAS_ALLOC} "
      f"({'TRUE adaptive-rank AdaLoRA' if _HAS_ALLOC else 'DEGRADES to fixed-rank LoRA — REPORT THIS'})")


class AdaLoRAAllocCallback(Callback):
    """Drive AdaLoRA's rank reallocation each optimizer step (the adaptive-rank mechanism).
    No-op if the peft encoder doesn't expose update_and_allocate (logged at init)."""
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if _HAS_ALLOC:
            try:
                pl_module.encoder.update_and_allocate(trainer.global_step)
            except Exception as e:
                if trainer.global_step < 3:
                    print("[adalora] update_and_allocate error:", e)


# --- OPTIMIZER: AdamW over adapter params, warmup+cosine (manual, like v2's differential path) ---
WARMUP = 0 if a.smoke else max(200, int(0.10 * TOTAL))
params = [p for p in m.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-3, betas=(0.9, 0.98))
_W, _T, _MINF = WARMUP, max(TOTAL, 1), (1e-6 / LR)
def _lr_lambda(step):
    if _W > 0 and step < _W:
        return float(step) / float(_W)
    prog = min(1.0, float(step - _W) / float(max(1, _T - _W)))
    return max(_MINF, 0.5 * (1.0 + math.cos(math.pi * prog)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
m.configure_optimizers = (lambda _o=opt, _s=sched: ([_o], [{"scheduler": _s, "interval": "step"}]))
m._optimizer = opt
m._scheduler = {"scheduler": sched, "interval": "step"}
print(f"[optim] adamw lr={LR} warmup={WARMUP} total={TOTAL}")

torch.set_float32_matmul_precision("high")
tk = dict(accelerator="gpu", devices=1, precision="bf16-mixed",
          num_sanity_val_steps=0, logger=False, default_root_dir=a.out_dir)
cbs = [AdaLoRAAllocCallback()]
if a.smoke:
    tk.update(max_steps=MAX_STEPS, limit_val_batches=2, enable_checkpointing=False)
else:
    tk.update(max_epochs=a.epochs, limit_val_batches=1.0, check_val_every_n_epoch=1, enable_checkpointing=True)
    ckpt = ModelCheckpoint(dirpath=a.out_dir, save_top_k=5, monitor="val_wer", mode="min",
                           filename="ada-{epoch}-{val_wer:.4f}")
    cbs += [ckpt, EarlyStopping(monitor="val_wer", mode="min", patience=2)]
trainer = pl.Trainer(callbacks=cbs, **tk)
m.set_trainer(trainer)
print(f"=== fit (adalora, smoke={a.smoke}, epochs={a.epochs}, max_steps={MAX_STEPS}) ===")
trainer.fit(m)

# --- MERGE adapters into base weights -> plain streaming-safe encoder, then save .nemo ---
print("=== merge_and_unload adapters into encoder ===")
try:
    merged_encoder = m.encoder.merge_and_unload()
    m.encoder = merged_encoder
    print("[adalora] merged; encoder is now a plain module (no adapter layers at inference)")
except Exception as e:
    print("[adalora] MERGE FAILED (deploy would still carry adapters):", e); raise

# re-pin context on the merged encoder before any checkpoint-average / save
try:
    m.encoder.set_default_att_context_size(TRAIN_CTX)
except Exception as e:
    print("[ctx] re-pin after merge failed:", e)

out_nemo = os.path.join(a.out_dir, "ft_adalora.nemo")
m.save_to(out_nemo); print("saved", out_nemo)

# --- post-train offline CER sanity (punct-stripped) at the pinned context ---
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
recs = [json.loads(l) for l in open(a.val_json)][:60]
m.eval()
try: m.encoder.set_default_att_context_size(TRAIN_CTX)
except Exception: pass
with torch.no_grad():
    hyps = m.transcribe([r["audio_filepath"] for r in recs], batch_size=16)
hyps = [(h.text if hasattr(h, "text") else h) for h in hyps]
cers = [cer(norm(h), norm(r["text"])) for h, r in zip(hyps, recs)]
print(f"mean CER (punct-stripped, OFFLINE) on {len(cers)} from {os.path.basename(a.val_json)} = {100*sum(cers)/len(cers):.2f}%")
print("FINETUNE_ADALORA_DONE")
