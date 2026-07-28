#!/usr/bin/env python3
# v2 STRUCTURAL finetune of Cache-Aware FastConformer-RNNT on SAP dysarthric speech.
# Conservative arm (Q-approved 2026-06-29): unfreeze ENCODER + JOINT (prednet/decoder FROZEN),
# differential LR (joint = encoder_lr * joint_lr_mult), gain+speed augmentation, CER-based selection.
# Targets the research/37 diagnosis: ALS empties (frozen joint over-blanks) + quiet-audio sensitivity.
# Diff vs v1 nemo_finetune.py is intentionally small. GPU-only checks are flagged [GATE0].
#   python nemo_finetune_v2.py --mode smoke --epochs 4 --freeze joint_unfreeze \
#       --train-json train.json --val-json val3k.json --out-dir OUT --train-ctx "[70,1]"
import argparse, os, json, re, ast, glob, math
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["overfit", "smoke"], required=True)
ap.add_argument("--base-nemo", default="/workspace/finetune/nemo_ft/nemotron-speech-streaming-en-0.6b.nemo")
ap.add_argument("--train-json", required=True)
ap.add_argument("--val-json", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--freeze", choices=["full", "encoder_only", "joint_unfreeze", "peft"], default="joint_unfreeze")
ap.add_argument("--train-ctx", default="[70,1]")
ap.add_argument("--lr", type=float, default=None, help="ENCODER lr (joint = lr * joint-lr-mult); for peft = adapter lr")
ap.add_argument("--joint-lr-mult", type=float, default=0.1, help="joint lr = encoder lr * this (conservative=0.1)")
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--max-steps", type=int, default=None)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--no-augment", action="store_true", help="disable gain+speed aug (Gate-0 A/B control)")
# --- PEFT arm (research/40 pivot): base frozen, only encoder adapters train -> generalization-correct ---
ap.add_argument("--adapter-dim", type=int, default=64, help="[peft] Linear bottleneck adapter dim")
ap.add_argument("--no-speed", action="store_true", help="drop speed-perturb; keep ONLY gain aug (research/40)")
ap.add_argument("--gain-min", type=float, default=-20.0, help="gain aug min dBFS (research/40 mild=-8)")
ap.add_argument("--gain-max", type=float, default=10.0, help="gain aug max dBFS (research/40 mild=+8)")
# --- FastEmit (research 2026-07-28, arXiv 2010.11148): scales token-grad by (1+lambda), blank-grad
# unchanged -> direct counter to the frozen-joint confident-blank empties AND lowers TTFT (both Pareto
# axes). NeMo-native RNNT loss kwarg. Safe band 0.004-0.01; regresses >=0.02. DEFAULT 0.0 = no change. ---
ap.add_argument("--fastemit-lambda", type=float, default=0.0,
                help="RNNT FastEmit regularization strength; 0.0=off. Sweep {0.003,0.005,0.01}, never >=0.02.")
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

# --- FastEmit: rebuild the RNNT loss with fastemit_lambda>0 (opt-in; default 0.0 leaves loss untouched).
# Targets the frozen-joint empties + lowers emission latency. VERIFY-ON-POD: RNNTLoss wants vocab size
# WITHOUT blank (num_classes_with_blank - 1); the GATE0 print below must show the lambda actually set. ---
if a.fastemit_lambda and a.fastemit_lambda > 0:
    from nemo.collections.asr.losses.rnnt import RNNTLoss
    nc = m.joint.num_classes_with_blank - 1
    m.loss = RNNTLoss(num_classes=nc, loss_name="default",
                      loss_kwargs={"fastemit_lambda": a.fastemit_lambda, "clamp": -1.0})
    with open_dict(m.cfg):
        m.cfg.loss = OmegaConf.create({"loss_name": "default",
            "warprnnt_numba_kwargs": {"fastemit_lambda": a.fastemit_lambda, "clamp": -1.0}})
    print(f"[fastemit] RNNTLoss rebuilt: fastemit_lambda={a.fastemit_lambda} num_classes={nc}")
    print("[GATE0][fastemit] loss=", type(m.loss).__name__,
          "lambda(cfg)=", m.cfg.loss.warprnnt_numba_kwargs.fastemit_lambda)
else:
    print("[fastemit] disabled (lambda=0.0) — loss unchanged")

# --- data: classic manifest dataloader (use_lhotse=False) + gain/speed augmentor [research/37 fix] ---
with open_dict(m.cfg):
    base_ds = {"sample_rate": 16000, "batch_size": a.bs, "num_workers": 8, "pin_memory": True,
               "use_lhotse": False, "is_tarred": False, "max_duration": 40.0, "min_duration": 0.1,
               "trim_silence": False, "shuffle_n": 2048}
    train_ds = {**base_ds, "manifest_filepath": a.train_json, "shuffle": (a.mode != "overfit")}
    if a.mode != "overfit" and not a.no_augment:
        # gain spans the quiet-ALS energy range (empties RMS ~0.025 vs non-empty ~0.059); speed 0.9-1.1
        # research/40: Arm A's -20/+10 + speed + severity hurt -> PEFT arm runs MILD gain (-8/+8), NO speed.
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
print("[aug]", "OFF" if (a.mode == "overfit" or a.no_augment) else m.cfg.train_ds.get("augmentor"))
# [GATE0] confirm the augmentor is actually wired into the dataset (config-set != applied)
try:
    ds = m._train_dl.dataset
    # NeMo AudioToBPEDataset holds the augmentor on the featurizer, not the dataset
    aug = getattr(getattr(ds, "featurizer", None), "augmentor", None) or getattr(ds, "augmentor", None)
    pipe = getattr(aug, "_pipeline", None)
    _exp = 0 if (a.mode == "overfit" or a.no_augment) else (1 if a.no_speed else 2)
    print(f"[GATE0][aug-check] dataset augmentor perturbations = "
          f"{len(pipe) if pipe is not None else 'NONE/unknown'} (expect {_exp})")
except Exception as e:
    print("[GATE0][aug-check] introspection failed:", e)

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

# CER-based selection: make the val metric compute CER, not WER [research/34 §B.3]. monitor stays "val_wer".
try:
    m.wer.use_cer = True
    print("[select] m.wer.use_cer=True -> 'val_wer' logged value is CER")
except Exception as e:
    print("[select] could not set use_cer (FIX in GATE0):", e)

# SpecAugment: overfit=off; else dysarthria-tuned
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

# --- FREEZE policy ---
if a.freeze == "encoder_only":
    for p in m.decoder.parameters(): p.requires_grad = False
    for p in m.joint.parameters():   p.requires_grad = False
    print("[freeze] decoder+joint frozen -> ENCODER-ONLY")
elif a.freeze == "joint_unfreeze":
    for p in m.decoder.parameters(): p.requires_grad = False   # prednet/LM frozen (streaming-safe, doc 20)
    for p in m.joint.parameters():   p.requires_grad = True    # <-- the lever: stop blank over-emission
    print("[freeze] joint_unfreeze: ENCODER+JOINT trainable, decoder(prednet) FROZEN")
elif a.freeze == "peft":
    # research/40 PIVOT: disease = overfit/poor Dev->Test transfer -> PEFT (FEWER trainable params), not
    # more capacity. Add Linear bottleneck adapters to the ENCODER, freeze the whole base, train adapters
    # only. NeMo EncDecRNNTBPEModel is an ASRAdapterModelMixin. [GATE0] verify trainable<<base on the pod.
    from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
    d_model = int(m.cfg.encoder.d_model)  # in_features must match encoder hidden dim (read, don't hardcode)
    adapter_cfg = LinearAdapterConfig(in_features=d_model, dim=a.adapter_dim, norm_position="pre")
    ADAPTER_NAME = "dysarthria_peft"
    # Gate-0 (2026-06-30) found this build's ConformerEncoder lacks AdapterModuleMixin, but each
    # ConformerLayer HAS it -> add/enable/unfreeze adapters PER LAYER (exactly what encoder.add_adapter loops).
    _enc_layers = list(m.encoder.layers)
    for _L in _enc_layers:
        _L.add_adapter(name=ADAPTER_NAME, cfg=adapter_cfg)
        _L.set_enabled_adapters(ADAPTER_NAME, enabled=True)
    for p in m.parameters(): p.requires_grad = False        # freeze EVERYTHING (base)...
    for _L in _enc_layers:
        _L.unfreeze_enabled_adapters()                      # ...then re-enable grad ONLY on adapter params
    print(f"[freeze] PEFT: per-layer Linear adapters dim={a.adapter_dim} on {len(_enc_layers)} layers "
          f"(d_model={d_model}); base FROZEN")
    # [GATE0] adapters must exist, be enabled, and be the ONLY trainable params (else the pivot is a no-op)
    try:
        print("[GATE0][peft] layer0 enabled adapters =", _enc_layers[0].get_enabled_adapters(),
              "| layers with adapters =", sum(1 for _L in _enc_layers if _L.is_adapter_available()))
        n_ad = sum(p.numel() for n, p in m.named_parameters() if p.requires_grad and "adapter" in n.lower())
        n_other = sum(p.numel() for n, p in m.named_parameters() if p.requires_grad and "adapter" not in n.lower())
        print(f"[GATE0][peft] trainable adapter params={n_ad/1e6:.3f}M  trainable NON-adapter params={n_other/1e6:.3f}M (MUST be 0)")
        assert n_ad > 0, "no adapter params train -> add_adapter/unfreeze failed"
        assert n_other == 0, "non-adapter params still train -> base not frozen"
    except Exception as e:
        print("[GATE0][peft] introspection FAILED (fix before paying for full run):", e)
        raise
# 'full' = nothing frozen

# --- OPTIMIZER ---
TOTAL = (a.epochs * nb) if (a.epochs and a.mode != "overfit") else MAX_STEPS
WARMUP = 0 if a.mode == "overfit" else max(200, int(0.10 * TOTAL))  # v1 was 15%; Phase0 -> trim to 10%

if a.freeze == "joint_unfreeze":
    # differential LR: encoder at LR, joint gentler (LR * joint_lr_mult). Manual AdamW + warmup/cosine.
    enc_p = [p for p in m.encoder.parameters() if p.requires_grad]
    jnt_p = [p for p in m.joint.parameters()   if p.requires_grad]
    assert enc_p and jnt_p, "joint_unfreeze: empty param group (encoder or joint)"
    groups = [{"params": enc_p, "lr": LR}, {"params": jnt_p, "lr": LR * a.joint_lr_mult}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-3, betas=(0.9, 0.98))
    _W, _T, _MINF = WARMUP, max(TOTAL, 1), (1e-6 / LR)
    def _lr_lambda(step):
        if _W > 0 and step < _W:
            return float(step) / float(_W)
        prog = min(1.0, float(step - _W) / float(max(1, _T - _W)))
        return max(_MINF, 0.5 * (1.0 + math.cos(math.pi * prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    # override so Lightning uses our param-grouped optimizer (NOT setup_optimization's single group)
    m.configure_optimizers = (lambda _o=opt, _s=sched: ([_o], [{"scheduler": _s, "interval": "step"}]))
    # NeMo's ModelPT logs self._optimizer.param_groups[0]['lr'] -> must be set (Gate-0 caught None here)
    m._optimizer = opt
    m._scheduler = {"scheduler": sched, "interval": "step"}
    print(f"[optim] DIFFERENTIAL adamw: enc_lr={LR} joint_lr={LR * a.joint_lr_mult} "
          f"warmup={WARMUP} total={TOTAL} groups={len(opt.param_groups)}")
else:
    with open_dict(m.cfg):
        m.cfg.optim = OmegaConf.create({"name": "adamw", "lr": LR, "weight_decay": 1e-3, "betas": [0.9, 0.98],
                                        "sched": {"name": "CosineAnnealing", "warmup_steps": WARMUP, "min_lr": 1e-6}})
    m.setup_optimization(m.cfg.optim)
    print(f"[optim] adamw lr={LR} cosine warmup={WARMUP}")

trainable = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
print(f"[optim] trainable {trainable:.1f}M params")

torch.set_float32_matmul_precision("high")
tk = dict(accelerator="gpu", devices=1, precision="bf16-mixed",
          num_sanity_val_steps=0, logger=False, default_root_dir=a.out_dir)
cbs = []
if a.mode == "overfit":
    tk.update(max_steps=MAX_STEPS, limit_train_batches=1, limit_val_batches=0.0, enable_checkpointing=False)
else:
    tk.update(limit_val_batches=1.0, check_val_every_n_epoch=1, enable_checkpointing=True)  # FULL val3k subset
    if a.epochs: tk.update(max_epochs=a.epochs)
    else:        tk.update(max_steps=MAX_STEPS)
    ckpt = ModelCheckpoint(dirpath=a.out_dir, save_top_k=5, monitor="val_wer", mode="min",
                           filename="ft-{epoch}-{val_wer:.4f}")  # val_wer == CER here
    cbs = [ckpt, EarlyStopping(monitor="val_wer", mode="min", patience=2)]
trainer = pl.Trainer(callbacks=cbs, **tk)
m.set_trainer(trainer)
print(f"=== fit ({a.mode}, freeze={a.freeze}, epochs={a.epochs}, max_steps={MAX_STEPS}) ===")
trainer.fit(m)

# [GATE0] verify the differential optimizer actually took (2 groups, both lrs present)
try:
    pg = trainer.optimizers[0].param_groups
    print(f"[GATE0] optimizer param_groups={len(pg)} lrs={[round(g['lr'],8) for g in pg]}")
except Exception as e:
    print("[GATE0] optimizer introspection failed:", e)

# --- post-fit: average top-5 (by CER-val) checkpoints, float params only ---
out_nemo = os.path.join(a.out_dir, f"ft_{a.mode}_{a.freeze}.nemo")
ckpts = sorted(glob.glob(os.path.join(a.out_dir, "ft-*.ckpt")))
if len(ckpts) >= 2:
    print(f"[avg] averaging {len(ckpts)} checkpoints (top-5 by CER-val)")
    avg = None; n = 0
    for c in ckpts:
        sd = torch.load(c, map_location="cpu", weights_only=False).get("state_dict", {})
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

# --- post-train CER check at the pinned context ---
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
print("FINETUNE_V2_DONE")
