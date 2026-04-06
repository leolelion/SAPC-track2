#!/usr/bin/env python3
"""
finetune.py — Finetune icefall streaming Zipformer on SAPC dysarthric speech data.

Usage:
    python3 finetuning/finetune.py --config finetuning/finetune_config.yaml

    # Resume from last checkpoint:
    python3 finetuning/finetune.py --config finetuning/finetune_config.yaml --resume

    # Override individual settings:
    python3 finetuning/finetune.py --config finetuning/finetune_config.yaml \
        --base-lr 0.0003 --num-epochs 30
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# ── Add icefall to path ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
ICEFALL_DIR = REPO_ROOT / "finetuning" / "icefall"
ZIPFORMER_DIR = ICEFALL_DIR / "egs" / "librispeech" / "ASR" / "zipformer"

for p in [str(ICEFALL_DIR), str(ZIPFORMER_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import sentencepiece as spm
from lhotse import CutSet, load_manifest_lazy
from lhotse.dataset import (
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    SpecAugment,
)
from lhotse.dataset.collation import collate_custom_field
from omegaconf import OmegaConf

# icefall imports (available after sys.path setup above)
try:
    from train import (
        add_model_arguments,
        get_model,
        get_params,
        load_checkpoint_if_available,
    )
    from icefall.checkpoint import load_checkpoint, save_checkpoint
    from icefall.env import get_env_info
    from icefall.utils import (
        AttributeDict,
        MetricsTracker,
        setup_logger,
        str2bool,
    )
except ImportError as e:
    sys.exit(
        f"Cannot import icefall modules: {e}\n"
        "Make sure you ran: bash finetuning/setup_finetune.sh"
    )

try:
    import k2
    from icefall.decode import get_lattice
    from icefall.graph_compiler import CtcTrainingGraphCompiler
except ImportError as e:
    sys.exit(f"k2 not found: {e}\nRun: bash finetuning/setup_finetune.sh")

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── Config helpers ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("training", cfg)


def merge_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """CLI flags override config file values."""
    overrides = {
        "base_lr":             args.base_lr,
        "num_epochs":          args.num_epochs,
        "max_duration":        args.max_duration,
        "exp_dir":             args.exp_dir,
        "train_cuts":          args.train_cuts,
        "dev_cuts":            args.dev_cuts,
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "bpe_model":           args.bpe_model,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


# ── Model params ───────────────────────────────────────────────────────────────

def build_zipformer_params(cfg: dict) -> AttributeDict:
    """Build icefall param object matching the standard pretrained model."""
    params = get_params()

    # Architecture — must exactly match the pretrained epoch-30.pt
    params.causal            = cfg.get("causal", True)
    params.chunk_size        = cfg.get("chunk_size", "16")
    params.left_context_frames = cfg.get("left_context_frames", "128")
    params.num_encoder_layers   = "2,2,3,4,3,2"
    params.downsampling_factor  = "1,2,4,8,4,2"
    params.feedforward_dim      = "512,768,1024,1536,1024,768"
    params.encoder_dim          = "192,256,384,512,384,256"
    params.encoder_unmasked_dim = "192,192,256,256,256,192"
    params.num_heads            = "4,4,4,8,4,4"
    params.cnn_module_kernel    = "31,31,15,15,15,31"
    params.dim_feedforward      = 2048
    params.num_decoder_layers   = 6
    params.decoder_dim          = 512
    params.joiner_dim           = 512
    params.feature_dim          = 80
    params.vocab_size           = 500   # BPE-500

    # Training params
    params.base_lr              = cfg.get("base_lr", 0.0005)
    params.weight_decay         = cfg.get("weight_decay", 1e-6)
    params.label_smoothing      = cfg.get("label_smoothing", 0.1)
    params.num_epochs           = cfg.get("num_epochs", 20)
    params.start_epoch          = 1
    params.exp_dir              = Path(cfg["exp_dir"])
    params.world_size           = cfg.get("world_size", 1)
    params.accumulate_grad_steps = cfg.get("accumulate_grad_steps", 4)
    params.grad_clip_threshold  = cfg.get("grad_clip", 5.0)
    params.seed                 = 42

    # Scheduler
    params.lr_batches  = 7500
    params.lr_epochs   = 3.5
    params.warmup_batches = cfg.get("warmup_steps", 500)

    return params


# ── Data loading ───────────────────────────────────────────────────────────────

def build_dataloader(cuts_path: str, cfg: dict, is_train: bool):
    cuts = load_manifest_lazy(cuts_path)

    sampler = DynamicBucketingSampler(
        cuts,
        max_duration=cfg.get("max_duration", 300),
        shuffle=is_train,
        drop_last=False,
        num_buckets=30,
    )

    on_the_fly_feats = None
    try:
        import kaldifeat
        opts = kaldifeat.FbankOptions()
        opts.device        = "cuda" if torch.cuda.is_available() else "cpu"
        opts.frame_opts.dither = 0.0
        opts.mel_opts.num_bins = 80
        on_the_fly_feats = kaldifeat.Fbank(opts)
    except ImportError:
        pass  # fall back to lhotse's built-in feature extraction

    dataset = K2SpeechRecognitionDataset(
        input_strategy=None if on_the_fly_feats is None
                        else "on-the-fly-fbank-kaldifeat",
        return_cuts=True,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=cfg.get("num_workers", 4),
        persistent_workers=cfg.get("num_workers", 4) > 0,
    )
    return loader


# ── Training ───────────────────────────────────────────────────────────────────

def compute_loss(
    params,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
    graph_compiler,
):
    """Run one forward pass and return the transducer loss."""
    device = next(model.parameters()).device

    feature        = batch["inputs"].to(device)
    feature_lengths = batch["input_lens"].to(device)
    texts          = [" ".join(c.supervisions[0].text.split()) for c in batch["cuts"]]

    token_ids = [sp.encode(t) for t in texts]
    with torch.set_grad_enabled(is_training):
        loss, _ = model(
            x                = feature,
            x_lens           = feature_lengths,
            y                = k2.RaggedTensor(token_ids).to(device),
            prune_range      = 5,
            am_scale         = 0.0,
            lm_scale         = 0.0,
        )
    return loss


def train_one_epoch(
    params,
    model: nn.Module,
    optimizer,
    scheduler,
    sp: spm.SentencePieceProcessor,
    train_loader,
    valid_loader,
    scaler: torch.cuda.amp.GradScaler,
    tb_writer: SummaryWriter,
    graph_compiler=None,
):
    model.train()
    tot_loss   = MetricsTracker()
    device     = next(model.parameters()).device
    step       = (params.cur_epoch - 1) * params.steps_per_epoch

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        params.batch_idx_train = step + batch_idx

        with torch.cuda.amp.autocast(enabled=True):
            loss = compute_loss(
                params, model, sp, batch, is_training=True,
                graph_compiler=graph_compiler,
            )
            loss = loss / params.accumulate_grad_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % params.accumulate_grad_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), params.grad_clip_threshold
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        tot_loss["loss"] += loss.item() * params.accumulate_grad_steps
        tot_loss["frames"] += batch["input_lens"].sum().item()

        if batch_idx % params.log_interval == 0:
            cur_lr = scheduler.get_last_lr()[0]
            avg_loss = tot_loss["loss"] / max(1, batch_idx + 1)
            log.info(
                f"Epoch {params.cur_epoch}/{params.num_epochs}  "
                f"batch {batch_idx}  loss={avg_loss:.4f}  lr={cur_lr:.2e}"
            )
            tb_writer.add_scalar("train/loss",  avg_loss, params.batch_idx_train)
            tb_writer.add_scalar("train/lr",    cur_lr,   params.batch_idx_train)

        if (batch_idx + 1) % params.valid_interval == 0:
            model.eval()
            valid_loss = validate(params, model, sp, valid_loader, graph_compiler)
            log.info(f"  [valid] loss={valid_loss:.4f}")
            tb_writer.add_scalar("valid/loss", valid_loss, params.batch_idx_train)
            model.train()

    return tot_loss["loss"] / max(1, len(train_loader))


def validate(params, model, sp, valid_loader, graph_compiler=None):
    tot_loss = 0.0
    n        = 0
    with torch.no_grad():
        for batch in valid_loader:
            loss = compute_loss(
                params, model, sp, batch, is_training=False,
                graph_compiler=graph_compiler,
            )
            tot_loss += loss.item()
            n        += 1
    return tot_loss / max(1, n)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="finetuning/finetune_config.yaml")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from last checkpoint in exp_dir")
    # Optional CLI overrides
    parser.add_argument("--base-lr",               type=float, default=None)
    parser.add_argument("--num-epochs",             type=int,   default=None)
    parser.add_argument("--max-duration",           type=float, default=None)
    parser.add_argument("--exp-dir",                type=str,   default=None)
    parser.add_argument("--train-cuts",             type=str,   default=None)
    parser.add_argument("--dev-cuts",               type=str,   default=None)
    parser.add_argument("--pretrained-checkpoint",  type=str,   default=None)
    parser.add_argument("--bpe-model",              type=str,   default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ── Params ────────────────────────────────────────────────────────────
    params = build_zipformer_params(cfg)
    params.exp_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(f"{params.exp_dir}/log/finetune")

    # ── BPE model ─────────────────────────────────────────────────────────
    sp = spm.SentencePieceProcessor()
    sp.load(cfg["bpe_model"])
    params.vocab_size = sp.get_piece_size()
    log.info(f"BPE vocab size: {params.vocab_size}")

    # ── Build model ───────────────────────────────────────────────────────
    log.info("Building model...")
    model = get_model(params)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"Model parameters: {num_params:.1f}M")
    model = model.to(device)

    # ── Load pretrained weights ───────────────────────────────────────────
    if args.resume:
        # Find the last saved checkpoint in exp_dir
        checkpoints = sorted(params.exp_dir.glob("epoch-*.pt"))
        if checkpoints:
            ckpt_path = checkpoints[-1]
            log.info(f"Resuming from {ckpt_path}")
            load_checkpoint(str(ckpt_path), model)
            # Parse epoch from filename
            params.start_epoch = int(ckpt_path.stem.split("-")[1]) + 1
        else:
            log.warning("No checkpoint found in exp_dir — loading pretrained weights")
            load_checkpoint(cfg["pretrained_checkpoint"], model)
    else:
        log.info(f"Loading pretrained weights from {cfg['pretrained_checkpoint']}")
        load_checkpoint(cfg["pretrained_checkpoint"], model)

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    from icefall.optim import Eden, ScaledAdam

    optimizer = ScaledAdam(
        model.parameters(),
        lr=params.base_lr,
        betas=(0.9, 0.98),
        clipping_scale=2,
        weight_decay=params.weight_decay,
    )

    scheduler = Eden(
        optimizer,
        lr_batches=params.lr_batches,
        lr_epochs=params.lr_epochs,
        warmup_batches=params.warmup_batches,
    )

    # ── Data loaders ──────────────────────────────────────────────────────
    log.info("Building data loaders...")
    train_loader = build_dataloader(cfg["train_cuts"], cfg, is_train=True)
    valid_loader = build_dataloader(cfg["dev_cuts"],   cfg, is_train=False)

    # ── TensorBoard ───────────────────────────────────────────────────────
    tb_writer = SummaryWriter(log_dir=str(params.exp_dir / "tensorboard"))

    # ── Training loop ─────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Estimate steps per epoch (used for logging progress)
    params.steps_per_epoch = 1000   # updated after first epoch
    params.log_interval    = cfg.get("log_interval",   50)
    params.valid_interval  = cfg.get("valid_interval", 500)
    params.batch_idx_train = 0

    log.info(f"Starting finetuning for {params.num_epochs} epochs")
    log.info(f"  LR={params.base_lr}  max_duration={cfg.get('max_duration', 300)}s")

    best_valid_loss = float("inf")

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        params.cur_epoch = epoch

        train_loss = train_one_epoch(
            params=params,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sp=sp,
            train_loader=train_loader,
            valid_loader=valid_loader,
            scaler=scaler,
            tb_writer=tb_writer,
        )

        valid_loss = validate(params, model, sp, valid_loader)
        log.info(
            f"Epoch {epoch} done | train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f}"
        )
        tb_writer.add_scalar("epoch/train_loss", train_loss, epoch)
        tb_writer.add_scalar("epoch/valid_loss", valid_loss, epoch)

        # Save checkpoint every N epochs
        if epoch % cfg.get("save_every_n", 2) == 0 or epoch == params.num_epochs:
            ckpt_path = params.exp_dir / f"epoch-{epoch}.pt"
            save_checkpoint(
                filename=str(ckpt_path),
                model=model,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=None,
                scaler=scaler,
                rank=0,
            )
            log.info(f"Saved checkpoint: {ckpt_path}")

            # Keep only last K checkpoints
            keep_last_k = cfg.get("keep_last_k", 5)
            all_ckpts = sorted(params.exp_dir.glob("epoch-*.pt"))
            for old in all_ckpts[:-keep_last_k]:
                old.unlink()
                log.info(f"Removed old checkpoint: {old}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_path = params.exp_dir / "best-valid-loss.pt"
            save_checkpoint(
                filename=str(best_path),
                model=model,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=None,
                scaler=scaler,
                rank=0,
            )
            log.info(f"New best valid loss {best_valid_loss:.4f} — saved {best_path}")

    tb_writer.close()
    log.info("Finetuning complete!")
    log.info(f"Best valid loss: {best_valid_loss:.4f}")
    log.info(
        f"Next: bash finetuning/export_onnx.sh --epoch {params.num_epochs} --avg 5"
    )


if __name__ == "__main__":
    main()
