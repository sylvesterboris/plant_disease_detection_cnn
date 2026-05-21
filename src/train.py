"""
train.py — Supervised fine-tuning of the Hybrid CNN-Transformer.

Pipeline:
  1. Load pre-trained encoder weights from SimCLR (optional — can also train
     from scratch with pretrained_cnn=True CNN stem weights).
  2. Attach a fresh classification head.
  3. Fine-tune for N epochs using AdamW + CosineAnnealing + label smoothing.
  4. Save the best model checkpoint based on validation accuracy.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from .model import HybridCNNTransformer
from .dataset import build_dataloaders
from .utils import (
    set_seed, get_logger, save_checkpoint, load_checkpoint, get_device
)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds.argmax(dim=1) == labels).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch routines
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler:                # mixed-precision (GPU only)
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_acc  += accuracy(logits.detach(), labels)

    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    for images, labels in tqdm(loader, desc="  val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)

    return total_loss / len(loader), total_acc / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Full training pipeline
# ─────────────────────────────────────────────────────────────────────────────
def train(
    data_root: str,
    checkpoint_dir: str = "checkpoints",
    pretrain_ckpt: str | None = None,
    num_classes: int = 38,
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    device: torch.device = None,
    freeze_cnn_epochs: int = 5,
) -> dict:
    """
    Fine-tune the Hybrid CNN-Transformer on PlantVillage.

    Args:
        data_root        : directory with train/ val/ test/ sub-folders
        checkpoint_dir   : where to save best model
        pretrain_ckpt    : path to SimCLR encoder checkpoint (None = train from scratch)
        num_classes      : number of disease classes
        num_epochs       : total supervised fine-tuning epochs
        batch_size       : images per batch
        lr               : peak LR for AdamW
        weight_decay     : L2 regularisation
        label_smoothing  : label smoothing factor (CrossEntropy)
        num_workers      : DataLoader workers
        seed             : random seed
        device           : torch device (auto-detected if None)
        freeze_cnn_epochs: freeze CNN stem for first N epochs (warmup strategy)

    Returns:
        dict with train/val loss and accuracy history
    """
    set_seed(seed)
    if device is None:
        device = get_device()

    log_path = os.path.join(checkpoint_dir, "finetune.log")
    logger = get_logger("train", log_path)
    logger.info(f"Fine-tuning | device={device} | epochs={num_epochs}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Loaders ───────────────────────────────────────────────────────────────
    train_loader, val_loader, _, class_names = build_dataloaders(
        data_root, batch_size=batch_size, num_workers=num_workers
    )
    num_classes = len(class_names)   # auto-detected from dataset
    logger.info(f"Classes: {num_classes} | train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HybridCNNTransformer(num_classes=num_classes, pretrained_cnn=True)

    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        ckpt = torch.load(pretrain_ckpt, map_location="cpu")
        # Load encoder state (excludes head — loaded strictly=False on purpose)
        model.load_state_dict(ckpt["model_state"], strict=False)
        logger.info(f"Loaded SimCLR encoder weights from: {pretrain_ckpt}")
    else:
        logger.info("No SimCLR checkpoint found — training with ImageNet-pretrained CNN stem only.")

    model = model.to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Mixed precision (CUDA only)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── History ───────────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_val_acc = 0.0
    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Gradual unfreeze: freeze CNN stem during warmup epochs
        if epoch <= freeze_cnn_epochs:
            for param in model.cnn_stem.parameters():
                param.requires_grad = False
        else:
            for param in model.cnn_stem.parameters():
                param.requires_grad = True

        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        vl_loss, vl_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:>3d}/{num_epochs} | "
            f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
            f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.1e} | {elapsed:.1f}s"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "class_names": class_names,
                    "history": history,
                },
                best_ckpt_path,
            )
            logger.info(f"  ✓ New best val acc {best_val_acc:.4f} — saved to {best_ckpt_path}")

    logger.info(f"Fine-tuning complete. Best val acc: {best_val_acc:.4f}")
    return history
