"""
simclr.py — SimCLR self-supervised contrastive pre-training.

SimCLR Reference: Chen et al., "A Simple Framework for Contrastive
Learning of Visual Representations", ICML 2020.

Key idea:
  • Two differently-augmented views of the same image should have similar
    representations; views from different images should be dissimilar.
  • NT-Xent loss maximises agreement between positive pairs while pushing
    negative pairs away — all in a single batch.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .model import SimCLRModel
from .utils import get_logger, save_checkpoint, set_seed, get_device


# ─────────────────────────────────────────────────────────────────────────────
# NT-Xent Loss (Normalised Temperature-scaled Cross Entropy)
# ─────────────────────────────────────────────────────────────────────────────
class NTXentLoss(nn.Module):
    """
    NT-Xent contrastive loss for a batch of 2N embeddings (N positive pairs).

    For each anchor z_i, the positive is z_j (the other view of the same image),
    and all other 2(N-1) embeddings are treated as negatives.

    Args:
        temperature: temperature scaling τ (0.5 is the SimCLR default)
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: normalised embeddings for view 1, shape (N, D)
            z2: normalised embeddings for view 2, shape (N, D)
        Returns:
            scalar loss
        """
        N = z1.size(0)
        device = z1.device

        # Concatenate all 2N representations: [z1 || z2]
        z = torch.cat([z1, z2], dim=0)          # (2N, D)

        # Similarity matrix (dot product, pre-normalised vectors)
        sim = torch.mm(z, z.T) / self.temperature  # (2N, 2N)

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pair indices
        # z1[i] ↔ z2[i]  →  positive for i is at position N+i
        # z2[i] ↔ z1[i]  →  positive for i is at position i
        pos_i = torch.arange(N, 2 * N, device=device)   # indices of z2 for z1 rows
        pos_j = torch.arange(0, N, device=device)        # indices of z1 for z2 rows
        labels = torch.cat([pos_i, pos_j], dim=0)         # (2N,)

        loss = F.cross_entropy(sim, labels)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Pre-training loop
# ─────────────────────────────────────────────────────────────────────────────
def pretrain(
    data_root: str,
    checkpoint_dir: str = "checkpoints",
    num_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    temperature: float = 0.5,
    num_workers: int = 4,
    seed: int = 42,
    device: torch.device = None,
) -> str:
    """
    Run SimCLR self-supervised pre-training.

    Args:
        data_root      : path containing the PlantVillage images (all unlabelled)
        checkpoint_dir : directory to save encoder weights
        num_epochs     : total pre-training epochs
        batch_size     : number of image PAIRS per batch (effective batch = 2×)
        lr             : peak learning rate for AdamW
        weight_decay   : L2 regularisation
        temperature    : NT-Xent τ
        num_workers    : DataLoader workers
        seed           : random seed
        device         : torch device (auto-detected if None)

    Returns:
        path to the saved encoder checkpoint
    """
    from .dataset import build_simclr_dataloader

    set_seed(seed)
    if device is None:
        device = get_device()

    logger = get_logger("simclr", os.path.join(checkpoint_dir, "pretrain.log"))
    logger.info(f"Starting SimCLR pre-training | device={device} | epochs={num_epochs}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    loader = build_simclr_dataloader(data_root, batch_size=batch_size,
                                     num_workers=num_workers)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SimCLRModel(pretrained_cnn=True).to(device)
    loss_fn = NTXentLoss(temperature=temperature)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_loss = float("inf")
    best_ckpt_path = os.path.join(checkpoint_dir, "simclr_encoder_best.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    history = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Pretrain Epoch {epoch}/{num_epochs}", leave=False)
        for view1, view2, _ in pbar:
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()
            z1, z2 = model(view1, view2)
            loss = loss_fn(z1, z2)
            loss.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / len(loader)
        history.append(avg_loss)
        logger.info(f"Epoch {epoch:>4d}/{num_epochs} | NT-Xent Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best encoder
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model_state": model.encoder.state_dict(),
                    "history": history,
                },
                best_ckpt_path,
            )
            logger.info(f"  ✓ New best loss {best_loss:.4f} — encoder saved to {best_ckpt_path}")

    logger.info("Pre-training complete.")
    return best_ckpt_path
