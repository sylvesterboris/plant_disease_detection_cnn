"""
evaluate.py — Comprehensive model evaluation on the test split.

Produces:
  • Overall accuracy, macro F1, weighted F1
  • Per-class precision / recall / F1 table
  • Confusion matrix heatmap (saved as PNG)
  • Top-5 worst-predicted samples (for error analysis)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, top_k_accuracy_score,
)

from .model import HybridCNNTransformer
from .dataset import build_dataloaders
from .utils import load_checkpoint, get_device, CLASS_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    class_names: list,
) -> dict:
    """
    Run the model on `loader` and compute classification metrics.

    Returns:
        dict with keys: accuracy, macro_f1, weighted_f1, top5_acc,
                        all_preds, all_labels, all_probs, report_df
    """
    model.eval()
    all_labels, all_preds, all_probs_list = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        logits = model(images)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs_list.append(probs)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.vstack(all_probs_list)

    acc        = accuracy_score(all_labels, all_preds)
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1= f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    n_classes  = all_probs.shape[1]
    k          = min(5, n_classes)
    top5_acc   = top_k_accuracy_score(
        all_labels, all_probs, k=k,
        labels=list(range(n_classes))
    )

    report     = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df  = pd.DataFrame(report).transpose().round(4)

    return {
        "accuracy":    acc,
        "macro_f1":    macro_f1,
        "weighted_f1": weighted_f1,
        "top5_acc":    top5_acc,
        "all_preds":   all_preds,
        "all_labels":  all_labels,
        "all_probs":   all_probs,
        "report_df":   report_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    class_names: list,
    save_path: str | None = None,
    figsize: tuple = (20, 18),
) -> plt.Figure:
    """
    Plots a normalised confusion matrix heatmap.
    Short class names (plant + disease) are used for readability.
    """
    cm = confusion_matrix(all_labels, all_preds, normalize="true")

    short_names = [c.replace("___", "\n").replace("_", " ")
                   for c in class_names]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=short_names, yticklabels=short_names,
        linewidths=0.4, linecolor="#cccccc",
        ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label",      fontsize=12, fontweight="bold")
    ax.set_title("Normalised Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Per-class F1 bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_per_class_f1(
    report_df: pd.DataFrame,
    class_names: list,
    save_path: str | None = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """Horizontal bar chart of per-class F1 scores, sorted descending."""
    per_class = report_df.loc[class_names, "f1-score"].sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#2ecc71" if v >= 0.9 else "#e67e22" if v >= 0.7 else "#e74c3c"
              for v in per_class.values]
    ax.barh(per_class.index, per_class.values, color=colors, height=0.7)
    ax.axvline(per_class.values.mean(), color="navy", linestyle="--",
               linewidth=1.5, label=f"Mean F1: {per_class.values.mean():.3f}")
    ax.set_xlabel("F1-score", fontsize=12)
    ax.set_title("Per-class F1-scores", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=10)
    ax.set_yticks(range(len(per_class.index)))
    ax.set_yticklabels(
        [n.replace("___", " | ").replace("_", " ") for n in per_class.index],
        fontsize=8
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Training curve plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(
    history: dict,
    save_path: str | None = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Plots loss and accuracy curves for train & validation."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train")
    ax1.plot(epochs, history["val_loss"],   "r-o", markersize=3, label="Val")
    ax1.set_title("Loss Curve", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train")
    ax2.plot(epochs, history["val_acc"],   "r-o", markersize=3, label="Val")
    ax2.set_title("Accuracy Curve", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Training & Validation Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_full_evaluation(
    checkpoint_path: str,
    data_root: str,
    output_dir: str = "outputs",
    batch_size: int = 32,
    num_workers: int = 4,
    num_classes: int = 38,
    device: torch.device = None,
) -> dict:
    """
    Load model from checkpoint, run evaluation on the test set,
    and save all plots to `output_dir`.

    Returns:
        metrics dict from evaluate_model()
    """
    if device is None:
        device = get_device()

    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint — infer num_classes from the saved head weights
    ckpt           = torch.load(checkpoint_path, map_location=device)
    class_names    = ckpt.get("class_names", CLASS_NAMES)
    # The head.1.weight shape tells us how many outputs the model actually has
    head_weight    = ckpt["model_state"].get("head.1.weight")
    model_classes  = head_weight.shape[0] if head_weight is not None else len(class_names)
    model = HybridCNNTransformer(num_classes=model_classes, pretrained_cnn=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    # Load test data
    _, _, test_loader, _ = build_dataloaders(
        data_root, batch_size=batch_size, num_workers=num_workers
    )

    # Evaluate
    metrics = evaluate_model(model, test_loader, device, class_names)

    print("\n" + "="*60)
    print(f"  Test Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1    : {metrics['weighted_f1']:.4f}")
    print(f"  Top-5 Accuracy : {metrics['top5_acc']:.4f}")
    print("="*60 + "\n")
    print(metrics["report_df"].to_string())

    # Save plots
    plot_confusion_matrix(
        metrics["all_labels"], metrics["all_preds"], class_names,
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )
    plot_per_class_f1(
        metrics["report_df"], class_names,
        save_path=os.path.join(output_dir, "per_class_f1.png")
    )

    # Training curves from checkpoint history
    if "history" in ckpt:
        plot_training_curves(
            ckpt["history"],
            save_path=os.path.join(output_dir, "training_curves.png")
        )

    return metrics
