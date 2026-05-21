"""
gradcam.py — GradCAM++ explainability for the Hybrid CNN-Transformer.

GradCAM++ Reference:
  Chattopadhay et al., "Grad-CAM++: Improved Visual Explanations for
  Deep Convolutional Networks", WACV 2018.

Why GradCAM++ over plain GradCAM?
  • GradCAM++ uses a weighted combination of partial derivatives of the loss
    w.r.t. each activation map — giving sharper localisation for multiple
    object instances (e.g., several lesions on the same leaf).
  • Particularly valuable for agricultural explainability where a plant
    pathologist needs to verify which exact lesions triggered the diagnosis.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


class GradCAMPlusPlus:
    """
    Gradient-weighted Class Activation Mapping++ for the CNN stem's last
    convolutional block.

    Usage:
        gcam = GradCAMPlusPlus(model)
        heatmap = gcam(image_tensor, class_idx=pred_class)
        gcam.remove_hooks()

    Args:
        model      : HybridCNNTransformer instance (in eval mode)
        target_layer: the CNN sub-module whose activations to hook;
                      defaults to the last stage of the CNN stem.
    """

    def __init__(self, model: torch.nn.Module, target_layer=None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            stages = list(model.cnn_stem.stages.children())
            # Use the LAST stage of the CNN stem — it has the richest
            # semantic features and strongest gradient signal, producing
            # heatmaps that clearly highlight diseased leaf regions.
            # The 7×7 spatial resolution is upsampled to input size via
            # bilinear interpolation in the __call__ method.
            target_parent = stages[-1]
            target_layer = None
            for m in target_parent.modules():
                if isinstance(m, torch.nn.Conv2d):
                    target_layer = m       # last Conv2d inside that stage
            if target_layer is None:
                # fallback: find last Conv2d across whole stem
                for m in model.cnn_stem.stages.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        target_layer = m

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ── Hooks ─────────────────────────────────────────────────────────────────
    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    # ── GradCAM++ computation ─────────────────────────────────────────────────
    def __call__(
        self,
        image_tensor: torch.Tensor,          # (1, 3, H, W)
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Returns a (H, W) numpy array with values in [0, 1] representing
        the GradCAM++ saliency map (before overlay).
        """
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(image_tensor)         # (1, C)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # ── GradCAM++ weights ─────────────────────────────────────────────────
        # A: activation maps  (1, C_feat, H_feat, W_feat)
        # dY: gradients        (1, C_feat, H_feat, W_feat)
        A  = self._activations   # (1, K, h, w)
        dY = self._gradients     # (1, K, h, w)

        # α_k^c = ∂²Y^c / (2·∂²Y^c + Σ_{a,b} A_k · ∂³Y^c ) — simplified form:
        dY2 = dY ** 2
        dY3 = dY ** 3
        sum_A = A.sum(dim=(2, 3), keepdim=True)       # Σ A_k over spatial
        alpha = dY2 / (2.0 * dY2 + sum_A * dY3 + 1e-7)

        # Weights: w_k^c = Σ_{i,j} α_k^c · ReLU(∂Y^c/∂A_k)
        weights = (alpha * F.relu(dY)).sum(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * A).sum(dim=1, keepdim=True)   # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input size
        H, W = image_tensor.shape[2], image_tensor.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam   # (H, W) float32 in [0,1]


# ─────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────────────────────────────────────────
def overlay_heatmap(
    original_image: np.ndarray,    # (H, W, 3)  uint8 RGB
    heatmap: np.ndarray,           # (H, W)     float in [0,1]
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a heatmap onto the original image. Returns (H, W, 3) uint8 RGB."""
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)        # BGR
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = (alpha * colored_rgb + (1 - alpha) * original_image).astype(np.uint8)
    return overlay


def visualize_gradcam(
    model,
    image_tensor: torch.Tensor,        # (1, 3, H, W) normalised
    original_pil: Image.Image,
    class_names: list,
    true_label: int | None = None,
    save_path: str | None = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Create a three-panel figure:
      [ Original leaf ] [ GradCAM++ heatmap ] [ Prediction info ]

    Args:
        model        : HybridCNNTransformer in eval mode
        image_tensor : pre-processed tensor (1, 3, H, W)
        original_pil : PIL Image of the original (un-normalised) leaf
        class_names  : list of class name strings (length = num_classes)
        true_label   : optional ground-truth class index
        save_path    : if provided, saves figure to this path
        figsize      : matplotlib figure size

    Returns:
        matplotlib Figure
    """
    gcam = GradCAMPlusPlus(model)

    with torch.no_grad():
        logits = model(image_tensor.to(next(model.parameters()).device))
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx  = int(probs.argmax())
    pred_prob = probs[pred_idx]

    # Compute heatmap (requires grad; temporary enable)
    heatmap = gcam(image_tensor, class_idx=pred_idx)
    gcam.remove_hooks()

    orig_np = np.array(original_pil.resize((224, 224)))
    overlay = overlay_heatmap(orig_np, heatmap)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#0f3460")

    axes[0].imshow(orig_np)
    axes[0].set_title("Original Leaf", color="white", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    im = axes[1].imshow(overlay)
    axes[1].set_title("GradCAM++ Saliency", color="#e94560", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    # Top-5 predictions bar chart
    top5_idx   = probs.argsort()[-5:][::-1]
    top5_probs = probs[top5_idx]
    top5_names = [class_names[i].replace("___", "\n").replace("_", " ")[:30] for i in top5_idx]
    colors     = ["#e94560" if i == pred_idx else "#0f3460" for i in top5_idx]

    bars = axes[2].barh(range(5), top5_probs[::-1], color=colors[::-1], height=0.6)
    axes[2].set_yticks(range(5))
    axes[2].set_yticklabels(top5_names[::-1], color="white", fontsize=8)
    axes[2].set_xlabel("Confidence", color="white", fontsize=10)
    axes[2].set_title("Top-5 Predictions", color="white", fontsize=13, fontweight="bold")
    axes[2].set_xlim(0, 1)
    axes[2].xaxis.label.set_color("white")
    axes[2].tick_params(axis="x", colors="white")

    pred_name = class_names[pred_idx].replace("___", " | ").replace("_", " ")
    status = "✓ Correct" if true_label == pred_idx else "✗ Incorrect"
    title_color = "#2ecc71" if true_label == pred_idx else "#e94560"
    title = f"Pred: {pred_name}\nConf: {pred_prob:.1%}"
    if true_label is not None:
        title += f"\n{status}"

    fig.suptitle(title, color=title_color, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())

    return fig


import os
