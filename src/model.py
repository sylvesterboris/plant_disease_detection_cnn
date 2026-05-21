"""
model.py — Novel Hybrid CNN-Transformer architecture.

Architecture overview:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Input: (B, 3, 224, 224)                                         │
  │                                                                  │
  │  ┌──────────────────────────────────────────────────────────┐   │
  │  │  CNN Stem  (EfficientNet-B0 first 5 stages, pretrained)  │   │
  │  │  Output: (B, 320, 7, 7)  →  flattened to (B, 49, 320)   │   │
  │  └──────────────────────────────────────────────────────────┘   │
  │                       ↓                                          │
  │  ┌──────────────────────────────────────────────────────────┐   │
  │  │  Learnable [CLS] token + Positional Embedding            │   │
  │  │  Sequence: (B, 50, 320)                                  │   │
  │  └──────────────────────────────────────────────────────────┘   │
  │                       ↓                                          │
  │  ┌──────────────────────────────────────────────────────────┐   │
  │  │  4 × Transformer Encoder Blocks                          │   │
  │  │    Multi-Head Self-Attention (8 heads) + FFN + LayerNorm │   │
  │  └──────────────────────────────────────────────────────────┘   │
  │                       ↓                                          │
  │         CLS token → Classification Head → logits                │
  └──────────────────────────────────────────────────────────────────┘

Why this is novel:
  • CNNs are excellent at local texture features (critical for spotting lesions).
  • Transformers capture long-range spatial dependencies (disease spread patterns
    across the entire leaf surface) — something pure CNNs cannot do efficiently.
  • Combining both via a CNN stem → Transformer approach is architecturally novel
    for plant disease classification on PlantVillage.
"""

import math
import torch
import torch.nn as nn
import timm


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CNN Stem  (EfficientNet-B0 feature extractor up to penultimate stage)
# ─────────────────────────────────────────────────────────────────────────────
class CNNStem(nn.Module):
    """
    EfficientNet-B0 up to the last MBConv stage (before global pooling).
    Output spatial map: (B, 320, 7, 7) for 224×224 input.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, features_only=True)
        # Keep stages 0-4 (out_channels = [16, 24, 40, 80, 112, 192, 320])
        # Stage 4 output channels = 320, spatial = 7×7
        self.stages = nn.Sequential(*list(backbone.children())[:7])
        self.out_channels = 320

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(x)  # (B, 320, 7, 7)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Positional Encoding for patch sequence
# ─────────────────────────────────────────────────────────────────────────────
class LearnablePositionalEncoding(nn.Module):
    """
    Fully learnable 1-D positional embedding for patch + CLS token sequence.
    """

    def __init__(self, seq_len: int, dim: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_len, dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Transformer Encoder Block
# ─────────────────────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer encoder block:
      x → LayerNorm → MultiHeadAttention → residual
        → LayerNorm → FFN (GELU) → residual
    """

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Full Hybrid CNN-Transformer —  the core novel model
# ─────────────────────────────────────────────────────────────────────────────
class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer for plant disease classification.

    Args:
        num_classes   : number of disease classes (38 for PlantVillage)
        pretrained_cnn: whether to initialise the CNN stem with ImageNet weights
        num_heads     : multi-head attention heads in each Transformer block
        num_blocks    : number of stacked Transformer blocks
        drop_rate     : dropout rate for classification head
    """

    def __init__(
        self,
        num_classes: int = 38,
        pretrained_cnn: bool = True,
        num_heads: int = 8,
        num_blocks: int = 4,
        drop_rate: float = 0.2,
    ):
        super().__init__()

        # ── CNN Stem ──────────────────────────────────────────────────────────
        self.cnn_stem = CNNStem(pretrained=pretrained_cnn)
        cnn_dim = self.cnn_stem.out_channels       # 320

        # ── CLS token ────────────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cnn_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Spatial map: 7×7 = 49 patches + 1 CLS = 50 tokens
        self.pos_enc = LearnablePositionalEncoding(seq_len=50, dim=cnn_dim)

        # ── Transformer blocks ───────────────────────────────────────────────
        self.transformer = nn.Sequential(
            *[TransformerBlock(dim=cnn_dim, num_heads=num_heads) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(cnn_dim)

        # ── Classification head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(cnn_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the CLS embedding (used internally and for SimCLR)."""
        # CNN feature map: (B, 320, 7, 7)
        feat = self.cnn_stem(x)
        B, C, H, W = feat.shape

        # Flatten spatial dims → patch sequence: (B, 49, 320)
        patches = feat.flatten(2).transpose(1, 2)

        # Prepend CLS token: (B, 50, 320)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)

        # Positional encoding + Transformer
        tokens = self.pos_enc(tokens)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # Return CLS token embedding
        return tokens[:, 0]  # (B, 320)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_emb = self.forward_features(x)
        return self.head(cls_emb)              # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Projection Head — used ONLY during SimCLR pre-training
# ─────────────────────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """
    3-layer MLP projection head mapping CLS embedding to a normalised
    128-dim space where the contrastive loss is applied.
    (Discarded after pre-training; fine-tuning uses forward_features directly.)
    """

    def __init__(self, in_dim: int = 320, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return nn.functional.normalize(z, dim=-1)  # L2-normalised


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SimCLR model = encoder + projection head
# ─────────────────────────────────────────────────────────────────────────────
class SimCLRModel(nn.Module):
    """
    Wrapper used during self-supervised pre-training.
    After pretraining, only `encoder` weights are saved/restored.
    """

    def __init__(self, num_classes: int = 38, pretrained_cnn: bool = True):
        super().__init__()
        self.encoder = HybridCNNTransformer(num_classes=num_classes,
                                            pretrained_cnn=pretrained_cnn)
        self.projector = ProjectionHead(in_dim=320)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        h1 = self.encoder.forward_features(x1)
        h2 = self.encoder.forward_features(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return z1, z2
