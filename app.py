"""
app.py — Streamlit web application for Plant Disease Detection.

Features:
  • Upload a leaf image → get disease prediction + confidence
  • GradCAM++ heatmap showing which leaf regions drove the prediction
  • Disease info panel (severity + treatment recommendations)
  • Model architecture overview sidebar
"""

import os
import sys
import io
import re
import numpy as np
import torch
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Allow imports from the src package
sys.path.insert(0, os.path.dirname(__file__))
from src.model import HybridCNNTransformer
from src.dataset import get_val_transforms
from src.gradcam import GradCAMPlusPlus, overlay_heatmap
from src.utils import get_device, CLASS_NAMES, DISEASE_INFO, load_checkpoint

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌿 PlantShield — AI Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS (dark, premium look)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%); }

.plantshield-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00d2ff);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s infinite linear;
    text-align: center;
    margin-bottom: 0.2rem;
}
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }

.subtitle {
    text-align: center; color: #8892b0;
    font-size: 1.05rem; margin-bottom: 2rem;
}

.prediction-card {
    background: linear-gradient(135deg, #1e1e3c, #2d2d5a);
    border: 1px solid #3a3a7c;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.4);
}

.healthy-badge {
    background: linear-gradient(135deg, #00b09b, #96c93d);
    color: white; padding: 4px 14px; border-radius: 20px;
    font-weight: 600; font-size: 0.85rem; display: inline-block;
}

.disease-badge {
    background: linear-gradient(135deg, #f7971e, #ffd200);
    color: #1a1a1a; padding: 4px 14px; border-radius: 20px;
    font-weight: 600; font-size: 0.85rem; display: inline-block;
}

.severe-badge {
    background: linear-gradient(135deg, #e94560, #c0392b);
    color: white; padding: 4px 14px; border-radius: 20px;
    font-weight: 600; font-size: 0.85rem; display: inline-block;
}

.metric-box {
    background: #1c1c3a; border-radius: 12px; padding: 1rem;
    text-align: center; border: 1px solid #2d2d5a;
}

.metric-value {
    font-size: 1.8rem; font-weight: 700; color: #00d2ff;
}

.metric-label {
    font-size: 0.8rem; color: #8892b0; margin-top: 4px;
}

.stButton>button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 10px;
    padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem;
    transition: all 0.3s; width: 100%;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102,126,234,0.5);
}

.info-box {
    background: #1c1c3a; border-left: 4px solid #00d2ff;
    border-radius: 0 12px 12px 0; padding: 1rem 1.2rem; margin: 1rem 0;
}

.arch-step {
    background: #1c1c3a; border-radius: 10px; padding: 0.6rem 1rem;
    margin: 0.4rem 0; border-left: 3px solid #3a7bd5;
    color: #ccd6f6; font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(checkpoint_path: str | None, num_classes: int = 38):
    device = get_device()

    # Try to load class names from checkpoint first
    ckpt_class_names = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            import torch as _torch
            _ckpt = _torch.load(checkpoint_path, map_location="cpu")
            ckpt_class_names = _ckpt.get("class_names", None)
            # Infer num_classes from saved head weight shape
            head_w = _ckpt.get("model_state", {}).get("head.1.weight")
            if head_w is not None:
                num_classes = head_w.shape[0]
        except Exception:
            pass

    model = HybridCNNTransformer(num_classes=num_classes, pretrained_cnn=True)

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            load_checkpoint(checkpoint_path, model, device=str(device))
            st.sidebar.success(f"✓ Loaded: {os.path.basename(checkpoint_path)}")
        except Exception as e:
            st.sidebar.warning(f"⚠ Could not load checkpoint: {e}\nUsing ImageNet-pretrained CNN stem.")
    else:
        st.sidebar.info("No fine-tuned checkpoint found. Using pretrained CNN weights for demo.")

    model = model.to(device)
    model.eval()

    # Use checkpoint class names if available, else fall back to CLASS_NAMES slice
    class_names = ckpt_class_names if ckpt_class_names else CLASS_NAMES[:num_classes]
    return model, device, class_names


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, image_tensor, device, top_k: int = 5):
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)
    probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_k_idx   = probs.argsort()[-top_k:][::-1]
    top_k_probs = probs[top_k_idx]
    return probs, top_k_idx, top_k_probs


def compute_gradcam(model, image_tensor, device, class_idx: int):
    gcam    = GradCAMPlusPlus(model)
    heatmap = gcam(image_tensor.to(device), class_idx=class_idx)
    gcam.remove_hooks()
    return heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    ckpt_path = st.text_input(
        "Checkpoint path",
        value="checkpoints/best_model.pth",
        help="Path to your fine-tuned model checkpoint"
    )

    show_gradcam = st.checkbox("Show GradCAM++ Heatmap", value=True)
    alpha = st.slider("Heatmap overlay α", 0.1, 0.9, 0.5, 0.05)
    top_k_display = st.slider("Top-K predictions to show", 3, 10, 5)

    st.markdown("---")
    st.markdown("## 🏗 Architecture")
    st.markdown('<div class="arch-step">① EfficientNet-B0 CNN Stem<br><small style="color:#8892b0">Local texture & lesion features</small></div>', unsafe_allow_html=True)
    st.markdown('<div class="arch-step">② Patch Tokenisation (7×7=49 patches)</div>', unsafe_allow_html=True)
    st.markdown('<div class="arch-step">③ [CLS] Token + Positional Encoding</div>', unsafe_allow_html=True)
    st.markdown('<div class="arch-step">④ 4× Transformer Blocks (8 heads)<br><small style="color:#8892b0">Global context attention</small></div>', unsafe_allow_html=True)
    st.markdown('<div class="arch-step">⑤ Classification Head → 38 classes</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.markdown("**PlantVillage** — 87,000 images, 38 classes, 14 crop species.")

    st.markdown("---")




# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="plantshield-title">🌿 PlantShield AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Automated Plant Disease Detection · Hybrid CNN-Transformer + SimCLR + GradCAM++</p>', unsafe_allow_html=True)

# Load model once
model, device, class_names = load_model(ckpt_path)

transform = get_val_transforms()

# ── Upload zone ───────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 2], gap="large")

with col_upload:
    st.markdown("### 📸 Upload Leaf Image")
    uploaded = st.file_uploader(
        "Drag & drop or click to upload",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="Uploaded Leaf", use_container_width=True)
        run_btn = st.button("🔍 Analyse Disease", key="analyse_btn")
    else:
        st.info("📌 Upload a leaf image to start. Supports JPG, PNG, WebP.")
        run_btn = False

# ── Results panel ─────────────────────────────────────────────────────────────
with col_result:
    if uploaded and run_btn:
        with st.spinner("🧠 Running AI analysis..."):
            # Preprocess
            img_tensor = transform(pil_img).unsqueeze(0)

            # Predict
            probs, top_k_idx, top_k_probs = predict(model, img_tensor, device, top_k=top_k_display)
            pred_idx   = int(top_k_idx[0])
            pred_class = class_names[pred_idx]
            pred_prob  = float(top_k_probs[0])

            # ── Parse plant / condition from any PlantVillage naming style ──
            # e.g. Tomato_Bacterial_spot  |  Tomato__Target_Spot  |  Potato___Early_blight
            parts = re.split(r'_{2,}', pred_class, maxsplit=1)
            if len(parts) == 2:
                plant     = parts[0].replace('_', ' ').strip()
                condition = parts[1].replace('_', ' ').strip()
            else:
                # single underscore only — first token is the plant
                tokens    = pred_class.split('_', 1)
                plant     = tokens[0]
                condition = tokens[1].replace('_', ' ').strip() if len(tokens) > 1 else ''
            is_healthy = 'healthy' in pred_class.lower()

            # ── Disease info — fuzzy match against DISEASE_INFO keys ─────────
            info = DISEASE_INFO.get(pred_class)
            if info is None:
                cleaned = pred_class.lower().replace('_', '')
                for k, v in DISEASE_INFO.items():
                    if k.lower().replace('_', '') == cleaned:
                        info = v
                        break
            if info is None:
                info = {'treatment': 'Consult a local agronomist for treatment advice.',
                        'severity': 'Unknown'}
            severity  = info["severity"]

        # ── Prediction card ───────────────────────────────────────────────────
        sev_color = {"None": "#2ecc71", "Moderate": "#f39c12",
                     "High": "#e74c3c", "Severe": "#c0392b", "Fatal": "#8e44ad"}.get(severity, "#8892b0")

        badge_html = ('<span class="healthy-badge">✅ Healthy</span>' if is_healthy
                      else f'<span class="disease-badge">⚠ {severity} Severity</span>')

        st.markdown(f"""
        <div class="prediction-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                <div>
                    <div style="color:#8892b0; font-size:0.8rem; margin-bottom:4px;">DETECTED PLANT</div>
                    <div style="color:#ccd6f6; font-size:1.3rem; font-weight:700;">{plant}</div>
                </div>
                <div>{badge_html}</div>
            </div>
            <div style="color:#8892b0; font-size:0.8rem; margin-bottom:4px;">CONDITION</div>
            <div style="color:{'#2ecc71' if is_healthy else '#e94560'}; font-size:1.1rem; font-weight:600; margin-bottom:1rem;">{condition}</div>
            <div style="background:#0d0d24; border-radius:8px; padding:0.8rem; margin-bottom:0.5rem;">
                <div style="color:#8892b0; font-size:0.75rem;">CONFIDENCE</div>
                <div style="color:#00d2ff; font-size:1.6rem; font-weight:700;">{pred_prob:.1%}</div>
                <div style="background:#2d2d5a; border-radius:4px; height:6px; margin-top:6px;">
                    <div style="background:linear-gradient(90deg,#667eea,#00d2ff); width:{pred_prob*100:.1f}%; height:6px; border-radius:4px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Disease info ─────────────────────────────────────────────────────
        if not is_healthy:
            st.markdown(f"""
            <div class="info-box">
                <div style="color:#00d2ff; font-weight:600; margin-bottom:0.4rem;">🌱 Recommended Treatment</div>
                <div style="color:#ccd6f6; font-size:0.9rem;">{info['treatment']}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Top-K bar chart ───────────────────────────────────────────────────
        st.markdown("#### 📊 Confidence Distribution")
        fig_chart, ax = plt.subplots(figsize=(8, 3))
        fig_chart.patch.set_facecolor("#0f0f23")
        ax.set_facecolor("#1a1a3e")

        names = [re.sub(r'_{2,}', ' | ', class_names[i]).replace('_', ' ')[:35] for i in top_k_idx]
        bar_colors = ["#00d2ff" if i == 0 else "#3a7bd5" for i in range(len(top_k_idx))]
        bars = ax.barh(names[::-1], top_k_probs[::-1], color=bar_colors[::-1], height=0.6)
        ax.set_xlabel("Confidence", color="white")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d2d5a")
        ax.set_xlim(0, 1)
        plt.tight_layout()
        st.pyplot(fig_chart, use_container_width=True)
        plt.close(fig_chart)

        # ── GradCAM++ ─────────────────────────────────────────────────────────
        if show_gradcam:
            st.markdown("#### 🔥 GradCAM++ Saliency Map")
            with st.spinner("Computing gradient-weighted activation map..."):
                # Enable grad temporarily
                for p in model.parameters():
                    p.requires_grad_(True)
                heatmap = compute_gradcam(model, img_tensor, device, pred_idx)
                for p in model.parameters():
                    p.requires_grad_(False)

            orig_np = np.array(pil_img.resize((224, 224)))
            overlay = overlay_heatmap(orig_np, heatmap, alpha=alpha)

            col_orig, col_heat, col_overlay = st.columns(3)
            with col_orig:
                st.image(orig_np, caption="Original", use_container_width=True)
            with col_heat:
                heat_rgb = plt.cm.jet(heatmap)[:, :, :3]
                heat_rgb = (heat_rgb * 255).astype(np.uint8)
                st.image(heat_rgb, caption="Activation Map", use_container_width=True)
            with col_overlay:
                st.image(overlay, caption="GradCAM++ Overlay", use_container_width=True)

            st.caption("🔴 Red = most influential regions | 🔵 Blue = least influential regions")

    elif not uploaded:
        # Hero placeholder
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1e1e3c,#2d2d5a); border-radius:20px;
                    padding:3rem; text-align:center; border:1px dashed #3a3a7c; margin-top:2rem;">
            <div style="font-size:4rem; margin-bottom:1rem;">🌿</div>
            <div style="color:#ccd6f6; font-size:1.2rem; font-weight:600; margin-bottom:0.5rem;">
                Upload a leaf image to get started
            </div>
            <div style="color:#8892b0; font-size:0.9rem;">
                Supports 14 plant species · 38 disease conditions<br>
                Powered by Hybrid CNN-Transformer + SimCLR
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8892b0; font-size:0.8rem; padding: 1rem 0;">
    PlantShield AI · Novel Hybrid CNN-Transformer with SimCLR Self-Supervised Pre-training &amp; GradCAM++ Explainability<br>
    Dissertation Project · PlantVillage Dataset · 38 Disease Classes
</div>
""", unsafe_allow_html=True)
