"""
utils.py — Utility helpers for the plant disease detection project.
Includes: seed fixing, logging, checkpoint I/O, class name mapping.
"""

import os
import random
import logging
import torch
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# PlantVillage 38-class label map
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

DISEASE_INFO = {
    "Apple___Apple_scab": {
        "treatment": "Apply fungicides (captan, myclobutanil). Remove infected leaves. Ensure good air circulation.",
        "severity": "Moderate"
    },
    "Apple___Black_rot": {
        "treatment": "Prune infected wood. Apply copper-based fungicides. Destroy mummified fruit.",
        "severity": "High"
    },
    "Apple___Cedar_apple_rust": {
        "treatment": "Apply fungicides at early leaf stage. Remove nearby cedar-apple galls.",
        "severity": "Moderate"
    },
    "Apple___healthy": {"treatment": "No treatment needed. Maintain proper irrigation and fertilization.", "severity": "None"},
    "Blueberry___healthy": {"treatment": "No treatment needed.", "severity": "None"},
    "Cherry_(including_sour)___Powdery_mildew": {
        "treatment": "Apply sulfur-based or neem oil sprays. Avoid overhead watering.",
        "severity": "Moderate"
    },
    "Cherry_(including_sour)___healthy": {"treatment": "No treatment needed.", "severity": "None"},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "treatment": "Apply triazole fungicides. Use resistant varieties. Rotate crops.",
        "severity": "High"
    },
    "Corn_(maize)___Common_rust_": {
        "treatment": "Apply fungicides early. Plant resistant hybrids.",
        "severity": "Moderate"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "treatment": "Apply foliar fungicides. Use resistant varieties. Crop rotation.",
        "severity": "High"
    },
    "Corn_(maize)___healthy": {"treatment": "No treatment needed.", "severity": "None"},
    "Grape___Black_rot": {
        "treatment": "Apply mancozeb or captan fungicides. Remove mummified berries.",
        "severity": "High"
    },
    "Grape___Esca_(Black_Measles)": {
        "treatment": "Prune infected wood. Protect pruning wounds. No reliable chemical control.",
        "severity": "Severe"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "treatment": "Apply copper-based fungicides. Improve canopy air circulation.",
        "severity": "Moderate"
    },
    "Grape___healthy": {"treatment": "No treatment needed.", "severity": "None"},
    "Orange___Haunglongbing_(Citrus_greening)": {
        "treatment": "No cure. Remove infected trees. Control psyllid vector with insecticides.",
        "severity": "Fatal"
    },
    "Peach___Bacterial_spot": {
        "treatment": "Apply copper sprays. Use resistant varieties. Avoid overhead irrigation.",
        "severity": "Moderate"
    },
    "Peach___healthy": {"treatment": "No treatment needed.", "severity": "None"},
    "Pepper,_bell___Bacterial_spot": {
        "treatment": "Apply copper-based bactericides. Use certified disease-free seeds.",
        "severity": "Moderate"
    },
    "Pepper,_bell___healthy": {"treatment": "No treatment needed.", "severity": "None"},
    "Potato___Early_blight": {
        "treatment": "Apply chlorothalonil or mancozeb. Remove infected tissue.",
        "severity": "Moderate"
    },
    "Potato___Late_blight": {
        "treatment": "Apply metalaxyl or cymoxanil fungicides. Destroy infected plants immediately.",
        "severity": "Severe"
    },
    "Potato___healthy": {"treatment": "No treatment needed.", "severity": "None"},
    "Raspberry___healthy": {"treatment": "No treatment needed.", "severity": "None"},
    "Soybean___healthy": {"treatment": "No treatment needed.", "severity": "None"},
    "Squash___Powdery_mildew": {
        "treatment": "Apply potassium bicarbonate or neem oil. Improve air circulation.",
        "severity": "Moderate"
    },
    "Strawberry___Leaf_scorch": {
        "treatment": "Apply fungicides (captan). Remove old infected leaves. Avoid overcrowding.",
        "severity": "Moderate"
    },
    "Strawberry___healthy": {"treatment": "No treatment needed.", "severity": "None"},
    "Tomato___Bacterial_spot": {
        "treatment": "Apply copper bactericides. Use disease-free transplants.",
        "severity": "Moderate"
    },
    "Tomato___Early_blight": {
        "treatment": "Apply fungicides (chlorothalonil). Stake plants. Mulch soil.",
        "severity": "Moderate"
    },
    "Tomato___Late_blight": {
        "treatment": "Apply fungicides (metalaxyl, chlorothalonil). Remove infected plants.",
        "severity": "Severe"
    },
    "Tomato___Leaf_Mold": {
        "treatment": "Reduce humidity. Apply fungicides. Ensure good ventilation in greenhouses.",
        "severity": "Moderate"
    },
    "Tomato___Septoria_leaf_spot": {
        "treatment": "Apply fungicides. Remove infected lower leaves. Avoid overhead watering.",
        "severity": "Moderate"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "treatment": "Apply miticides or neem oil. Increase humidity. Use predatory mites.",
        "severity": "Moderate"
    },
    "Tomato___Target_Spot": {
        "treatment": "Apply fungicides (azoxystrobin). Remove infected tissues.",
        "severity": "Moderate"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "treatment": "No cure. Control whitefly vector with insecticides. Remove infected plants.",
        "severity": "Severe"
    },
    "Tomato___Tomato_mosaic_virus": {
        "treatment": "No cure. Destroy infected plants. Sanitize tools. Use resistant varieties.",
        "severity": "Severe"
    },
    "Tomato___healthy": {"treatment": "No treatment needed.", "severity": "None"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """Returns a configured logger that prints to console (and optionally file)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", "%H:%M:%S")

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(state: dict, path: str):
    """Save model checkpoint dict to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, device="cpu"):
    """
    Load checkpoint into model (and optionally optimizer).
    Returns the checkpoint dict so callers can read epoch / best_metric etc.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Device helper
# ─────────────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
