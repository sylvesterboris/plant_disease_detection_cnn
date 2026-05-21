# 🌿 PlantShield — Automated Plant Disease Detection

> **Dissertation Project** · Novel Hybrid CNN-Transformer with SimCLR Self-Supervised Pre-training & GradCAM++ Explainability  
> Dataset: [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) · 38 disease classes · 14 crop species

---

## 🚀 Novel Contributions

This project introduces a **three-pillar novelty** not previously combined in any published plant disease detection work:

| Contribution | Description | Why Novel |
|---|---|---|
| **Hybrid CNN-Transformer** | EfficientNet-B0 stem → 4 Transformer blocks with CLS token | Best of local texture (CNN) + global context (Transformer) |
| **SimCLR Pre-training** | Self-supervised contrastive learning *without labels* before fine-tuning | Data-efficient; learns representations from the full unlabelled dataset |
| **GradCAM++ Explainability** | Gradient-weighted heatmaps on every prediction | Agricultural trust — shows *which leaf regions* triggered the diagnosis |

---

## 📁 Project Structure

```
plant_disease_detection/
├── src/
│   ├── model.py        ← Hybrid CNN-Transformer architecture
│   ├── dataset.py      ← Data loading + SimCLR augmentation
│   ├── simclr.py       ← NT-Xent loss + pre-training loop
│   ├── train.py        ← Supervised fine-tuning
│   ├── evaluate.py     ← Metrics, confusion matrix, curves
│   ├── gradcam.py      ← GradCAM++ implementation
│   └── utils.py        ← Seeds, logging, checkpoints, class names
│
├── notebooks/
│   ├── 01_EDA.ipynb            ← Dataset exploration
│   ├── 02_pretraining.ipynb    ← SimCLR pre-training + t-SNE
│   └── 03_finetuning.ipynb     ← Fine-tuning + evaluation + GradCAM++
│
├── app.py              ← Streamlit web app
├── run_pretrain.py     ← CLI launcher for SimCLR pre-training
├── run_train.py        ← CLI launcher for fine-tuning
├── run_evaluate.py     ← CLI launcher for evaluation
├── requirements.txt
└── checkpoints/        ← Saved model weights
```

---

## 🏗 Architecture

```
Input (3×224×224)
      │
      ▼
┌─────────────────────────────────────┐
│  CNN Stem (EfficientNet-B0)         │  ← Local texture features
│  Output: (B, 320, 7, 7)             │
└─────────────────────────────────────┘
      │  flatten spatial → patch tokens
      ▼
┌─────────────────────────────────────┐
│  [CLS] token + 49 patch tokens      │
│  + Learnable Positional Encoding    │
│  Sequence: (B, 50, 320)             │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  4 × Transformer Encoder Blocks     │  ← Global context attention
│  (8-head self-attention + FFN)      │
└─────────────────────────────────────┘
      │  CLS token
      ▼
┌─────────────────────────────────────┐
│  Classification Head → 38 classes   │
└─────────────────────────────────────┘
```

**SimCLR Pre-training Pipeline:**

```
Unlabelled leaf image
   ├──[Aug1]──► View 1 ──► Encoder ──► Projector ──► z1 ─┐
   └──[Aug2]──► View 2 ──► Encoder ──► Projector ──► z2 ─┤
                                                          NT-Xent Loss
                                   (maximise z1·z2, minimise all others)
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
cd plant_disease_detection
pip install -r requirements.txt
```

### 2. Prepare PlantVillage dataset

Download from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) and split into train/val/test:

```
data/PlantVillage/
    train/   ← 70% of each class
    val/     ← 15% of each class
    test/    ← 15% of each class
```

A quick split script:
```python
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import shutil, os

# (Adjust paths as needed)
src = 'path/to/PlantVillage'
for cls in os.listdir(src):
    files = os.listdir(os.path.join(src, cls))
    train, temp = train_test_split(files, test_size=0.3, random_state=42)
    val, test   = train_test_split(temp,  test_size=0.5, random_state=42)
    for split, split_files in [('train', train), ('val', val), ('test', test)]:
        os.makedirs(f'data/PlantVillage/{split}/{cls}', exist_ok=True)
        for f in split_files:
            shutil.copy(os.path.join(src, cls, f), f'data/PlantVillage/{split}/{cls}/{f}')
```

---

## 🏃 Running the Project

### Step 1 — (Optional) SimCLR Self-Supervised Pre-training

```bash
python run_pretrain.py \
  --data_root data/PlantVillage \
  --epochs 100 \
  --batch_size 128
```

Saves encoder to `checkpoints/simclr_encoder_best.pth`

### Step 2 — Supervised Fine-tuning

```bash
python run_train.py \
  --data_root data/PlantVillage \
  --pretrain_ckpt checkpoints/simclr_encoder_best.pth \
  --epochs 50 \
  --batch_size 32
```

Saves best model to `checkpoints/best_model.pth`

### Step 3 — Evaluate on Test Set

```bash
python run_evaluate.py \
  --data_root data/PlantVillage \
  --checkpoint checkpoints/best_model.pth \
  --output_dir outputs/
```

### Step 4 — Launch Web App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📓 Notebooks

| Notebook | Content |
|---|---|
| `01_EDA.ipynb` | Class distribution, sample grids, pixel stats, augmentation preview |
| `02_pretraining.ipynb` | SimCLR training, loss curve, t-SNE embedding visualisation |
| `03_finetuning.ipynb` | Fine-tuning, curves, confusion matrix, GradCAM++ analysis |

```bash
cd notebooks
jupyter notebook
```

---

## 📊 Expected Results

| Metric | Target |
|---|---|
| Test Accuracy | ~97–98% |
| Macro F1 | ~97% |
| Top-5 Accuracy | ~99%+ |

---

## 🔬 Technical Details

### SimCLR NT-Xent Loss

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$$

Where τ = 0.5 (temperature), sim = cosine similarity.

### GradCAM++ Weights

$$\alpha_k^c = \frac{\frac{\partial^2 Y^c}{(\partial A_k^{ij})^2}}{2\frac{\partial^2 Y^c}{(\partial A_k^{ij})^2} + \sum_{a,b} A_k^{ab} \frac{\partial^3 Y^c}{(\partial A_k^{ij})^3}}$$

### Supported Crop Species

Apple, Blueberry, Cherry, Corn/Maize, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

---

## 📄 Citation

If you use this project in your research:

```bibtex
@misc{plantshield2026,
  title  = {Automated Plant Disease Detection using Hybrid CNN-Transformer with SimCLR},
  author = {[Your Name]},
  year   = {2026},
  note   = {Dissertation Project, PlantVillage Dataset}
}
```

---

## 📚 References

1. Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*, ICML 2020
2. Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*, ICLR 2021
3. Chattopadhay et al., *Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks*, WACV 2018
4. Tan & Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019
5. Hughes & Salathé, *An Open Access Repository of Images on Plant Health*, arXiv 2015
