"""
dataset.py — Data loading and augmentation pipelines.

Two modes:
  1. Supervised  — standard crops + flips for fine-tuning.
  2. SimCLR      — two stochastically augmented views per image for
                   contrastive self-supervised pre-training.

Dataset structure supported:
  - Flat layout (PlantVillage default):
      /path/to/PlantVillage/Apple___healthy/  image.jpg ...
    → automatically split 70/15/15 into a '_split' directory on first use.
  - Pre-split layout:
      data_root/train/  data_root/val/  data_root/test/
"""

import os
import shutil
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

def _pin_memory() -> bool:
    """pin_memory is only supported on CUDA, not MPS or CPU."""
    return torch.cuda.is_available()


# ─────────────────────────────────────────────────────────────────────────────
# Image size used throughout the project
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 224

# ─────────────────────────────────────────────────────────────────────────────
# Default PlantVillage dataset path
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DATA_ROOT = "/Users/aman/Documents/PlantVillage"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Supervised / evaluation transforms
# ─────────────────────────────────────────────────────────────────────────────
def get_train_transforms() -> transforms.Compose:
    """Rich augmentation pipeline for supervised fine-tuning."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(30),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms() -> transforms.Compose:
    """Deterministic transforms for validation / test."""
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SimCLR augmentation
# ─────────────────────────────────────────────────────────────────────────────
class SimCLRAugmentation:
    """
    Returns a pair (view1, view2) of differently augmented versions of the same image.
    Following the original SimCLR paper (Chen et al., 2020).
    """

    def __init__(self, size: int = IMG_SIZE, s: float = 1.0):
        color_jitter = transforms.ColorJitter(
            brightness=0.8 * s,
            contrast=0.8 * s,
            saturation=0.8 * s,
            hue=0.2 * s,
        )
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * size) | 1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Auto-split helper for flat PlantVillage layout
# ─────────────────────────────────────────────────────────────────────────────
def prepare_split_dataset(
    flat_root: str,
    split_root: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Copies images from a flat PlantVillage directory into train/val/test splits.

    Args:
        flat_root   : dir containing one sub-folder per class
        split_root  : output dir; will contain train/ val/ test/
        train_ratio : fraction for training
        val_ratio   : fraction for validation (remainder = test)
        seed        : reproducibility seed
    """
    if os.path.isdir(os.path.join(split_root, "train")):
        print(f"Split already exists at {split_root} — skipping.")
        return

    random.seed(seed)
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff", ".webp"}

    def _has_images(path):
        """True if directory directly contains at least one image file."""
        try:
            return any(
                os.path.splitext(f)[1].lower() in VALID_EXTS
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            )
        except PermissionError:
            return False

    class_dirs = sorted([
        d for d in os.listdir(flat_root)
        if os.path.isdir(os.path.join(flat_root, d))
        and _has_images(os.path.join(flat_root, d))
    ])

    print(f"Splitting {len(class_dirs)} classes from {flat_root} -> {split_root}")
    for cls in class_dirs:
        cls_path = os.path.join(flat_root, cls)
        images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)

        n_train = int(len(images) * train_ratio)
        n_val   = int(len(images) * val_ratio)
        splits  = {
            "train": images[:n_train],
            "val":   images[n_train: n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            dst_dir = os.path.join(split_root, split_name, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for fname in split_files:
                src = os.path.join(cls_path, fname)
                dst = os.path.join(dst_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    print(f"Split complete -> {split_root}/train  val  test")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Dataset wrappers
# ─────────────────────────────────────────────────────────────────────────────
class PlantVillageDataset(Dataset):
    """
    Thin wrapper around torchvision ImageFolder.
    Handles both pre-split and flat layouts.
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            split_dir = root          # flat layout fallback
        self._dataset     = datasets.ImageFolder(split_dir, transform=transform)
        self.classes      = self._dataset.classes
        self.class_to_idx = self._dataset.class_to_idx

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]


class SimCLRDataset(Dataset):
    """Produces (view1, view2, label) pairs for contrastive pre-training."""

    def __init__(self, root: str, size: int = IMG_SIZE):
        self._dataset = datasets.ImageFolder(root, transform=None)
        self._aug     = SimCLRAugmentation(size=size)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        img, label   = self._dataset[idx]
        view1, view2 = self._aug(img)
        return view1, view2, label


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(
    data_root: str = DEFAULT_DATA_ROOT,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    split_root: str = None,
):
    """
    Returns (train_loader, val_loader, test_loader, class_names).
    Auto-splits flat dataset to `split_root` (default: data_root + '_split').
    """
    train_check = os.path.join(data_root, "train")
    if not os.path.isdir(train_check):
        if split_root is None:
            split_root = data_root.rstrip("/") + "_split"
        prepare_split_dataset(data_root, split_root)
        data_root = split_root

    train_ds = PlantVillageDataset(data_root, "train", transform=get_train_transforms())
    val_ds   = PlantVillageDataset(data_root, "val",   transform=get_val_transforms())
    test_ds  = PlantVillageDataset(data_root, "test",  transform=get_val_transforms())

    def make_loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=_pin_memory(),
        )

    return (
        make_loader(train_ds, True),
        make_loader(val_ds,   False),
        make_loader(test_ds,  False),
        train_ds.classes,
    )


def build_simclr_dataloader(
    data_root: str = DEFAULT_DATA_ROOT,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """Dataloader for SimCLR pre-training. Works with flat or split layouts."""
    root = data_root if not os.path.isdir(os.path.join(data_root, "train")) \
        else os.path.join(data_root, "train")
    ds = SimCLRDataset(root)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=_pin_memory(), drop_last=True,
    )
