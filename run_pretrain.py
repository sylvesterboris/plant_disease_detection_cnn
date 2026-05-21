"""
run_pretrain.py  ─  Quick launcher for SimCLR self-supervised pre-training.

Usage:
    python run_pretrain.py --data_root /path/to/PlantVillage
                           --epochs 100
                           --batch_size 128
                           --checkpoint_dir checkpoints

The PlantVillage directory should contain class sub-directories, e.g.:
    PlantVillage/
        Apple___Apple_scab/
        Apple___Black_rot/
        ...
"""

import argparse
from src.simclr import pretrain
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser(
        description="SimCLR self-supervised pre-training on PlantVillage"
    )
    parser.add_argument("--data_root",       required=True, help="Path to PlantVillage dataset")
    parser.add_argument("--checkpoint_dir",  default="checkpoints")
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--batch_size",      type=int,   default=128)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--temperature",     type=float, default=0.5)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    ckpt_path = pretrain(
        data_root      = args.data_root,
        checkpoint_dir = args.checkpoint_dir,
        num_epochs     = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        temperature    = args.temperature,
        num_workers    = args.num_workers,
        seed           = args.seed,
        device         = get_device(),
    )
    print(f"\nPre-training complete. Encoder saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
