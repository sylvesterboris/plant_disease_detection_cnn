"""
run_train.py  ─  CLI launcher for supervised fine-tuning.

Usage:
    python run_train.py --data_root /path/to/PlantVillage
                        --pretrain_ckpt checkpoints/simclr_encoder_best.pth
                        --epochs 50

Expected data layout:
    data_root/
        train/
            Apple___Apple_scab/  ...
        val/
            Apple___Apple_scab/  ...
        test/
            Apple___Apple_scab/  ...
"""

import argparse
from src.train import train
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning of Hybrid CNN-Transformer"
    )
    parser.add_argument("--data_root",         required=True)
    parser.add_argument("--checkpoint_dir",    default="checkpoints")
    parser.add_argument("--pretrain_ckpt",     default=None,
                        help="Path to SimCLR pre-trained encoder (optional)")
    parser.add_argument("--num_classes",       type=int,   default=38)
    parser.add_argument("--epochs",            type=int,   default=50)
    parser.add_argument("--batch_size",        type=int,   default=32)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--weight_decay",      type=float, default=1e-4)
    parser.add_argument("--label_smoothing",   type=float, default=0.1)
    parser.add_argument("--freeze_cnn_epochs", type=int,   default=5,
                        help="Freeze CNN stem for first N epochs (warmup)")
    parser.add_argument("--num_workers",       type=int,   default=4)
    parser.add_argument("--seed",              type=int,   default=42)
    args = parser.parse_args()

    history = train(
        data_root        = args.data_root,
        checkpoint_dir   = args.checkpoint_dir,
        pretrain_ckpt    = args.pretrain_ckpt,
        num_classes      = args.num_classes,
        num_epochs       = args.epochs,
        batch_size       = args.batch_size,
        lr               = args.lr,
        weight_decay     = args.weight_decay,
        label_smoothing  = args.label_smoothing,
        num_workers      = args.num_workers,
        seed             = args.seed,
        device           = get_device(),
        freeze_cnn_epochs= args.freeze_cnn_epochs,
    )
    print("Fine-tuning complete!")


if __name__ == "__main__":
    main()
