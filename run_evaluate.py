"""
run_evaluate.py  ─  CLI launcher for full test-set evaluation.

Usage:
    python run_evaluate.py --data_root /path/to/PlantVillage
                           --checkpoint checkpoints/best_model.pth
                           --output_dir outputs/
"""

import argparse
from src.evaluate import run_full_evaluation
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate plant disease model on test set"
    )
    parser.add_argument("--data_root",    required=True)
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--output_dir",   default="outputs")
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--num_classes",  type=int, default=38)
    args = parser.parse_args()

    metrics = run_full_evaluation(
        checkpoint_path = args.checkpoint,
        data_root       = args.data_root,
        output_dir      = args.output_dir,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        num_classes     = args.num_classes,
        device          = get_device(),
    )

    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
