import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all final experiments.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = ["lstm", "transformer", "cnn-transformer"]
    horizons = [90, 365]

    for horizon in horizons:
        for model in models:
            command = [
                sys.executable,
                "scripts/train.py",
                "--model",
                model,
                "--horizon",
                str(horizon),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--patience",
                str(args.patience),
                "--device",
                args.device,
                "--seeds",
                *[str(seed) for seed in args.seeds],
            ]
            print("Running:", " ".join(command))
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
