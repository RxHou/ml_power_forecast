import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from power_forecast.config import CHECKPOINT_DIR, DAILY_CSV_PATH, FIGURE_DIR, METRIC_DIR
from power_forecast.data import build_window_data, load_daily_data
from power_forecast.train_utils import save_json, train_one_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a power forecasting model.")
    parser.add_argument("--model", choices=["lstm", "transformer", "cnn-transformer"], required=True)
    parser.add_argument("--horizon", type=int, choices=[90, 365], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-checkpoint", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def plot_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: list[str],
    model: str,
    horizon: int,
    seed: int,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 5), dpi=160)
    x = pd.to_datetime(dates)
    plt.plot(x, y_true, label="Ground Truth", linewidth=2.0)
    plt.plot(x, y_pred, label="Prediction", linewidth=2.0)
    plt.title(f"{model} forecast, horizon={horizon}, seed={seed}")
    plt.xlabel("Date")
    plt.ylabel("Daily global active power")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def append_metrics_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "horizon",
                "seed",
                "epochs_ran",
                "best_val_loss_scaled",
                "mse",
                "mae",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if not DAILY_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{DAILY_CSV_PATH} not found. Run scripts/preprocess_data.py first."
        )

    daily = load_daily_data(DAILY_CSV_PATH)
    window_data = build_window_data(daily, horizon=args.horizon)
    print(
        "Window shapes:",
        window_data.x_train.shape,
        window_data.x_val.shape,
        window_data.x_test.shape,
    )

    run_results = []
    for seed in args.seeds:
        checkpoint_path = None
        if not args.no_checkpoint:
            checkpoint_path = (
                CHECKPOINT_DIR / f"{args.model}_h{args.horizon}_seed{seed}.pt"
            )
        result = train_one_run(
            window_data=window_data,
            model_name=args.model,
            horizon=args.horizon,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            device_name=device,
            checkpoint_path=checkpoint_path,
        )

        metrics = result["metrics"]
        row = {
            "model": args.model,
            "horizon": args.horizon,
            "seed": seed,
            "epochs_ran": result["epochs_ran"],
            "best_val_loss_scaled": result["best_val_loss_scaled"],
            "mse": metrics["mse"],
            "mae": metrics["mae"],
        }
        append_metrics_csv(METRIC_DIR / "runs.csv", row)

        json_result = {
            "model": args.model,
            "horizon": args.horizon,
            "seed": seed,
            "epochs_ran": result["epochs_ran"],
            "best_val_loss_scaled": result["best_val_loss_scaled"],
            "metrics": metrics,
        }
        save_json(METRIC_DIR / f"{args.model}_h{args.horizon}_seed{seed}.json", json_result)

        sample_idx = -1
        figure_path = FIGURE_DIR / f"{args.model}_h{args.horizon}_seed{seed}.png"
        plot_prediction(
            result["y_true"][sample_idx],
            result["y_pred"][sample_idx],
            result["test_dates"][sample_idx],
            args.model,
            args.horizon,
            seed,
            figure_path,
        )
        print(
            f"seed={seed}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, "
            f"figure={figure_path}"
        )
        run_results.append(row)

    mse_values = np.array([r["mse"] for r in run_results], dtype=float)
    mae_values = np.array([r["mae"] for r in run_results], dtype=float)
    summary = {
        "model": args.model,
        "horizon": args.horizon,
        "seeds": args.seeds,
        "mse_mean": float(mse_values.mean()),
        "mse_std": float(mse_values.std(ddof=1)) if len(mse_values) > 1 else 0.0,
        "mae_mean": float(mae_values.mean()),
        "mae_std": float(mae_values.std(ddof=1)) if len(mae_values) > 1 else 0.0,
    }
    save_json(METRIC_DIR / f"{args.model}_h{args.horizon}_summary.json", summary)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
