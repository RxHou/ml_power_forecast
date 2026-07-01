import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from power_forecast.config import METRIC_DIR


def main() -> None:
    runs_path = METRIC_DIR / "runs.csv"
    if not runs_path.exists():
        raise FileNotFoundError(f"{runs_path} not found. Run experiments first.")

    runs = pd.read_csv(runs_path)
    summary = (
        runs.groupby(["horizon", "model"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            runs=("seed", "count"),
        )
        .sort_values(["horizon", "mse_mean"])
    )
    output_path = METRIC_DIR / "summary.csv"
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
