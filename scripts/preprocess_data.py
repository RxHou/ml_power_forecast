import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from power_forecast.config import (
    DAILY_CSV_PATH,
    FEATURE_COLUMNS,
    MEAN_COLUMNS,
    RAW_TXT_PATH,
    SUMMARY_JSON_PATH,
    SUM_COLUMNS,
)


RAW_NUMERIC_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    day_of_week = daily.index.dayofweek.to_numpy()
    day_of_year = daily.index.dayofyear.to_numpy()

    daily["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    daily["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    daily["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    daily["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return daily


def main() -> None:
    if not RAW_TXT_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_TXT_PATH} not found. Run scripts/download_data.py first."
        )

    print(f"Reading raw data: {RAW_TXT_PATH}")
    df = pd.read_csv(
        RAW_TXT_PATH,
        sep=";",
        na_values="?",
        low_memory=False,
    )
    df.columns = [column.lower() for column in df.columns]
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.drop(columns=["date", "time"])
    df = df.set_index("datetime").sort_index()

    for column in RAW_NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    missing_before = df[RAW_NUMERIC_COLUMNS].isna().sum().to_dict()

    df["sub_metering_remainder"] = (
        df["global_active_power"] * 1000 / 60
        - df[["sub_metering_1", "sub_metering_2", "sub_metering_3"]].sum(axis=1)
    )

    daily_sum = df[SUM_COLUMNS].resample("D").sum(min_count=1)
    daily_mean = df[MEAN_COLUMNS].resample("D").mean()
    daily = pd.concat([daily_sum, daily_mean], axis=1).sort_index()

    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index)
    daily.index.name = "date"

    missing_daily_before = daily.isna().sum().to_dict()
    daily = daily.interpolate(method="time").ffill().bfill()
    daily = add_calendar_features(daily)

    daily = daily[FEATURE_COLUMNS]
    DAILY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY_CSV_PATH, index=True)

    summary = {
        "raw_rows": int(len(df)),
        "raw_start": str(df.index.min()),
        "raw_end": str(df.index.max()),
        "daily_rows": int(len(daily)),
        "daily_start": str(daily.index.min().date()),
        "daily_end": str(daily.index.max().date()),
        "raw_missing_before": {k: int(v) for k, v in missing_before.items()},
        "daily_missing_before_interpolation": {
            k: int(v) for k, v in missing_daily_before.items()
        },
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "global_active_power",
        "split_policy": (
            "Chronological sliding-window samples. For each horizon, the final "
            "20 percent of windows are used for testing and the earlier windows "
            "are used for training/validation."
        ),
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved daily data to {DAILY_CSV_PATH}")
    print(f"Saved preprocessing summary to {SUMMARY_JSON_PATH}")
    print(f"Daily shape: {daily.shape}")


if __name__ == "__main__":
    main()
