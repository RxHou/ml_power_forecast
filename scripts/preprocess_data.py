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
    WEATHER_RAW_PATH,
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

WEATHER_STATION_ID = "75114001"
WEATHER_STATION_NAME = "PARIS-MONTSOURIS"
WEATHER_SOURCE_COLUMNS = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]
WEATHER_FEATURE_COLUMNS = [
    "weather_rr_mm",
    "weather_rain_days_ge_1mm",
    "weather_rain_days_ge_5mm",
    "weather_rain_days_ge_10mm",
    "weather_fog_days",
]


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    day_of_week = daily.index.dayofweek.to_numpy()
    day_of_year = daily.index.dayofyear.to_numpy()

    daily["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    daily["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    daily["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    daily["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return daily


def load_monthly_weather(path: Path = WEATHER_RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_data.py first."
        )

    weather = pd.read_csv(path, sep=";", compression="gzip", dtype=str)
    station = weather[weather["NUM_POSTE"] == WEATHER_STATION_ID].copy()
    if station.empty:
        raise ValueError(
            f"Weather station {WEATHER_STATION_ID} ({WEATHER_STATION_NAME}) "
            f"not found in {path}."
        )

    station["AAAAMM"] = pd.to_numeric(station["AAAAMM"], errors="coerce").astype("Int64")
    station = station.dropna(subset=["AAAAMM"]).set_index("AAAAMM").sort_index()

    for column in WEATHER_SOURCE_COLUMNS:
        station[column] = pd.to_numeric(station[column], errors="coerce")

    monthly = pd.DataFrame(index=station.index.astype(int))
    monthly["weather_rr_mm"] = station["RR"] / 10.0
    monthly["weather_rain_days_ge_1mm"] = station["NBJRR1"]
    monthly["weather_rain_days_ge_5mm"] = station["NBJRR5"]
    monthly["weather_rain_days_ge_10mm"] = station["NBJRR10"]
    monthly["weather_fog_days"] = station["NBJBROU"]

    return monthly[WEATHER_FEATURE_COLUMNS]


def add_weather_features(daily: pd.DataFrame, monthly_weather: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    month_keys = daily.index.year * 100 + daily.index.month
    weather_for_days = monthly_weather.reindex(month_keys).set_index(daily.index)

    if weather_for_days[WEATHER_FEATURE_COLUMNS].isna().any().any():
        missing_months = sorted(set(month_keys[weather_for_days.isna().any(axis=1)]))
        raise ValueError(f"Missing monthly weather records for: {missing_months}")

    return pd.concat([daily, weather_for_days[WEATHER_FEATURE_COLUMNS]], axis=1)


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
    monthly_weather = load_monthly_weather()
    weather_overlap = monthly_weather.loc[
        (monthly_weather.index >= daily.index.min().year * 100 + daily.index.min().month)
        & (monthly_weather.index <= daily.index.max().year * 100 + daily.index.max().month)
    ]
    daily = add_weather_features(daily, monthly_weather)

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
        "weather_dataset": "MENS_departement_75_periode_1950-2024",
        "weather_url": "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/MENS/MENSQ_75_previous-1950-2024.csv.gz",
        "weather_station": {
            "num_poste": WEATHER_STATION_ID,
            "nom_usuel": WEATHER_STATION_NAME,
            "reason": (
                "Paris-Montsouris is close to Sceaux and has complete monthly "
                "RR/NBJRR1/NBJRR5/NBJRR10/NBJBROU values for 2006-12 to 2010-11."
            ),
        },
        "weather_monthly_rows_for_power_period": int(len(weather_overlap)),
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
