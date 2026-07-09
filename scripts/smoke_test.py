import sys
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from power_forecast.data import build_window_data
from power_forecast.models import build_model


def main() -> None:
    dates = pd.date_range("2020-01-01", periods=520, freq="D")
    t = np.arange(len(dates))
    daily = pd.DataFrame(index=dates)
    daily["global_active_power"] = 1500 + 250 * np.sin(2 * np.pi * t / 365) + 80 * np.sin(2 * np.pi * t / 7)
    daily["global_reactive_power"] = 120 + 15 * np.sin(2 * np.pi * t / 30)
    daily["voltage"] = 240 + 2 * np.sin(2 * np.pi * t / 14)
    daily["global_intensity"] = daily["global_active_power"] / daily["voltage"]
    daily["sub_metering_1"] = daily["global_active_power"] * 0.12
    daily["sub_metering_2"] = daily["global_active_power"] * 0.08
    daily["sub_metering_3"] = daily["global_active_power"] * 0.18
    daily["sub_metering_remainder"] = daily["global_active_power"] * 0.62
    daily["day_of_week_sin"] = np.sin(2 * np.pi * daily.index.dayofweek / 7)
    daily["day_of_week_cos"] = np.cos(2 * np.pi * daily.index.dayofweek / 7)
    daily["day_of_year_sin"] = np.sin(2 * np.pi * daily.index.dayofyear / 365.25)
    daily["day_of_year_cos"] = np.cos(2 * np.pi * daily.index.dayofyear / 365.25)
    daily["weather_rr_mm"] = 40 + 20 * np.sin(2 * np.pi * daily.index.month / 12)
    daily["weather_rain_days_ge_1mm"] = 8 + (daily.index.month % 3)
    daily["weather_rain_days_ge_5mm"] = 3 + (daily.index.month % 2)
    daily["weather_rain_days_ge_10mm"] = daily.index.month % 2
    daily["weather_fog_days"] = daily.index.month % 4

    data = build_window_data(daily, horizon=90)
    assert data.x_train.ndim == 3
    assert data.y_train.shape[1] == 90

    model = build_model("cnn-transformer", n_features=data.x_train.shape[-1], horizon=90)
    x = torch.from_numpy(data.x_train[:2])
    y = model(x)
    assert tuple(y.shape) == (2, 90)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
