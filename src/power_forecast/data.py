from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from power_forecast.config import FEATURE_COLUMNS, INPUT_DAYS, TARGET_COLUMN


@dataclass
class WindowData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_dates: list[list[str]]
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_columns: list[str]


def load_daily_data(path: Path) -> pd.DataFrame:
    daily = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return daily


def make_windows(
    daily: pd.DataFrame,
    horizon: int,
    input_days: int = INPUT_DAYS,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    features = daily[feature_columns].to_numpy(dtype=np.float32)
    target = daily[target_column].to_numpy(dtype=np.float32)
    dates = daily.index

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    y_dates: list[list[str]] = []
    max_start = len(daily) - input_days - horizon + 1
    for start in range(max_start):
        input_end = start + input_days
        output_end = input_end + horizon
        xs.append(features[start:input_end])
        ys.append(target[input_end:output_end])
        y_dates.append([d.strftime("%Y-%m-%d") for d in dates[input_end:output_end]])

    if not xs:
        raise ValueError(
            f"Not enough rows ({len(daily)}) for input_days={input_days}, horizon={horizon}."
        )

    return np.stack(xs), np.stack(ys), y_dates


def chronological_split(
    x: np.ndarray,
    y: np.ndarray,
    dates: list[list[str]],
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
) -> tuple:
    n = len(x)
    test_size = max(1, int(round(n * test_ratio)))
    train_val_size = n - test_size
    val_size = max(1, int(round(train_val_size * val_ratio)))
    train_size = train_val_size - val_size

    if train_size <= 0:
        raise ValueError("Not enough samples after chronological split.")

    train_slice = slice(0, train_size)
    val_slice = slice(train_size, train_val_size)
    test_slice = slice(train_val_size, n)

    return (
        x[train_slice],
        y[train_slice],
        x[val_slice],
        y[val_slice],
        x[test_slice],
        y[test_slice],
        dates[test_slice],
    )


def build_window_data(daily: pd.DataFrame, horizon: int) -> WindowData:
    x, y, dates = make_windows(daily, horizon=horizon)
    x_train, y_train, x_val, y_val, x_test, y_test, test_dates = chronological_split(
        x, y, dates
    )

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    n_features = x_train.shape[-1]
    feature_scaler.fit(x_train.reshape(-1, n_features))
    target_scaler.fit(y_train.reshape(-1, 1))

    def scale_x(values: np.ndarray) -> np.ndarray:
        original_shape = values.shape
        scaled = feature_scaler.transform(values.reshape(-1, n_features))
        return scaled.reshape(original_shape).astype(np.float32)

    def scale_y(values: np.ndarray) -> np.ndarray:
        original_shape = values.shape
        scaled = target_scaler.transform(values.reshape(-1, 1))
        return scaled.reshape(original_shape).astype(np.float32)

    return WindowData(
        x_train=scale_x(x_train),
        y_train=scale_y(y_train),
        x_val=scale_x(x_val),
        y_val=scale_y(y_val),
        x_test=scale_x(x_test),
        y_test=scale_y(y_test),
        test_dates=test_dates,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_columns=FEATURE_COLUMNS,
    )


def inverse_target(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    original_shape = values.shape
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(original_shape)
