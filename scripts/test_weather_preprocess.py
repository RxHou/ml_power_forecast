import gzip
import sys
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from preprocess_data import WEATHER_FEATURE_COLUMNS, add_weather_features, load_monthly_weather


def test_monthly_weather_is_scaled_and_joined_to_daily_rows(tmp_path: Path) -> None:
    weather_path = tmp_path / "weather.csv.gz"
    rows = [
        "NUM_POSTE;NOM_USUEL;AAAAMM;RR;NBJRR1;NBJRR5;NBJRR10;NBJBROU\n",
        "75114001;PARIS-MONTSOURIS;200612;123;10;4;1;2\n",
        "75114001;PARIS-MONTSOURIS;200701;45;7;3;0;1\n",
        "75114001;PARIS-MONTSOURIS;195001;11;1;0;0;\n",
        "92048001;MEUDON;200612;999;99;99;99;99\n",
    ]
    with gzip.open(weather_path, "wt", encoding="utf-8", newline="") as handle:
        handle.writelines(rows)

    monthly = load_monthly_weather(weather_path)

    assert list(monthly.columns) == WEATHER_FEATURE_COLUMNS
    assert monthly.loc[200612, "weather_rr_mm"] == 12.3
    assert pd.isna(monthly.loc[195001, "weather_fog_days"])
    assert monthly.loc[200612, "weather_rain_days_ge_1mm"] == 10
    assert monthly.loc[200612, "weather_rain_days_ge_5mm"] == 4
    assert monthly.loc[200612, "weather_rain_days_ge_10mm"] == 1
    assert monthly.loc[200612, "weather_fog_days"] == 2

    daily = pd.DataFrame(
        {"global_active_power": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2006-12-16", "2006-12-31", "2007-01-01"]),
    )

    with_weather = add_weather_features(daily, monthly)

    assert with_weather.loc["2006-12-16", "weather_rr_mm"] == 12.3
    assert with_weather.loc["2006-12-31", "weather_fog_days"] == 2
    assert with_weather.loc["2007-01-01", "weather_rr_mm"] == 4.5
    assert with_weather[WEATHER_FEATURE_COLUMNS].isna().sum().sum() == 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_monthly_weather_is_scaled_and_joined_to_daily_rows(Path(tmp_dir))
