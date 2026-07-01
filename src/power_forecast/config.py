from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

UCI_ZIP_URLS = [
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/"
    "household_power_consumption.zip",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "household_power_consumption.zip",
    "https://github.com/KurochkinAlexey/Recurrent-neural-processes/raw/"
    "refs/heads/master/household_power_consumption.zip",
]
UCI_ZIP_URL = UCI_ZIP_URLS[0]
RAW_ZIP_PATH = RAW_DIR / "individual_household_power_consumption.zip"
RAW_TXT_PATH = RAW_DIR / "household_power_consumption.txt"
DAILY_CSV_PATH = PROCESSED_DIR / "daily_power.csv"
SUMMARY_JSON_PATH = PROCESSED_DIR / "preprocess_summary.json"

INPUT_DAYS = 90
TARGET_COLUMN = "global_active_power"

SUM_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
]

MEAN_COLUMNS = [
    "voltage",
    "global_intensity",
]

FEATURE_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]
