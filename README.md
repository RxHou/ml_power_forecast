# Household Power Forecasting

Machine learning final project for multivariate time-series forecasting on the UCI
Individual Household Electric Power Consumption dataset.

## Task

- Input: past 90 days of daily aggregated features.
- Outputs:
  - short-term forecast: next 90 days;
  - long-term forecast: next 365 days.
- Models:
  - LSTM;
  - Transformer;
  - CNN-Transformer improved model.
- Metrics: MSE and MAE, reported as mean and standard deviation over at least five runs.

## Quick Start

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/preprocess_data.py
python scripts/train.py --model lstm --horizon 90 --epochs 5 --seeds 42
```

Use longer training and multiple seeds for the final report:

```powershell
python scripts/run_experiments.py --epochs 50 --seeds 42 43 44 45 46
```

Outputs are written to `outputs/metrics` and `outputs/figures`.

If automatic download is slow, manually download the UCI file from:

- https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip
- Mirror fallback: https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.zip
- Mirror fallback: https://github.com/KurochkinAlexey/Recurrent-neural-processes/raw/refs/heads/master/household_power_consumption.zip

Then extract `household_power_consumption.txt` into `data/raw/` and run:

```powershell
python scripts/preprocess_data.py
```
