# Final Submission Checklist

- [ ] Download or place `household_power_consumption.txt` in `data/raw`.
- [ ] Run `python scripts/preprocess_data.py`.
- [ ] Run final experiments with 5 seeds:
  `python scripts/run_experiments.py --epochs 50 --seeds 42 43 44 45 46`.
- [ ] Run `python scripts/summarize_results.py`.
- [ ] Copy metrics from `outputs/metrics/summary.csv` into the report.
- [ ] Insert prediction-vs-ground-truth figures from `outputs/figures` into the PDF.
- [ ] Fill author contribution and research field.
- [ ] Upload code to GitHub.
- [ ] Add GitHub link to the PDF report.
- [ ] Submit before 2026-07-15 12:00.
