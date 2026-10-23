# IDX Exchange Data Science

Predicting the close price of single-family homes in California from CRMLS sold records (IDX Exchange Data Science Internship).

## Progress

| Week | Notebook | Summary |
| --- | --- | --- |
| 1 | `docs/week1_dataset_access_and_key_columns.docx` | Dataset access confirmed; key columns and their planned use |
| 2 | `notebooks/01_exploration.ipynb` | Data exploration |
| 3 | `notebooks/02_preprocessing.ipynb` | Cleaning, feature building, train/test split (train Jan 2025 – Mar 2026, test Apr 2026) |
| 4 | `notebooks/03_baseline_model.ipynb` | Linear Regression baseline, evaluated on the April 2026 test month |
| 5 | `notebooks/04_model_comparison.ipynb` | Decision Tree and Random Forest compared with the baseline |

## Current results (test month: April 2026)

| Model | MdAPE | R² | Within 10% of sale price |
| --- | --- | --- | --- |
| Reference: comp median price (no model) | 12.0% | 0.707 | 43.8% |
| Reference: comp price/sq ft × living area (no model) | 12.7% | 0.861 | 40.5% |
| Baseline: Linear Regression | 8.6% | 0.889 | 56.2% |
| Decision Tree | 9.0% | 0.886 | 54.0% |
| **Random Forest (best so far)** | **7.6%** | **0.906** | **60.9%** |

MdAPE (median absolute percentage error) is the headline metric. The full metric set is in `results/model_results.csv`. The Random Forest also has the lowest MdAPE at all three earlier validation cutoffs (Jan–Mar 2026), so its lead is not specific to April.

## Layout

| Path | Contents |
| --- | --- |
| `notebooks/` | One notebook per week (see Progress above) |
| `py/preprocessing.py` | The cleaning and feature code (used by every notebook from Week 3 on) |
| `py/build_processed.py` | Rebuilds the files in `processed/` without opening the notebook |
| `py/modeling.py` | Shared training and evaluation helpers (Week 4 on) |
| `processed/train.csv` | Training data: Jan 2025 – Mar 2026 (155,216 rows) |
| `processed/test.csv` | Test data: Apr 2026 (12,175 rows) |
| `processed/preprocessing_config.json` | Chosen training window, outlier cutoffs, feature list |
| `processed/train_window_experiment.csv` | Results of the training-window experiment |
| `models/baseline_linear_regression.joblib` | Fitted baseline pipeline (preprocessor + Linear Regression) |
| `models/decision_tree.joblib` | Fitted Decision Tree pipeline. The Random Forest (~190 MB) is not saved; rerun the Week 5 notebook to rebuild it |
| `results/model_results.csv` | Leaderboard: every model's test metrics |
| `results/baseline_*.csv`, `results/baseline_run_info.json` | Baseline backtest, error by price band and county, test predictions, run details |
| `results/week5_*` | Week 5 tuning results, backtest MdAPE, error by price band, test predictions, chosen hyperparameters |
| `figures/` | Chart images produced by the notebooks, one folder per notebook |
| `docs/` | Week 1 notes on dataset access and key columns |

## Setup

```
pip install -r requirements.txt
```

The raw monthly CRMLS files are not in this repository. Put the `CRMLSSold*.csv` files in a `csv/` folder at the top level and run:

```
python3 py/build_processed.py
```

This rebuilds `processed/`, including two files that are too large or regenerable to commit: `sold_sfr_clean.csv` and `preprocessor.joblib`. The Week 4 notebook needs both and the Week 5 notebook needs `sold_sfr_clean.csv`, so run this step before opening them.
