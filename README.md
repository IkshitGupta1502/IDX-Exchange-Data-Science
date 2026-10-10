# IDX Exchange Data Science

Predicting the close price of single-family homes in California from CRMLS sold records (IDX Exchange Data Science Internship).

## Layout

| Path | Contents |
| --- | --- |
| `notebooks/01_exploration.ipynb` | Week 2: data exploration |
| `notebooks/02_preprocessing.ipynb` | Week 3: cleaning, feature building, train/test split |
| `py/preprocessing.py` | The cleaning code (required by the preprocessing notebook) |
| `py/build_processed.py` | Rebuilds the files in `processed/` without opening the notebook |
| `processed/train.csv` | Training data: Jan 2025 – Mar 2026 (155,216 rows) |
| `processed/test.csv` | Test data: Apr 2026 (12,175 rows) |
| `processed/preprocessing_config.json` | Chosen training window, outlier cutoffs, feature list |
| `processed/train_window_experiment.csv` | Results of the training-window experiment |
| `figures/` | Chart images produced by the notebooks |
| `docs/` | Week 1 notes on dataset access and key columns |

## Setup

```
pip install -r requirements.txt
```

The raw monthly CRMLS files are not in this repository. To rebuild `processed/` from scratch, put the `CRMLSSold*.csv` files in a `csv/` folder at the top level and run:

```
python3 py/build_processed.py
```
