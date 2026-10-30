"""
modeling.py - shared training and evaluation helpers for Week 4 onward (baseline, model comparison,
advanced models, app).

Every model is a scikit-learn Pipeline: the Week 3 preprocessor (pp.build_preprocessor, fit on the
training window only) followed by a regressor trained on log(ClosePrice). Predictions are turned
back into dollars with exp(), and all metrics are computed in dollars.

Metric set follows the AVM best-practices doc (section 08): MdAPE is the headline figure, reported
with R2, MAPE, MAE and RMSE, plus the share of homes within 5/10/20% of the sale price and the
median signed error (bias). Error is also broken out by price band and by county.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import preprocessing as pp

STR_COLS = {"zip5": str, "City": str, "HighSchoolDistrict": str, "CountyOrParish": str,
            "levels_cat": str, "ListingKey": str}


def features_of(pipe: Pipeline) -> dict:
    """Feature groups a pipeline was built with (read back from its ColumnTransformer)."""
    return {name: list(cols) for name, _, cols in pipe.named_steps["prep"].transformers}


# ------------------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------------------
def load_split(path: Path) -> pd.DataFrame:
    """Read processed/train.csv or processed/test.csv with the dtypes the preprocessor expects."""
    d = pd.read_csv(path, dtype=STR_COLS, low_memory=False)
    d["close_month"] = pd.PeriodIndex(d["close_month"], freq="M")
    return d


def load_clean(path: Path) -> pd.DataFrame:
    """Read processed/sold_sfr_clean.csv (all cleaned months) for rolling-origin backtests."""
    d = pd.read_csv(path, dtype=STR_COLS, low_memory=False, parse_dates=["CloseDate"])
    d["close_month"] = pd.PeriodIndex(d["close_month"], freq="M")
    return d


# ------------------------------------------------------------------------------------------------
# Pipelines
# ------------------------------------------------------------------------------------------------
def make_pipeline(model, min_frequency: int = 30, features: dict | None = None) -> Pipeline:
    """Week 3 preprocessor + regressor. Fit on train only; the target is log price.
    `features` (Week 6+): feature groups from feature_engineering.feature_set(); default = Week 3 inputs."""
    pp.assert_no_leakage(sum(features.values(), []) if features else pp.FEATURES)
    return Pipeline([("prep", pp.build_preprocessor(min_frequency, features)), ("model", model)])


def fit(pipe: Pipeline, train: pd.DataFrame) -> Pipeline:
    return clone(pipe).fit(pp.model_frame(train, features_of(pipe)), train[pp.TARGET])


def predict_price(fitted: Pipeline, d: pd.DataFrame) -> np.ndarray:
    """Predicted close price in dollars (exp of the predicted log price, i.e. a median estimate)."""
    return np.exp(fitted.predict(pp.model_frame(d, features_of(fitted))))


# ------------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------------
def full_metrics(y_true, y_pred) -> dict:
    """Dollar-scale metrics. Percentages are in percent (7.4 means 7.4%)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = (y_pred - y_true) / y_true
    ape = np.abs(err)
    return {
        "n": len(y_true),
        "MdAPE": np.median(ape) * 100,
        "R2": r2_score(y_true, y_pred),
        "R2_log": r2_score(np.log(y_true), np.log(y_pred)),
        "MAPE": ape.mean() * 100,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MedAE": float(np.median(np.abs(y_pred - y_true))),
        "within_5pct": (ape <= 0.05).mean() * 100,
        "within_10pct": (ape <= 0.10).mean() * 100,
        "within_20pct": (ape <= 0.20).mean() * 100,
        "median_bias_pct": np.median(err) * 100,
    }


def price_bands(y_true, q: int = 5) -> pd.Series:
    """Quantile bands of the ACTUAL close price, labelled with their dollar ranges."""
    s = pd.Series(np.asarray(y_true, float))
    cats, edges = pd.qcut(s, q, retbins=True)
    labels = [f"Q{i+1}: ${edges[i]/1e3:,.0f}K-${edges[i+1]/1e3:,.0f}K" for i in range(q)]
    return pd.Series(pd.Categorical.from_codes(cats.cat.codes, labels, ordered=True), index=s.index)


def error_by_group(y_true, y_pred, groups, min_n: int = 0) -> pd.DataFrame:
    """MdAPE, MAPE, MAE, share within 10% and median bias for each group with at least min_n homes."""
    d = pd.DataFrame({"y": np.asarray(y_true, float), "p": np.asarray(y_pred, float),
                      "g": np.asarray(groups)})
    d["err"] = (d.p - d.y) / d.y
    d["ape"] = d.err.abs()
    out = d.groupby("g", observed=True).agg(
        n=("y", "size"), median_price=("y", "median"), MdAPE=("ape", "median"), MAPE=("ape", "mean"),
        MAE=("ape", lambda s: np.abs(d.loc[s.index, "p"] - d.loc[s.index, "y"]).mean()),
        within_10pct=("ape", lambda s: (s <= 0.10).mean()), median_bias=("err", "median"))
    for c in ["MdAPE", "MAPE", "within_10pct", "median_bias"]:
        out[c] *= 100
    return out[out.n >= min_n]


# ------------------------------------------------------------------------------------------------
# Rolling-origin backtest (best-practices section 01)
# ------------------------------------------------------------------------------------------------
def rolling_backtest(clean: pd.DataFrame, pipe: Pipeline, test_months, x_months: int):
    """Re-run the full split -> train-only thresholds -> fit -> score procedure for each test month.

    Returns (metrics table, dict of test-month predictions). Each cutoff learns its own outlier
    thresholds and its own preprocessor from its own training window, exactly as in production.
    """
    rows, preds = [], {}
    for m in test_months:
        tr, te = pp.window_split(clean, m, x_months)
        th = pp.fit_thresholds(tr)
        tr, te = tr[pp.within_thresholds(tr, th)], te[pp.within_thresholds(te, th)]
        fitted = fit(pipe, tr)
        p = predict_price(fitted, te)
        preds[str(m)] = pd.DataFrame({"ListingKey": te.ListingKey.values, "ClosePrice": te.ClosePrice.values,
                                      "pred": p})
        rows.append({"test_month": str(m), "train_months": f"{tr.close_month.min()} to {tr.close_month.max()}",
                     "n_train": len(tr), **full_metrics(te.ClosePrice, p)})
    return pd.DataFrame(rows), preds
