"""
build_processed.py - rebuild the Week 3 outputs in processed/ without opening the notebook.

Runs the same steps as notebooks/02_preprocessing.ipynb sections 2-10 with a fixed training window
(default: the X chosen in the notebook, read from processed/preprocessing_config.json, else 15).
It does NOT re-run the X experiment. Usage (from the project folder):
    python3 py/build_processed.py            # use the saved X
    python3 py/build_processed.py --x 12     # override X
    python3 py/build_processed.py --test-month 2026-05   # roll forward when a new month arrives
"""
import argparse, json, platform, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
import preprocessing as pp

OUT_COLS = ["ListingKey", "UnparsedAddress", "City", "zip5", "CountyOrParish", "Latitude", "Longitude",
            "HighSchoolDistrict", "CloseDate", "close_month", "ClosePrice", "log_price",
            "LivingArea", "BedroomsTotal", "BathroomsTotalInteger", "LotSizeSquareFeet", "YearBuilt",
            "GarageSpaces", "ParkingTotal", "hoa_monthly", "PoolPrivateYN", "ViewYN", "FireplaceYN",
            "AttachedGarageYN", "levels_cat", "log_living_area", "log_lot_sqft", "age", "month_sin", "month_cos",
            "comp_log_ppsf", "comp_log_price", "comp_log_dist_km", "comp_pool_months", "split"]


def main():
    out_dir = ROOT / "processed"; out_dir.mkdir(exist_ok=True)
    cfg_path = out_dir / "preprocessing_config.json"
    saved = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=int, default=saved.get("train_window_months", 15))
    ap.add_argument("--test-month", default=saved.get("test_month", "2026-04"))
    a = ap.parse_args()
    pp.assert_no_leakage()

    raw, inventory, _ = pp.load_sold(ROOT / "csv")
    clean, removal_log, _ = pp.clean_rows(raw)
    clean = pp.add_comp_features(pp.add_row_features(clean), k=10, lookback_months=12)

    train, test = pp.window_split(clean, a.test_month, a.x)
    th = pp.fit_thresholds(train)
    train, test = train[pp.within_thresholds(train, th)], test[pp.within_thresholds(test, th)]
    prep = pp.build_preprocessor().fit(pp.model_frame(train), train[pp.TARGET])

    clean["split"] = "outside_window"
    clean.loc[train.index, "split"] = "train"; clean.loc[test.index, "split"] = "test"
    in_window = clean.close_month.between(pd.Period(a.test_month) - a.x, pd.Period(a.test_month))
    clean.loc[in_window & (clean.split == "outside_window"), "split"] = "removed_by_train_thresholds"
    assert not set(OUT_COLS) & set(pp.LEAKAGE_FIELDS)

    clean[OUT_COLS].to_csv(out_dir / "sold_sfr_clean.csv", index=False)
    model_cols = ["ListingKey", "close_month"] + pp.FEATURES + ["ClosePrice", pp.TARGET]
    train[model_cols].to_csv(out_dir / "train.csv", index=False)
    test[model_cols].to_csv(out_dir / "test.csv", index=False)
    joblib.dump(prep, out_dir / "preprocessor.joblib")
    cfg = {
        "test_month": a.test_month, "train_window_months": a.x,
        "train_months": [str(train.close_month.min()), str(train.close_month.max())],
        "thresholds_learned_on_train": th, "target": "log(ClosePrice)",
        "features": {"numeric": pp.NUM_FEATURES, "binary": pp.BIN_FEATURES,
                     "low_cardinality": pp.LOWCARD_FEATURES, "high_cardinality": pp.HIGHCARD_FEATURES},
        "excluded_fields": pp.LEAKAGE_FIELDS,
        "comp_features": {"k": 10, "lookback_months": 12, "rule": "strictly earlier months only"},
        "rows": {"single_family_loaded": len(raw), "after_row_cleaning": len(clean), "train": len(train), "test": len(test)},
        "seed": pp.SEED, "data_snapshot": inventory.file.tolist(),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__},
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"X = {a.x}, test = {a.test_month}: train {len(train):,} rows, test {len(test):,} rows -> {out_dir}")


if __name__ == "__main__":
    main()
