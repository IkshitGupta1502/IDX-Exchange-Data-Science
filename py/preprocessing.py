"""
preprocessing.py - Week 3 data preprocessing for the California single-family close-price model.

Goal of the model: predict the close price, at the time of prediction, of ANY single-family
address in California, whether it is on the market or not.

Feature rule (team policy): a field may be a model input only if it is available for any address
at the time of prediction. Fields that only exist because a home was listed or sold are excluded
(see LEAKAGE_FIELDS). ListingContractDate is used for one data-cleaning check and then dropped.

Everything that learns from data (outlier thresholds, imputers, encoders, scalers) is fit on the
training window only. Comp-based neighborhood features use only sales that closed in months
strictly before the row's own close month, so they are leakage-free for any split.

Used by: notebooks/02_preprocessing.ipynb and all later weeks (baseline, model comparison, app).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

SEED = 42
EARTH_KM = 6371.0

CA_COUNTIES = {
    "Alameda", "Alpine", "Amador", "Butte", "Calaveras", "Colusa", "Contra Costa", "Del Norte",
    "El Dorado", "Fresno", "Glenn", "Humboldt", "Imperial", "Inyo", "Kern", "Kings", "Lake", "Lassen",
    "Los Angeles", "Madera", "Marin", "Mariposa", "Mendocino", "Merced", "Modoc", "Mono", "Monterey",
    "Napa", "Nevada", "Orange", "Placer", "Plumas", "Riverside", "Sacramento", "San Benito",
    "San Bernardino", "San Diego", "San Francisco", "San Joaquin", "San Luis Obispo", "San Mateo",
    "Santa Barbara", "Santa Clara", "Santa Cruz", "Shasta", "Sierra", "Siskiyou", "Solano", "Sonoma",
    "Stanislaus", "Sutter", "Tehama", "Trinity", "Tulare", "Tuolumne", "Ventura", "Yolo", "Yuba",
}
CA_LAT, CA_LON = (32.0, 42.1), (-124.5, -114.0)

# Fields that are never model inputs, with the reason. Checked by assert_no_leakage().
LEAKAGE_FIELDS = {
    "ListPrice": "Only exists for listed homes; near-1.0 correlation with ClosePrice",
    "OriginalListPrice": "Only exists for listed homes; near-1.0 correlation with ClosePrice",
    "DaysOnMarket": "Only exists once a sale process has happened",
    "CumulativeDaysOnMarket": "Only exists once a sale process has happened",
    "PurchaseContractDate": "Only exists once a sale process has happened",
    "ListingContractDate": "Only exists for listed homes; with CloseDate it rebuilds Days on Market "
                           "(used for one cleaning check only, then dropped)",
    "ContractStatusChangeDate": "Sale-process event date",
    "MlsStatus": "Sale-process status (always Closed here)",
    "BuyerAgencyCompensation": "Sale-process term",
    "NewConstructionYN": "Listing-status flag; age (from YearBuilt) carries the same information for any address",
}
# Patterns that would indicate a ClosePrice- or ListPrice-derived feature slipped in
_LEAK_PATTERNS = re.compile(r"listprice|list_price|daysonmarket|dom\b|contractdate|sale_to_list|ratio", re.I)

# Columns read from the raw files. Agent/office fields are never loaded.
LOAD_COLS = [
    "ListingKey", "UnparsedAddress", "City", "PostalCode", "CountyOrParish", "Latitude", "Longitude",
    "HighSchoolDistrict", "PropertyType", "PropertySubType",
    "ClosePrice", "CloseDate", "ListingContractDate",
    "LivingArea", "BedroomsTotal", "BathroomsTotalInteger", "LotSizeSquareFeet", "YearBuilt",
    "Stories", "Levels", "GarageSpaces", "ParkingTotal",
    "PoolPrivateYN", "ViewYN", "FireplaceYN", "AttachedGarageYN",
    "AssociationFee", "AssociationFeeFrequency",
]

# ---- Model feature lists (all available for any address at prediction time) ----
NUM_FEATURES = [
    "log_living_area", "BedroomsTotal", "BathroomsTotalInteger", "log_lot_sqft", "age",
    "GarageSpaces", "ParkingTotal", "hoa_monthly", "Latitude", "Longitude",
    "comp_log_ppsf", "comp_log_price", "comp_log_dist_km", "month_sin", "month_cos",
]
BIN_FEATURES = ["PoolPrivateYN", "ViewYN", "FireplaceYN", "AttachedGarageYN"]
LOWCARD_FEATURES = ["CountyOrParish", "levels_cat"]
HIGHCARD_FEATURES = ["zip5", "City", "HighSchoolDistrict"]
FEATURES = NUM_FEATURES + BIN_FEATURES + LOWCARD_FEATURES + HIGHCARD_FEATURES
TARGET = "log_price"


def assert_no_leakage(features=FEATURES):
    """Structural guard: fail loudly if a banned or list-price/DOM-derived field is a model input."""
    bad = [f for f in features if f in LEAKAGE_FIELDS or _LEAK_PATTERNS.search(f) or f == "ClosePrice"]
    assert not bad, f"Leakage: banned fields in feature list: {bad}"
    return True


# ------------------------------------------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------------------------------------------
def load_sold(data_dir: Path):
    """Read every CRMLSSold*.csv, keep Residential / SingleFamilyResidence rows and LOAD_COLS.

    Returns (df, inventory, missingness) where missingness is % missing per raw column on the
    single-family subset (a column absent from a file counts as missing for that file's rows).
    """
    files = sorted(Path(data_dir).glob("CRMLSSold*.csv"))
    frames, inv, nulls, n_total = [], [], {}, 0
    for f in files:
        month = re.search(r"(\d{6})", f.name).group(1)
        raw = pd.read_csv(f, low_memory=False, dtype={"PostalCode": str})
        sfr = raw[(raw["PropertyType"] == "Residential") & (raw["PropertySubType"] == "SingleFamilyResidence")]
        inv.append({"month": month, "file": f.name, "rows_raw": len(raw), "rows_sfr": len(sfr)})
        for c in raw.columns:  # null count, and how many SFR rows had this column at all
            nulls[c] = nulls.get(c, 0) + int(sfr[c].isna().sum())
            nulls["__n__" + c] = nulls.get("__n__" + c, 0) + len(sfr)
        n_total += len(sfr)
        frames.append(sfr[[c for c in LOAD_COLS if c in sfr.columns]].assign(source_month=month))
        del raw
    df = pd.concat(frames, ignore_index=True)
    cols = [c for c in nulls if not c.startswith("__")]
    miss = pd.Series({c: (nulls[c] + n_total - nulls["__n__" + c]) / n_total * 100 for c in cols})
    return df, pd.DataFrame(inv), miss.sort_values()


# ------------------------------------------------------------------------------------------------
# 2. Row-level cleaning (deterministic rules, safe to apply to all months before splitting)
# ------------------------------------------------------------------------------------------------
def _to_bool(s):
    return s.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0})


def clean_rows(df: pd.DataFrame):
    """Remove records that cannot be true. Returns (clean_df, removal_log, value_fix_log).

    These rules use fixed, domain-based limits (not percentiles of the data), so they do not learn
    anything from the test month. Price-based outlier filters are learned later, on train only.
    """
    d = df.copy()
    d["CloseDate"] = pd.to_datetime(d["CloseDate"], errors="coerce")
    d["ListingContractDate"] = pd.to_datetime(d["ListingContractDate"], errors="coerce")
    d["PostalCode"] = d["PostalCode"].fillna("").astype(str).str[:5]
    log = [{"step": "Start: single-family residential rows", "removed": 0, "remaining": len(d)}]

    def drop(mask, name):
        nonlocal d
        mask = mask.fillna(False)
        log.append({"step": name, "removed": int(mask.sum()), "remaining": int((~mask).sum())})
        d = d.loc[~mask].copy()

    drop(d["ClosePrice"].isna() | (d["ClosePrice"] <= 0), "ClosePrice missing or <= 0")
    d = d.sort_values(["CloseDate", "source_month"])
    drop(d.duplicated("ListingKey", keep="last"), "Duplicate ListingKey (kept latest record)")
    drop(d.duplicated(["UnparsedAddress", "PostalCode", "CloseDate"], keep="last"),
         "Duplicate address + ZIP + CloseDate")
    drop(d["CloseDate"].isna() | (d["ListingContractDate"].notna() & (d["CloseDate"] < d["ListingContractDate"])),
         "CloseDate missing or before ListingContractDate")
    outside = (d["Latitude"].isna() | d["Longitude"].isna()
               | ~d["Latitude"].between(*CA_LAT) | ~d["Longitude"].between(*CA_LON)
               | ~d["CountyOrParish"].isin(CA_COUNTIES)
               | ~d["PostalCode"].str.match(r"^9[0-6]\d{3}$"))
    drop(outside, "Missing coordinates or outside California (coords, county, ZIP)")
    drop(d["LivingArea"].isna() | ~d["LivingArea"].between(300, 20000),
         "LivingArea missing or outside 300-20,000 sq ft")
    drop((d["BedroomsTotal"] > 20) | (d["BathroomsTotalInteger"] > 20),
         "More than 20 bedrooms or bathrooms")
    drop((d["BedroomsTotal"] > 0) & (d["LivingArea"] / d["BedroomsTotal"] < 70),
         "Bedrooms inconsistent with living area (< 70 sq ft per bedroom)")

    # ---- value-level fixes: implausible values become missing and are imputed later ----
    fixes = []

    def to_nan(col, mask, name):
        mask = mask.fillna(False)
        fixes.append({"column": col, "rule": name, "values_set_to_missing": int(mask.sum())})
        d.loc[mask, col] = np.nan

    close_year = d["CloseDate"].dt.year
    to_nan("BedroomsTotal", d["BedroomsTotal"] == 0, "0 bedrooms")
    to_nan("BathroomsTotalInteger", d["BathroomsTotalInteger"] == 0, "0 bathrooms")
    to_nan("YearBuilt", (d["YearBuilt"] < 1850) | (d["YearBuilt"] > close_year + 1), "before 1850 or after close year + 1")
    to_nan("LotSizeSquareFeet", ~d["LotSizeSquareFeet"].between(100, 43_560_000) & d["LotSizeSquareFeet"].notna(),
           "lot size < 100 sq ft or > 1,000 acres")
    to_nan("GarageSpaces", d["GarageSpaces"] > 20, "more than 20 garage spaces")
    to_nan("ParkingTotal", d["ParkingTotal"] > 50, "more than 50 parking spaces")

    # HOA fee to a monthly amount (frequency is often blank: blank is treated as monthly)
    per = {"Monthly": 1, "Annually": 12, "Quarterly": 3, "SemiAnnually": 6}
    div = d["AssociationFeeFrequency"].map(per).fillna(1)
    d["hoa_monthly"] = d["AssociationFee"] / div
    to_nan("hoa_monthly", d["hoa_monthly"] > 5000, "HOA fee above $5,000 per month")

    d = d.drop(columns=["ListingContractDate"])  # cleaning check done; never a model input
    log = pd.DataFrame(log)
    log["pct_of_start"] = (log["removed"] / log.loc[0, "remaining"] * 100).round(3)
    return d.reset_index(drop=True), log, pd.DataFrame(fixes)


# ------------------------------------------------------------------------------------------------
# 3. Deterministic per-row features (no learning from other rows)
# ------------------------------------------------------------------------------------------------
def add_row_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["log_price"] = np.log(d["ClosePrice"])
    d["close_month"] = d["CloseDate"].dt.to_period("M")
    d["log_living_area"] = np.log(d["LivingArea"])
    d["log_lot_sqft"] = np.log(d["LotSizeSquareFeet"])
    # Age at the valuation date (CloseDate is the valuation date for historical rows)
    d["age"] = (d["CloseDate"].dt.year - d["YearBuilt"]).clip(lower=0)
    m = d["CloseDate"].dt.month
    d["month_sin"], d["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    for c in BIN_FEATURES:
        d[c] = _to_bool(d[c])
    lv = d["Levels"].fillna("").astype(str)
    d["levels_cat"] = np.select(
        [lv.str.contains("MultiSplit"), lv.str.contains("ThreeOrMore"), lv.str.contains("Two"), lv.str.contains("One")],
        ["MultiSplit", "ThreeOrMore", "Two", "One"], default=None)
    # fall back to Stories when Levels is blank
    fb = d["levels_cat"].isna()
    d.loc[fb & (d["Stories"] == 1), "levels_cat"] = "One"
    d.loc[fb & (d["Stories"] == 2), "levels_cat"] = "Two"
    d.loc[fb & (d["Stories"] >= 3), "levels_cat"] = "ThreeOrMore"
    d["zip5"] = d["PostalCode"]
    d["City"] = d["City"].astype("string").str.strip().str.title()
    hsd = d["HighSchoolDistrict"].astype("string").str.strip()
    d["HighSchoolDistrict"] = hsd.mask(hsd.isin(["Other", "", "None", "N/A"]))
    return d


# ------------------------------------------------------------------------------------------------
# 4. Comp-based neighborhood features (time-safe)
# ------------------------------------------------------------------------------------------------
def add_comp_features(d: pd.DataFrame, k: int = 10, lookback_months: int = 12,
                      ppsf_bounds=(50, 3000)) -> pd.DataFrame:
    """For each sale, summarise the k nearest sales that closed in the `lookback_months` months
    strictly BEFORE its own close month (never the same month, never later).

    This is the comp-based replacement for price per square foot: the row's own ClosePrice is
    never used, and the same rule works for an off-market address valued today.
      comp_log_ppsf     log of the median price per sq ft of the k nearest prior sales
      comp_log_price    median log close price of the k nearest prior sales
      comp_log_dist_km  log of the median distance (km) to those comps (comp density / reliability)
      comp_pool_months  how many prior months were available (early 2024 rows have fewer)
    Comp pool: cleaned sales with a sane price per sq ft (fixed domain bounds, not percentiles).
    """
    d = d.copy()
    for c in ["comp_log_ppsf", "comp_log_price", "comp_log_dist_km", "comp_pool_months"]:
        d[c] = np.nan
    ppsf = d["ClosePrice"] / d["LivingArea"]
    pool_ok = ppsf.between(*ppsf_bounds)
    rad = np.radians(d[["Latitude", "Longitude"]].to_numpy())
    months = sorted(d["close_month"].unique())
    first = months[0]
    for m in months:
        in_pool = pool_ok & (d["close_month"] < m) & (d["close_month"] >= m - lookback_months)
        idx_pool = np.flatnonzero(in_pool.to_numpy())
        idx_q = np.flatnonzero((d["close_month"] == m).to_numpy())
        if len(idx_pool) < k:
            continue
        tree = BallTree(rad[idx_pool], metric="haversine")
        dist, nn = tree.query(rad[idx_q], k=k)
        nn = idx_pool[nn]
        d.iloc[idx_q, d.columns.get_loc("comp_log_ppsf")] = np.log(np.median(ppsf.to_numpy()[nn], axis=1))
        d.iloc[idx_q, d.columns.get_loc("comp_log_price")] = np.median(d["log_price"].to_numpy()[nn], axis=1)
        d.iloc[idx_q, d.columns.get_loc("comp_log_dist_km")] = np.log(np.median(dist, axis=1) * EARTH_KM + 0.01)
        d.iloc[idx_q, d.columns.get_loc("comp_pool_months")] = min(lookback_months, (m - first).n)
    return d


# ------------------------------------------------------------------------------------------------
# 5. Chronological split and train-only outlier thresholds
# ------------------------------------------------------------------------------------------------
def window_split(d: pd.DataFrame, test_month, x_months: int):
    """Test = `test_month`; train = the `x_months` months immediately before it."""
    tm = pd.Period(test_month, "M")
    train = d[(d["close_month"] >= tm - x_months) & (d["close_month"] < tm)]
    test = d[d["close_month"] == tm]
    return train, test


def fit_thresholds(train: pd.DataFrame, q=(0.005, 0.995)) -> dict:
    """0.5th / 99.5th percentiles of ClosePrice and of price per sq ft, learned on TRAIN ONLY."""
    ppsf = train["ClosePrice"] / train["LivingArea"]
    lo, hi = train["ClosePrice"].quantile(q)
    plo, phi = ppsf.quantile(q)
    return {"price_lo": float(lo), "price_hi": float(hi), "ppsf_lo": float(plo), "ppsf_hi": float(phi)}


def within_thresholds(d: pd.DataFrame, t: dict) -> pd.Series:
    """Boolean mask; the same frozen thresholds are applied to train and test."""
    ppsf = d["ClosePrice"] / d["LivingArea"]
    return d["ClosePrice"].between(t["price_lo"], t["price_hi"]) & ppsf.between(t["ppsf_lo"], t["ppsf_hi"])


# ------------------------------------------------------------------------------------------------
# 6. Imputation, encoding, scaling: one ColumnTransformer, fit on train only
# ------------------------------------------------------------------------------------------------
def build_preprocessor(min_frequency: int = 30, features: dict | None = None) -> ColumnTransformer:
    """Missing values, categorical encoding and scaling, wrapped so `fit` only ever sees train.

    numeric      median imputation + missing-indicator columns, then standard scaling
    binary       missing -> 0 plus a missing-indicator column (missing is not the same as "No")
    low-card     County, Levels: one-hot; categories with < min_frequency train rows pooled as infrequent
    high-card    ZIP, City, High-school district: cross-fitted target encoding of log price (each
                 training row's encoding is computed without its own fold), then scaled

    `features` (optional, Week 6+): dict with keys num / bin / low / high giving the column lists.
    Default = the Week 3 lists above, so earlier notebooks behave exactly as before.
    """
    f = features or {"num": NUM_FEATURES, "bin": BIN_FEATURES, "low": LOWCARD_FEATURES, "high": HIGHCARD_FEATURES}
    assert_no_leakage(sum(f.values(), []))
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
    binary = SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True)
    low = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                             min_frequency=min_frequency, sparse_output=False))])
    high = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                     ("target_enc", TargetEncoder(target_type="continuous", cv=5, shuffle=True,
                                                  random_state=SEED)),
                     ("scale", StandardScaler())])
    return ColumnTransformer([("num", num, list(f["num"])), ("bin", binary, list(f["bin"])),
                              ("low", low, list(f["low"])), ("high", high, list(f["high"]))],
                             verbose_feature_names_out=True)


def model_frame(d: pd.DataFrame, features: dict | None = None) -> pd.DataFrame:
    """Feature columns in the exact dtypes the preprocessor expects (default: Week 3 lists)."""
    f = features or {"num": NUM_FEATURES, "bin": BIN_FEATURES, "low": LOWCARD_FEATURES, "high": HIGHCARD_FEATURES}
    X = d[sum(f.values(), [])].copy()
    for c in list(f["low"]) + list(f["high"]):
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


# ------------------------------------------------------------------------------------------------
# 7. Metrics (dollar scale)
# ------------------------------------------------------------------------------------------------
def price_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ape = np.abs(y_pred - y_true) / y_true
    return {"R2": r2_score(y_true, y_pred), "MAPE": ape.mean() * 100, "MdAPE": np.median(ape) * 100,
            "MAE": mean_absolute_error(y_true, y_pred), "within_10pct": (ape <= 0.10).mean() * 100}
