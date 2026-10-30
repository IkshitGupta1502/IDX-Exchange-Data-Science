"""
feature_engineering.py - Week 6 engineered features for the California single-family close-price model.

Two groups of new inputs, both available for any address at prediction time (team feature rule):

1. Home-shape features, computed from the home's own physical fields only (never from a price):
     beds_per_bath      bedrooms / bathrooms
     log_sqft_per_bed   log(living area / bedrooms): room size
     log_lot_to_living  log(lot size / living area): how much land per sq ft of house
   (Age of the home in years has been a model input since Week 3.)

2. School district layer: each home's coordinates are spatially joined to the CDE
   "California School District Areas 2024-25" boundaries. Elementary and high school districts
   overlap (they serve different grades in the same territory), so every home gets
     sd_elem_name   the district serving grades K-8 (a unified district, or else the elementary district)
     sd_high_name   the district serving grades 9-12 (a unified district, or else the high school district)
     sd_type        "Unified" or "Elementary + High"
     sd_assist      state accountability status of the K-8 district (General / Differentiated)
   plus numeric attributes of the K-8 district from the same file (SD_NUM). Names are
   target-encoded by the preprocessor exactly like ZIP and City.

The join result is cached to processed/school_district_join.csv (one row per ListingKey), so later
weeks do not need the boundary file or shapely.

Used by: notebooks/05_feature_engineering.ipynb and later weeks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import preprocessing as pp

# ------------------------------------------------------------------------------------------------
# Feature groups (added on top of the Week 3 lists in preprocessing.py)
# ------------------------------------------------------------------------------------------------
HOME_NUM = ["beds_per_bath", "log_sqft_per_bed", "log_lot_to_living"]

# School district layer (CDE "California School District Areas 2024-25", 937 polygons)
NAME_COL, TYPE_COL, COUNTY_COL = "DistrictName", "DistrictType", "CountyName"
SD_NUM = ["sd_sed_pct", "sd_el_pct", "sd_charter_pct", "sd_log_enroll_density"]
SD_NUM_DESC = {
    "sd_sed_pct": "% socioeconomically disadvantaged students",
    "sd_el_pct": "% English learners",
    "sd_charter_pct": "% of enrollment in charter schools",
    "sd_log_enroll_density": "log students per sq mi",
}
SD_LOWCARD = ["sd_type", "sd_assist"]
SD_HIGHCARD = ["sd_elem_name", "sd_high_name"]
# Deliberately NOT used: the race/ethnicity counts and percentages in the file (AA/AI/AS/FI/HI/PI/WH/MR).
# Valuing a home by the racial makeup of its area is a fair-housing problem, even if it predicts.
# ELpct in the file is mis-scaled (e.g. 0.2 for a district with 8% English learners), so the share is
# recomputed from ELcount / EnrollTotal.


def district_attributes(attrs: pd.DataFrame) -> pd.DataFrame:
    """Per-district numeric and categorical attributes used as features (index = polygon order)."""
    enroll = attrs["EnrollTotal"].astype(float)
    a = pd.DataFrame(index=attrs.index)
    a["sd_sed_pct"] = attrs["SEDpct"].astype(float)
    a["sd_el_pct"] = attrs["ELcount"] / enroll.where(enroll > 0) * 100
    a["sd_charter_pct"] = attrs["EnrollCharter"] / enroll.where(enroll > 0) * 100
    a["sd_log_enroll_density"] = np.log1p(enroll / attrs["DistrctAreaSqMi"].astype(float))
    # state accountability status: "Differentiated" = the district receives differentiated assistance
    # because one or more student groups are performing poorly (a school-performance signal)
    a["sd_assist"] = np.where(attrs["AssistStatus"].astype(str).str.startswith("Differentiated"),
                              "Differentiated", np.where(attrs["AssistStatus"].astype(str) == "General", "General", None))
    return a


def find_boundary_file(root: Path) -> Path:
    """The district boundary GeoJSON in the project folder (top level or geo/)."""
    hits = sorted(list(Path(root).glob("*School_District_Areas*.geojson")) +
                  list((Path(root) / "geo").glob("*School_District_Areas*.geojson")))
    assert hits, "Download 'California School District Areas 2024-25' (GeoJSON) into the project folder"
    return hits[0]


def feature_set(home: bool = False, school: bool = False) -> dict:
    """Feature groups for the preprocessor: Week 3 inputs, optionally plus the Week 6 groups."""
    fs = {"num": list(pp.NUM_FEATURES), "bin": list(pp.BIN_FEATURES),
          "low": list(pp.LOWCARD_FEATURES), "high": list(pp.HIGHCARD_FEATURES)}
    if home:
        fs["num"] += HOME_NUM
    if school:
        fs["num"] += SD_NUM
        fs["low"] += SD_LOWCARD
        fs["high"] += SD_HIGHCARD
    pp.assert_no_leakage(sum(fs.values(), []))
    return fs


# ------------------------------------------------------------------------------------------------
# 1. Home-shape features
# ------------------------------------------------------------------------------------------------
def add_home_features(d: pd.DataFrame) -> pd.DataFrame:
    """Per-row features from physical fields only. Missing inputs give a missing feature (imputed later)."""
    d = d.copy()
    beds, baths = d["BedroomsTotal"], d["BathroomsTotalInteger"]
    d["beds_per_bath"] = beds / baths
    d["log_sqft_per_bed"] = np.log(d["LivingArea"] / beds)
    d["log_lot_to_living"] = np.log(d["LotSizeSquareFeet"] / d["LivingArea"])
    return d


# ------------------------------------------------------------------------------------------------
# 2. School district spatial join
# ------------------------------------------------------------------------------------------------
def load_districts(path: Path):
    """Read the CDE district boundary GeoJSON. Returns (attribute table, list of shapely geometries)."""
    from shapely.geometry import shape

    gj = json.loads(Path(path).read_text())
    rows, geoms = [], []
    for f in gj["features"]:
        if f.get("geometry") is None:
            continue
        rows.append(f["properties"])
        geoms.append(shape(f["geometry"]).buffer(0))  # buffer(0) repairs any self-touching rings
    return pd.DataFrame(rows), geoms


def spatial_join(lat, lon, attrs: pd.DataFrame, geoms, name_col: str, type_col: str,
                 snap_m: float = 500.0) -> pd.DataFrame:
    """Point-in-polygon join of home coordinates to district polygons.

    Returns one row per point with the K-8 district, the 9-12 district, the district type and
    the index of the polygon used for each (for attaching numeric attributes). A point that falls in
    no polygon (coastline or boundary slivers) is snapped to the nearest polygon of each kind if it
    is within `snap_m` metres; otherwise it is left missing and imputed as "Unknown".
    """
    import shapely
    from shapely.strtree import STRtree

    pts = shapely.points(np.asarray(lon, float), np.asarray(lat, float))
    tree = STRtree(geoms)
    pi, gi = tree.query(pts, predicate="within")
    kind = attrs[type_col].astype(str).str.strip().str.lower()

    out = pd.DataFrame({"elem_idx": np.full(len(pts), -1), "high_idx": np.full(len(pts), -1),
                        "uni_idx": np.full(len(pts), -1), "snapped": False})
    for k, col in [("unified", "uni_idx"), ("elementary", "elem_idx"), ("high", "high_idx")]:
        m = kind.to_numpy()[gi] == k
        # if a point sits in two polygons of the same kind (rare overlap), keep the first
        s = pd.Series(gi[m], index=pi[m])
        s = s[~s.index.duplicated()]
        out.loc[s.index, col] = s.values

    # snap points that fall in no polygon at all to the nearest polygon of each kind
    none = np.flatnonzero((out[["elem_idx", "high_idx", "uni_idx"]] < 0).all(axis=1).to_numpy())
    if len(none):
        deg = snap_m / 111_000.0
        for k, col in [("unified", "uni_idx"), ("elementary", "elem_idx"), ("high", "high_idx")]:
            idx_k = np.flatnonzero(kind.to_numpy() == k)
            sub = STRtree([geoms[i] for i in idx_k])
            near_p, near_g = sub.query_nearest(pts[none], max_distance=deg, all_matches=False)
            out.loc[none[near_p], col] = idx_k[near_g]
        out.loc[none, "snapped"] = (out.loc[none, ["elem_idx", "high_idx", "uni_idx"]] >= 0).any(axis=1).to_numpy()

    names = attrs[name_col].astype(str).str.strip().to_numpy()
    uni = out.uni_idx.to_numpy()
    k8 = np.where(uni >= 0, uni, out.elem_idx.to_numpy())
    hs = np.where(uni >= 0, uni, out.high_idx.to_numpy())
    res = pd.DataFrame({
        "sd_elem_name": np.where(k8 >= 0, names[np.maximum(k8, 0)], None),
        "sd_high_name": np.where(hs >= 0, names[np.maximum(hs, 0)], None),
        "sd_type": np.select([uni >= 0, (k8 >= 0) | (hs >= 0)], ["Unified", "Elementary + High"], default=None),
        "sd_k8_idx": k8, "sd_hs_idx": hs, "sd_snapped": out.snapped.to_numpy(),
    })
    return res


def build_join_table(clean: pd.DataFrame, boundary_path: Path) -> pd.DataFrame:
    """Join every cleaned sale to its districts and attach the district attributes.

    Numeric attributes and assistance status come from the K-8 district (unified or elementary):
    one value per attribute per home. District names are suffixed with the county because 11
    district names repeat across California.
    """
    attrs, geoms = load_districts(boundary_path)
    attrs = attrs.copy()
    attrs["_key"] = attrs[NAME_COL].astype(str).str.strip() + " (" + attrs[COUNTY_COL].astype(str) + ")"
    j = spatial_join(clean["Latitude"], clean["Longitude"], attrs, geoms, "_key", TYPE_COL)
    da = district_attributes(attrs)
    k8 = j.sd_k8_idx.to_numpy()
    for c in da.columns:
        vals = da[c].to_numpy()
        j[c] = np.where(k8 >= 0, vals[np.maximum(k8, 0)], None if da[c].dtype == object else np.nan)
    j.insert(0, "ListingKey", clean["ListingKey"].to_numpy())
    return j.drop(columns=["sd_k8_idx", "sd_hs_idx"])


def add_school_features(d: pd.DataFrame, join_table: pd.DataFrame) -> pd.DataFrame:
    """Merge the cached join table onto rows by ListingKey (row order preserved)."""
    cols = [c for c in join_table.columns if c != "ListingKey"]
    out = d.drop(columns=[c for c in cols if c in d.columns]).merge(join_table, on="ListingKey", how="left",
                                                                   validate="m:1")
    assert len(out) == len(d)
    out.index = d.index
    return out
