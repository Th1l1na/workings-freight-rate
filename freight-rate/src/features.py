"""Cleaning and feature engineering shared by training and inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_MI = 3958.8
QUARTER_END_WINDOW_DAYS = 28
TREND_ORIGIN = pd.Timestamp("2025-01-01")
WEIGHT_FLOOR = 1_000.0

# Features used by the linear base model (stage 1). Every one of these is
# chosen so that it extrapolates sensibly to dates outside the training window.
LINEAR_FEATURES = [
    "log_distance",
    "eq_reefer",
    "eq_flatbed",
    "log_weight",
    "log_market_index",
    "quote_signal",
    "quarter_end_ramp",
    "trend_years",
]

# Features used by the gradient-boosted residual model (stage 2). No city IDs
# (8 validation cities never appear in training) and no absolute time
# (trees cannot extrapolate a trend into Nov/Dec).
GBM_FEATURES = [
    "log_distance",
    "circuity",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "delta_lat",
    "delta_lon",
    "weight",
    "weight_missing",
    "log_market_index",
    "market_index_dev",
    "quote_signal",
    "eq_reefer",
    "eq_flatbed",
    "days_to_quarter_end",
    "day_of_week",
]


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))


def daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """Median market_index per calendar date across all supplied frames.

    market_index is a market-level signal that varies only slightly between
    loads on the same day (std ~0.025), so the daily median is a robust
    imputation value and lets us reconstruct it for the December chart rows.
    """
    stacked = pd.concat([f[["date", "market_index"]] for f in frames], ignore_index=True)
    stacked["date"] = pd.to_datetime(stacked["date"])
    return stacked.groupby("date")["market_index"].median().sort_index()


def clean(frame: pd.DataFrame, daily_mi: pd.Series) -> pd.DataFrame:
    """Fix the data-quality issues found during EDA. Never drops rows."""
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["equipment"] = df["equipment"].str.strip()

    # Weight: ~0.6% negative (sign errors, magnitudes match the normal range),
    # ~0.6% missing. Take abs; leave NaN for LightGBM and impute for the
    # linear model with the equipment median.
    df["weight_was_negative"] = (df["weight"] < 0).astype(float)
    df["weight"] = df["weight"].abs()
    df["weight_missing"] = df["weight"].isna().astype(float)

    # market_index: ~0.8% missing -> daily median of the market signal.
    df["market_index_missing"] = df["market_index"].isna().astype(float)
    day_mi = df["date"].map(daily_mi)
    df["market_index"] = df["market_index"].fillna(day_mi)
    df["market_index_day"] = day_mi.fillna(df["market_index"])
    return df


def build_features(df: pd.DataFrame, weight_fill: dict[str, float] | None = None) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    great_circle = haversine_miles(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    f["log_distance"] = np.log(df["distance"].astype(float))
    f["circuity"] = df["distance"] / np.maximum(great_circle, 1.0)
    for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        f[col] = df[col].astype(float)
    f["delta_lat"] = df["delivery_lat"] - df["pickup_lat"]
    f["delta_lon"] = df["delivery_lon"] - df["pickup_lon"]

    f["eq_reefer"] = (df["equipment"] == "Reefer").astype(float)
    f["eq_flatbed"] = (df["equipment"] == "Flatbed").astype(float)

    f["weight"] = df["weight"].astype(float)
    f["weight_missing"] = df["weight"].isna().astype(float)
    weight_filled = df["weight"].copy()
    if weight_fill:
        weight_filled = weight_filled.fillna(df["equipment"].map(weight_fill))
    weight_filled = weight_filled.fillna(31_000.0)
    f["log_weight"] = np.log(weight_filled.clip(lower=WEIGHT_FLOOR))

    f["log_market_index"] = np.log(df["market_index"].astype(float))
    f["market_index_dev"] = np.log(df["market_index"] / df["market_index_day"])
    f["quote_signal"] = df["quote_signal"].astype(float)

    date = pd.to_datetime(df["date"])
    quarter_end = date + pd.offsets.QuarterEnd(0)
    f["days_to_quarter_end"] = (quarter_end - date).dt.days.astype(float)
    f["quarter_end_ramp"] = np.clip(1 - f["days_to_quarter_end"] / QUARTER_END_WINDOW_DAYS, 0, 1)
    f["day_of_week"] = date.dt.dayofweek.astype(float)
    f["trend_years"] = (date - TREND_ORIGIN).dt.days / 365.0
    return f
