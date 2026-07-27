"""Feature engineering for the quantile demand model.

All lag features are shifted so that the features available for day D use only
information known at the end of day D-1 (plus the day-ahead weather forecast
and the promo calendar, which are known in advance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ["store_id", "product_id", "dow"]
NUMERIC = ["doy_sin", "doy_cos", "temp_fc", "promo", "lag_7", "lag_14", "same_dow_4w", "roll_28"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "demand"


def build_features(sales: pd.DataFrame) -> pd.DataFrame:
    df = sales.sort_values(["store_id", "product_id", "date"]).copy()
    doy = df["date"].dt.dayofyear
    df["dow"] = df["date"].dt.dayofweek
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    g = df.groupby(["store_id", "product_id"], sort=False)["demand"]
    df["lag_7"] = g.shift(7)
    df["lag_14"] = g.shift(14)
    df["same_dow_4w"] = (g.shift(7) + g.shift(14) + g.shift(21) + g.shift(28)) / 4.0
    df["roll_28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())

    return df.dropna(subset=["lag_7", "lag_14", "same_dow_4w", "roll_28"]).reset_index(drop=True)


def train_test_split_by_date(df: pd.DataFrame, test_days: int = 84) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["date"].max() - pd.Timedelta(days=test_days - 1)
    return df[df["date"] < cutoff].copy(), df[df["date"] >= cutoff].copy()
