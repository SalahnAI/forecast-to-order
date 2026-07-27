"""Quantile demand models.

Two models with the same interface (fit / predict_quantiles):

- ``GBMQuantile``: one gradient-boosted tree per quantile (scikit-learn
  ``HistGradientBoostingRegressor`` with pinball loss). Deliberately boring
  and dependency-light — the point of this repo is the decision layer, and a
  well-featured GBM is a strong tabular baseline that trains in seconds.
- ``SeasonalNaiveQuantile``: empirical quantiles per (store, product, weekday)
  computed on the training window. The honesty check every learned model has
  to beat.

Predicted quantiles are sorted per row before being returned (quantile
crossing is possible since each quantile is a separate model).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import CATEGORICAL, FEATURES, TARGET

QUANTILES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def q_col(q: float) -> str:
    return f"q_{q:g}"


def _sort_quantile_columns(pred: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    cols = [q_col(q) for q in quantiles]
    pred[cols] = np.sort(pred[cols].to_numpy(), axis=1)
    return pred


class GBMQuantile:
    def __init__(self, quantiles: list[float] = QUANTILES, max_iter: int = 150):
        self.quantiles = quantiles
        self.max_iter = max_iter
        self.models: dict[float, HistGradientBoostingRegressor] = {}

    def fit(self, df: pd.DataFrame) -> "GBMQuantile":
        X, y = df[FEATURES], df[TARGET]
        for q in self.quantiles:
            m = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=self.max_iter,
                learning_rate=0.08,
                max_depth=6,
                categorical_features=CATEGORICAL,
                random_state=0,
            )
            m.fit(X, y)
            self.models[q] = m
        return self

    def predict_quantiles(self, df: pd.DataFrame) -> pd.DataFrame:
        pred = df[["store_id", "product_id", "date"]].copy()
        for q in self.quantiles:
            pred[q_col(q)] = np.clip(self.models[q].predict(df[FEATURES]), 0.0, None)
        return _sort_quantile_columns(pred, self.quantiles)


class SeasonalNaiveQuantile:
    """Empirical per-(store, product, weekday) quantiles from the train window."""

    def __init__(self, quantiles: list[float] = QUANTILES):
        self.quantiles = quantiles
        self.table: pd.DataFrame | None = None
        self.global_q: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> "SeasonalNaiveQuantile":
        def emp(group: pd.Series) -> pd.Series:
            return pd.Series(
                np.quantile(group, self.quantiles), index=[q_col(q) for q in self.quantiles]
            )

        self.table = (
            df.groupby(["store_id", "product_id", "dow"])[TARGET].apply(emp).unstack().reset_index()
        )
        self.global_q = pd.Series(
            np.quantile(df[TARGET], self.quantiles), index=[q_col(q) for q in self.quantiles]
        )
        return self

    def predict_quantiles(self, df: pd.DataFrame) -> pd.DataFrame:
        pred = df[["store_id", "product_id", "date", "dow"]].merge(
            self.table, on=["store_id", "product_id", "dow"], how="left"
        )
        cols = [q_col(q) for q in self.quantiles]
        pred[cols] = pred[cols].fillna(self.global_q)
        return _sort_quantile_columns(pred.drop(columns=["dow"]), self.quantiles)
