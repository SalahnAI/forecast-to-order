"""Forecast evaluation metrics.

Point metrics (wMAPE, MedMAPE) are computed on the median forecast; the
distribution itself is scored with pinball loss and interval coverage — for an
ordering decision, calibrated quantiles matter more than the point estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pinball_loss(y: np.ndarray, yhat: np.ndarray, q: float) -> float:
    diff = y - yhat
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def wmape(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sum(np.abs(y - yhat)) / max(np.sum(np.abs(y)), 1e-9))


def medmape(df: pd.DataFrame, actual: str, pred: str) -> float:
    """Median across (store, product) series of the per-series wMAPE.

    Robust to the long tail: a handful of erratic slow movers cannot dominate
    the headline number the way they do with a plain mean of per-row APEs.
    """
    per_series = df.groupby(["store_id", "product_id"]).apply(
        lambda g: wmape(g[actual].to_numpy(), g[pred].to_numpy()), include_groups=False
    )
    return float(per_series.median())
