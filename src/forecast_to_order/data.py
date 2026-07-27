"""Synthetic fresh-retail sales generator.

Simulates daily unit demand for fresh products across a store network with the
structure real fresh data has: strong weekday cycles, yearly seasonality,
weather sensitivity, promotions, and a long tail of slow movers whose demand is
intermittent. Demand is drawn from a negative binomial so counts are integer,
overdispersed, and naturally sparse for low-velocity products.

Everything is seeded and deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Weekday demand multipliers, Monday..Sunday (grocery peaks on the weekend).
WEEKDAY_FACTOR = np.array([0.95, 0.85, 0.90, 0.95, 1.10, 1.35, 1.25])

# Category -> (shelf life in days once on the shelf, demand sensitivity per °C
# above the yearly average temperature).
CATEGORIES = {
    "produce": {"shelf_life": 3, "temp_sens": 0.020},
    "bakery": {"shelf_life": 1, "temp_sens": 0.000},
    "meat": {"shelf_life": 3, "temp_sens": -0.005},
    "seafood": {"shelf_life": 2, "temp_sens": 0.005},
    "deli": {"shelf_life": 2, "temp_sens": 0.000},
}

MEAN_TEMP_C = 12.0
NEGBIN_DISPERSION = 3.0  # lower = more overdispersed


def generate_catalog(n_products: int, rng: np.random.Generator) -> pd.DataFrame:
    """Product master data: economics, shelf life, pack size, base velocity."""
    names = list(CATEGORIES)
    rows = []
    for pid in range(n_products):
        cat = names[pid % len(names)]
        base = float(np.exp(rng.normal(1.6, 1.1)))  # lognormal: fast and slow movers
        cost = round(float(rng.uniform(0.8, 6.0)), 2)
        margin = float(rng.uniform(0.25, 0.55))
        rows.append(
            {
                "product_id": pid,
                "category": cat,
                "base_demand": base,
                "cost": cost,
                "price": round(cost / (1.0 - margin), 2),
                "shelf_life": CATEGORIES[cat]["shelf_life"],
                "temp_sens": CATEGORIES[cat]["temp_sens"],
                "pack_size": int(rng.choice([1, 1, 4, 6, 10])),
            }
        )
    return pd.DataFrame(rows)


def generate_stores(n_stores: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_id": np.arange(n_stores),
            "store_size": np.exp(rng.normal(0.0, 0.4, size=n_stores)),
        }
    )


def generate_sales(
    n_stores: int = 15,
    n_products: int = 24,
    n_days: int = 730,
    start: str = "2024-01-01",
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (sales, catalog, stores).

    sales has one row per (store, product, day) with the realized integer
    demand, the actual temperature, the day-ahead temperature *forecast* the
    model is allowed to see, and the promo flag.
    """
    rng = np.random.default_rng(seed)
    catalog = generate_catalog(n_products, rng)
    stores = generate_stores(n_stores, rng)

    dates = pd.date_range(start, periods=n_days, freq="D")
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()

    # Shared daily temperature: yearly sinusoid + noise, plus a noisy
    # day-ahead forecast of it.
    temp = MEAN_TEMP_C + 9.0 * np.sin(2 * np.pi * (doy - 100) / 365.25)
    temp = temp + rng.normal(0, 2.0, size=n_days)
    temp_fc = temp + rng.normal(0, 1.5, size=n_days)

    # Promotions: per product, whole weeks flagged with ~5% coverage.
    week_idx = np.arange(n_days) // 7
    n_weeks = week_idx.max() + 1
    promo_weeks = rng.random((n_products, n_weeks)) < 0.05
    promo = promo_weeks[:, week_idx]  # (P, D)
    promo_uplift = rng.uniform(1.5, 2.5, size=n_products)

    base = catalog["base_demand"].to_numpy()[:, None]  # (P, 1)
    sens = catalog["temp_sens"].to_numpy()[:, None]
    size = stores["store_size"].to_numpy()[:, None, None]  # (S, 1, 1)

    mu = base * WEEKDAY_FACTOR[dow][None, :]  # (P, D)
    mu = mu * (1.0 + sens * (temp - MEAN_TEMP_C)[None, :])
    mu = mu * np.where(promo, promo_uplift[:, None], 1.0)
    mu = np.clip(mu, 0.02, None)
    mu = mu * size  # (S, P, D)

    k = NEGBIN_DISPERSION
    p = k / (k + mu)
    demand = rng.negative_binomial(k, p)  # (S, P, D)

    s_idx, p_idx, d_idx = np.meshgrid(
        np.arange(n_stores), np.arange(n_products), np.arange(n_days), indexing="ij"
    )
    sales = pd.DataFrame(
        {
            "store_id": s_idx.ravel(),
            "product_id": p_idx.ravel(),
            "date": dates.to_numpy()[d_idx.ravel()],
            "demand": demand.ravel(),
            "temp": temp[d_idx.ravel()],
            "temp_fc": temp_fc[d_idx.ravel()],
            "promo": promo[p_idx.ravel(), d_idx.ravel()].astype(int),
        }
    )
    sales = sales.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    return sales, catalog, stores
