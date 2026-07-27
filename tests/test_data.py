import numpy as np

from forecast_to_order.data import generate_sales
from forecast_to_order.features import FEATURES, build_features


def test_generator_shape_and_domains():
    sales, catalog, stores = generate_sales(n_stores=3, n_products=5, n_days=120)
    assert len(sales) == 3 * 5 * 120
    assert (sales["demand"] >= 0).all()
    assert np.issubdtype(sales["demand"].dtype, np.integer)
    assert (catalog["price"] > catalog["cost"]).all()
    assert (catalog["shelf_life"] >= 1).all()
    assert (catalog["pack_size"] >= 1).all()


def test_generator_is_deterministic():
    a, _, _ = generate_sales(n_stores=2, n_products=3, n_days=60, seed=42)
    b, _, _ = generate_sales(n_stores=2, n_products=3, n_days=60, seed=42)
    assert a.equals(b)


def test_features_have_no_nans_and_no_leakage():
    sales, _, _ = generate_sales(n_stores=2, n_products=3, n_days=120)
    feats = build_features(sales)
    assert feats[FEATURES].notna().all().all()
    # lag_7 for a given day must equal demand 7 days earlier for that series.
    g = feats[(feats["store_id"] == 0) & (feats["product_id"] == 0)].sort_values("date")
    raw = sales[(sales["store_id"] == 0) & (sales["product_id"] == 0)].sort_values("date")
    raw_by_date = raw.set_index("date")["demand"]
    row = g.iloc[10]
    assert row["lag_7"] == raw_by_date[row["date"] - np.timedelta64(7, "D")]
