import numpy as np
import pandas as pd
import pytest

from forecast_to_order.data import generate_sales
from forecast_to_order.features import build_features, train_test_split_by_date
from forecast_to_order.models import QUANTILES, SeasonalNaiveQuantile, q_col
from forecast_to_order.simulate import simulate_policy


@pytest.fixture(scope="module")
def small_world():
    sales, catalog, _ = generate_sales(n_stores=2, n_products=4, n_days=200)
    feats = build_features(sales)
    train, test = train_test_split_by_date(feats, test_days=28)
    model = SeasonalNaiveQuantile().fit(train)
    test = test.merge(model.predict_quantiles(test), on=["store_id", "product_id", "date"])
    return test, catalog


def test_unit_conservation(small_world):
    test, catalog = small_world
    r = simulate_policy(test, catalog, "q=0.7", QUANTILES)
    # Every delivered unit is sold, wasted, or still on the shelf at the end;
    # sold + wasted can never exceed delivered.
    assert r.sold + r.wasted <= r.delivered
    assert r.sold <= r.demand
    assert 0.0 <= r.fill_rate <= 1.0
    assert 0.0 <= r.waste_rate <= 1.0


def test_higher_quantile_raises_availability_and_waste(small_world):
    test, catalog = small_world
    lo = simulate_policy(test, catalog, "q=0.3", QUANTILES)
    hi = simulate_policy(test, catalog, "q=0.95", QUANTILES)
    assert hi.fill_rate > lo.fill_rate
    assert hi.waste_rate >= lo.waste_rate


def test_policies_run_on_all_supported_names(small_world):
    test, catalog = small_world
    for pol in ["naive", "newsvendor", "q=0.5"]:
        r = simulate_policy(test, catalog, pol, QUANTILES)
        assert r.delivered > 0


def test_order_log_is_consistent(small_world):
    test, catalog = small_world
    r = simulate_policy(test, catalog, "q=0.7", QUANTILES, collect_orders=True)
    log = r.orders
    assert log is not None and len(log) > 0
    assert (log["order"] >= 0).all()
    assert (log["inventory"] >= 0).all()
    assert (log["target_q"] == 0.7).all()
    # Every ordered quantity is a whole number of packs.
    packs = log.merge(catalog[["product_id", "pack_size"]], on="product_id")
    assert (packs["order"] % packs["pack_size"] == 0).all()
    # One decision per (store, product) per day after the first.
    n_pairs = test.groupby(["store_id", "product_id"]).ngroups
    n_days = test["date"].nunique()
    assert len(log) == n_pairs * (n_days - 1)
