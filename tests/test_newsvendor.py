import numpy as np
import pytest

from forecast_to_order.newsvendor import critical_ratio, order_quantity, quantile_from_forecast


def test_critical_ratio_matches_margin_share():
    # Cu = price - cost, Co = cost => q* = margin / price
    assert critical_ratio(price=4.0, cost=1.0) == pytest.approx(0.75)
    assert critical_ratio(price=2.0, cost=1.0) == pytest.approx(0.5)


def test_critical_ratio_salvage_raises_target():
    assert critical_ratio(4.0, 2.0, salvage=1.0) > critical_ratio(4.0, 2.0, salvage=0.0)


def test_critical_ratio_rejects_degenerate_economics():
    with pytest.raises(ValueError):
        critical_ratio(price=1.0, cost=1.0)


def test_quantile_interpolation_is_monotone_and_clamped():
    qs = np.array([0.1, 0.5, 0.9])
    vals = np.array([2.0, 5.0, 11.0])
    assert quantile_from_forecast(qs, vals, 0.5) == pytest.approx(5.0)
    assert quantile_from_forecast(qs, vals, 0.7) == pytest.approx(8.0)
    # Clamped at the edges rather than extrapolated.
    assert quantile_from_forecast(qs, vals, 0.01) == pytest.approx(2.0)
    assert quantile_from_forecast(qs, vals, 0.99) == pytest.approx(11.0)


def test_order_rounds_up_to_whole_packs():
    assert order_quantity(target_level=10.0, inventory_position=0.0, pack_size=6) == 12
    assert order_quantity(target_level=12.0, inventory_position=0.0, pack_size=6) == 12
    assert order_quantity(target_level=10.0, inventory_position=4.5, pack_size=6) == 6


def test_no_order_when_stock_covers_target():
    assert order_quantity(target_level=5.0, inventory_position=5.0, pack_size=6) == 0
    assert order_quantity(target_level=5.0, inventory_position=9.0, pack_size=6) == 0
