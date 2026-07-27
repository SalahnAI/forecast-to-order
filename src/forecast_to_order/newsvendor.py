"""Turning a demand distribution into an order quantity.

The classic newsvendor result: with underage cost Cu (margin lost per unit of
unmet demand) and overage cost Co (cash lost per unit ordered and wasted), the
expected-profit-maximizing order is the critical-ratio quantile of the demand
distribution:

    q* = Cu / (Cu + Co)

The mean is the optimal order only in the special case Cu == Co. This is the
whole argument for probabilistic forecasting in replenishment: the decision
consumes a quantile, not a point estimate — and which quantile is a business
choice (waste vs availability), not a modeling one.

The single-period model is conservative for products with shelf life > 1 day
(an unsold unit is not always wasted — it can sell tomorrow), so the effective
overage cost is lower and the profit-maximizing quantile sits above q*. The
simulation sweep in ``simulate.py`` measures that gap empirically.
"""
from __future__ import annotations

import math

import numpy as np


def critical_ratio(price: float, cost: float, salvage: float = 0.0) -> float:
    cu = price - cost
    co = cost - salvage
    if cu <= 0 or co <= 0:
        raise ValueError("need price > cost > salvage")
    return cu / (cu + co)


def quantile_from_forecast(quantiles: np.ndarray, values: np.ndarray, target_q: float) -> float:
    """Read the inverse CDF at ``target_q`` by interpolating between predicted
    quantiles (clamped at the edges)."""
    return float(np.interp(target_q, quantiles, values))


def order_quantity(target_level: float, inventory_position: float, pack_size: int) -> int:
    """Order-up-to policy under a pack-size constraint.

    Order enough whole packs to bring the inventory position up to at least
    ``target_level``; never order a negative quantity.
    """
    shortfall = target_level - inventory_position
    if shortfall <= 0:
        return 0
    return int(math.ceil(shortfall / pack_size)) * pack_size
