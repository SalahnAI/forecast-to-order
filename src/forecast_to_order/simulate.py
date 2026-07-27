"""Replay a daily order-and-sell loop over the test window.

For every (store, product) pair, each simulated day:

1. the order placed yesterday evening arrives before opening (lead time 1 day)
   with the product's full shelf life remaining;
2. the day's actual demand is served FIFO (oldest stock first); unmet demand
   is lost;
3. stock that reaches the end of its shelf life unsold becomes waste;
4. an order for tomorrow is placed from tomorrow's forecast: the target level
   is a chosen quantile of the predicted demand distribution, and the order is
   the pack-rounded shortfall vs the current inventory position.

KPIs per policy: fill rate (demand served), waste rate (share of delivered
units wasted), realized profit. Sweeping the target quantile traces the
waste/availability frontier that a point forecast collapses to a single,
implicit point on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import q_col
from .newsvendor import critical_ratio, order_quantity, quantile_from_forecast


@dataclass
class SimResult:
    policy: str
    fill_rate: float
    waste_rate: float
    profit: float
    demand: int
    sold: int
    wasted: int
    delivered: int
    orders: pd.DataFrame | None = None  # per-line decision log (collect_orders=True)


def _target_level(day: dict, policy: str, quantiles: np.ndarray, crit_q: float) -> float:
    if policy == "naive":
        return float(day["lag_7"])
    if policy == "newsvendor":
        q = crit_q
    else:  # "q=0.5", "q=0.9", ...
        q = float(policy.split("=")[1])
    values = np.array([day[q_col(x)] for x in quantiles], dtype=float)
    return quantile_from_forecast(quantiles, values, q)


def simulate_policy(
    test: pd.DataFrame,
    catalog: pd.DataFrame,
    policy: str,
    quantiles: list[float],
    collect_orders: bool = False,
) -> SimResult:
    """``test`` needs demand, lag_7 and the q_* prediction columns, one row per
    (store, product, date)."""
    qs = np.asarray(quantiles)
    cat = catalog.set_index("product_id")
    tot_demand = tot_sold = tot_waste = tot_delivered = 0
    profit = 0.0
    order_log: list[dict] = []

    for (sid, pid), g in test.groupby(["store_id", "product_id"], sort=False):
        g = g.sort_values("date")
        prod = cat.loc[pid]
        life = int(prod["shelf_life"])
        pack = int(prod["pack_size"])
        price, cost = float(prod["price"]), float(prod["cost"])
        crit_q = critical_ratio(price, cost)

        days = g.to_dict("records")

        # inv[i] = units with i+1 days of remaining life.
        inv = np.zeros(life, dtype=float)
        # Warm start: seed the shelf at the first day's target so no policy is
        # penalized for an empty day-0 shelf.
        first_target = _target_level(days[0], policy, qs, crit_q)
        inv[-1] = order_quantity(first_target, 0.0, pack)
        tot_delivered += inv[-1]
        profit -= cost * inv[-1]
        on_order = 0.0

        for i, day in enumerate(days):
            # Morning: receive yesterday evening's order.
            if on_order > 0:
                inv[-1] += on_order
                tot_delivered += on_order
                profit -= cost * on_order
                on_order = 0.0

            # Serve demand FIFO (oldest stock first).
            d = float(day["demand"])
            remaining = d
            for a in range(life):
                take = min(inv[a], remaining)
                inv[a] -= take
                remaining -= take
            sold = d - remaining
            tot_demand += d
            tot_sold += sold
            profit += price * sold

            # Evening: expire and age stock.
            tot_waste += inv[0]
            inv[:-1] = inv[1:]
            inv[-1] = 0.0

            # Place tomorrow's order from tomorrow's forecast.
            if i + 1 < len(days):
                tomorrow = days[i + 1]
                target = _target_level(tomorrow, policy, qs, crit_q)
                position = float(inv.sum())
                on_order = order_quantity(target, position, pack)
                if collect_orders:
                    order_log.append(
                        {
                            "store_id": sid,
                            "product_id": pid,
                            "date": tomorrow["date"],
                            "target_q": crit_q if policy == "newsvendor" else (
                                np.nan if policy == "naive" else float(policy.split("=")[1])
                            ),
                            "target_level": target,
                            "inventory": position,
                            "order": on_order,
                            "q10": tomorrow[q_col(0.1)],
                            "q50": tomorrow[q_col(0.5)],
                            "q90": tomorrow[q_col(0.9)],
                        }
                    )

        # Unsold, unexpired stock at the horizon: count its cash back so the
        # profit comparison is not distorted by end-of-window inventory.
        profit += cost * inv.sum()

    return SimResult(
        policy=policy,
        fill_rate=tot_sold / max(tot_demand, 1e-9),
        waste_rate=tot_waste / max(tot_delivered, 1e-9),
        profit=profit,
        demand=int(tot_demand),
        sold=int(tot_sold),
        wasted=int(tot_waste),
        delivered=int(tot_delivered),
        orders=pd.DataFrame(order_log) if collect_orders else None,
    )
