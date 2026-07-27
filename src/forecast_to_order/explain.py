"""Human-readable explanations for order recommendations.

Store staff accept or override each recommendation, and acceptance rate is a
product metric: an order line that can say *why* earns more trust than a bare
number. Explanations here are deterministic templates built from the exact
inputs of the decision (forecast quantiles, critical ratio, pack rounding,
current stock) — auditable, reproducible, and cheap enough to generate for
every line of every order.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrderContext:
    store_id: int
    product_id: int
    category: str
    date: str
    q10: float
    q50: float
    q90: float
    target_q: float
    target_level: float
    inventory: float
    pack_size: int
    order: int
    price: float
    cost: float


def explain_order(ctx: OrderContext) -> str:
    margin = ctx.price - ctx.cost
    lines = [
        f"Order {ctx.order} units of product {ctx.product_id} ({ctx.category}) "
        f"for store {ctx.store_id} on {ctx.date}.",
        f"Demand forecast: most likely around {ctx.q50:.0f} units, "
        f"with an 80% range of {ctx.q10:.0f} to {ctx.q90:.0f}.",
        f"Each unmet unit forgoes {margin:.2f} of margin while each wasted unit "
        f"loses {ctx.cost:.2f}, so we cover demand up to its "
        f"{ctx.target_q:.0%} quantile ({ctx.target_level:.1f} units).",
    ]
    if ctx.inventory > 0:
        lines.append(f"{ctx.inventory:.0f} units are already on the shelf and are deducted.")
    shortfall = max(ctx.target_level - ctx.inventory, 0.0)
    if ctx.order > 0 and ctx.pack_size > 1:
        lines.append(
            f"The shortfall of {shortfall:.1f} is rounded up to whole packs of "
            f"{ctx.pack_size}, giving {ctx.order} units."
        )
    elif ctx.order == 0:
        lines.append("Current stock already covers the target level, so no order is needed.")
    return " ".join(lines)
