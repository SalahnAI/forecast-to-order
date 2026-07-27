#!/usr/bin/env python
"""End-to-end demo: data -> quantile forecasts -> orders -> simulation -> report.

Usage:
    python scripts/run_demo.py [--explain N]

Writes metrics tables (CSV) and two figures (forecast fan chart, waste vs
availability frontier) to outputs/.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast_to_order.data import generate_sales
from forecast_to_order.explain import OrderContext, explain_order
from forecast_to_order.features import build_features, train_test_split_by_date
from forecast_to_order.metrics import coverage, medmape, pinball_loss, wmape
from forecast_to_order.models import QUANTILES, GBMQuantile, SeasonalNaiveQuantile, q_col
from forecast_to_order.newsvendor import critical_ratio, order_quantity, quantile_from_forecast
from forecast_to_order.simulate import simulate_policy

SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def log(msg: str, t0: float) -> None:
    print(f"[{time.perf_counter() - t0:6.1f}s] {msg}", flush=True)


def forecast_metrics(name: str, test: pd.DataFrame) -> dict:
    y = test["demand"].to_numpy(dtype=float)
    med = test[q_col(0.5)].to_numpy()
    promo = test[test["promo"] == 1]
    return {
        "model": name,
        "mean_pinball": np.mean([pinball_loss(y, test[q_col(q)].to_numpy(), q) for q in QUANTILES]),
        "wmape_q50": wmape(y, med),
        "wmape_promo_days": wmape(promo["demand"].to_numpy(dtype=float), promo[q_col(0.5)].to_numpy()),
        "medmape_q50": medmape(test, "demand", q_col(0.5)),
        "coverage_80": coverage(y, test[q_col(0.1)].to_numpy(), test[q_col(0.9)].to_numpy()),
    }


def fan_chart(test: pd.DataFrame, outdir: Path) -> None:
    # A representative mid-volume series: median total demand over the window.
    vol = test.groupby(["store_id", "product_id"])["demand"].sum().sort_values()
    store, prod = vol.index[int(len(vol) * 0.75)]
    g = test[(test["store_id"] == store) & (test["product_id"] == prod)].sort_values("date")
    g = g.tail(56)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(g["date"], g[q_col(0.1)], g[q_col(0.9)], alpha=0.2, label="10–90% band")
    ax.fill_between(g["date"], g[q_col(0.3)], g[q_col(0.7)], alpha=0.3, label="30–70% band")
    ax.plot(g["date"], g[q_col(0.5)], lw=1.5, label="median forecast")
    ax.plot(g["date"], g["demand"], "k.", ms=5, label="actual demand")
    ax.set_title(f"Day-ahead demand forecast — store {store}, product {prod}")
    ax.set_ylabel("units/day")
    ax.legend(loc="upper left", frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outdir / "fan_chart.png", dpi=150)
    plt.close(fig)


def frontier_chart(sweep_df: pd.DataFrame, named: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(sweep_df["waste_rate"] * 100, sweep_df["fill_rate"] * 100, "-o", ms=4,
            label="quantile policies (service-level sweep)")
    for _, r in sweep_df.iterrows():
        ax.annotate(r["policy"], (r["waste_rate"] * 100, r["fill_rate"] * 100),
                    textcoords="offset points", xytext=(6, -4), fontsize=8)
    markers = {"naive": ("s", "tab:red"), "newsvendor": ("*", "tab:green")}
    for _, r in named.iterrows():
        m, c = markers.get(r["policy"], ("D", "tab:gray"))
        ax.plot(r["waste_rate"] * 100, r["fill_rate"] * 100, m, ms=12, color=c, label=r["policy"])
    ax.set_xlabel("waste (% of delivered units)")
    ax.set_ylabel("availability (fill rate, % of demand served)")
    ax.set_title("The waste/availability trade-off is a dial, not a fact")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "frontier.png", dpi=150)
    plt.close(fig)


def print_explanations(test: pd.DataFrame, catalog: pd.DataFrame, n: int) -> None:
    print("\n=== Sample order explanations (newsvendor policy, empty shelf) ===")
    qs = np.asarray(QUANTILES)
    cat = catalog.set_index("product_id")
    sample = test.sample(n, random_state=0)
    for _, row in sample.iterrows():
        prod = cat.loc[row["product_id"]]
        crit = critical_ratio(float(prod["price"]), float(prod["cost"]))
        values = np.array([row[q_col(q)] for q in QUANTILES], dtype=float)
        target = quantile_from_forecast(qs, values, crit)
        order = order_quantity(target, 0.0, int(prod["pack_size"]))
        ctx = OrderContext(
            store_id=int(row["store_id"]),
            product_id=int(row["product_id"]),
            category=str(prod["category"]),
            date=str(pd.Timestamp(row["date"]).date()),
            q10=float(row[q_col(0.1)]),
            q50=float(row[q_col(0.5)]),
            q90=float(row[q_col(0.9)]),
            target_q=crit,
            target_level=target,
            inventory=0.0,
            pack_size=int(prod["pack_size"]),
            order=order,
            price=float(prod["price"]),
            cost=float(prod["cost"]),
        )
        print("\n- " + explain_order(ctx))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", type=int, default=15)
    ap.add_argument("--products", type=int, default=24)
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--test-days", type=int, default=84)
    ap.add_argument("--explain", type=int, default=3, help="print N sample order explanations")
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    sales, catalog, stores = generate_sales(args.stores, args.products, args.days)
    log(f"generated {len(sales):,} rows ({args.stores} stores x {args.products} products x {args.days} days)", t0)

    feats = build_features(sales)
    train, test = train_test_split_by_date(feats, args.test_days)
    log(f"features built — train {len(train):,} rows, test {len(test):,} rows ({args.test_days} days)", t0)

    gbm = GBMQuantile().fit(train)
    log(f"trained {len(QUANTILES)} quantile GBMs", t0)
    baseline = SeasonalNaiveQuantile().fit(train)

    keys = ["store_id", "product_id", "date"]
    test_gbm = test.merge(gbm.predict_quantiles(test), on=keys)
    test_base = test.merge(baseline.predict_quantiles(test), on=keys)

    metrics = pd.DataFrame(
        [forecast_metrics("gbm_quantile", test_gbm), forecast_metrics("seasonal_naive", test_base)]
    )
    metrics.to_csv(args.outdir / "forecast_metrics.csv", index=False)
    log("forecast metrics (nominal coverage_80 = 0.80):", t0)
    print(metrics.round(3).to_string(index=False))

    fan_chart(test_gbm, args.outdir)
    log("saved fan_chart.png", t0)

    sweep_rows, named_rows = [], []
    for q in SWEEP:
        r = simulate_policy(test_gbm, catalog, f"q={q}", QUANTILES)
        sweep_rows.append(r.__dict__)
        log(f"simulated policy q={q}: fill {r.fill_rate:.1%}, waste {r.waste_rate:.1%}, profit {r.profit:,.0f}", t0)
    for pol in ["naive", "newsvendor"]:
        r = simulate_policy(test_gbm, catalog, pol, QUANTILES)
        named_rows.append(r.__dict__)
        log(f"simulated policy {pol}: fill {r.fill_rate:.1%}, waste {r.waste_rate:.1%}, profit {r.profit:,.0f}", t0)

    sweep_df, named_df = pd.DataFrame(sweep_rows), pd.DataFrame(named_rows)
    pd.concat([sweep_df, named_df]).to_csv(args.outdir / "policies.csv", index=False)
    frontier_chart(sweep_df, named_df, args.outdir)
    log("saved frontier.png and policies.csv", t0)

    if args.explain > 0:
        print_explanations(test_gbm, catalog, args.explain)

    best = pd.concat([sweep_df, named_df]).sort_values("profit", ascending=False).iloc[0]
    log(
        f"done. most profitable policy: {best['policy']} "
        f"(fill {best['fill_rate']:.1%}, waste {best['waste_rate']:.1%})",
        t0,
    )


if __name__ == "__main__":
    main()
