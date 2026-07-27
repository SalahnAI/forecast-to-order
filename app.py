"""Store-manager view of the forecast-to-order pipeline.

One screen, three ideas:
- the service-level dial: pick where the chain wants to sit on the
  waste/availability frontier and watch the network KPIs move (every point is
  a full replay simulation, not an interpolation);
- today's proposed orders for one store, with the exact decision inputs;
- a plain-language explanation per order line — the thing store staff accept
  or override.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from forecast_to_order.data import generate_sales
from forecast_to_order.explain import OrderContext, explain_order
from forecast_to_order.features import build_features, train_test_split_by_date
from forecast_to_order.models import QUANTILES, GBMQuantile, q_col
from forecast_to_order.simulate import simulate_policy

SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

st.set_page_config(page_title="forecast-to-order", page_icon="🥬", layout="wide")


@st.cache_data(show_spinner="Generating two years of data and training 11 quantile models (~30 s, first load only)…")
def load_world():
    sales, catalog, _ = generate_sales()
    feats = build_features(sales)
    train, test = train_test_split_by_date(feats)
    gbm = GBMQuantile().fit(train)
    test_pred = test.merge(
        gbm.predict_quantiles(test), on=["store_id", "product_id", "date"]
    )
    return test_pred, catalog


@st.cache_data(show_spinner=False)
def run_policy(policy: str):
    test_pred, catalog = load_world()
    return simulate_policy(test_pred, catalog, policy, QUANTILES, collect_orders=True)


test_pred, catalog = load_world()

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Ordering policy")
mode = st.sidebar.radio(
    "Target service level",
    ["Network dial (fixed quantile)", "Newsvendor critical ratio (per product)"],
    help="The order for each line is the chosen quantile of its demand "
    "distribution, minus stock, rounded up to whole packs.",
)
if mode.startswith("Network"):
    q = st.sidebar.slider(
        "Demand quantile to cover", 0.30, 0.95, 0.60, 0.05,
        help="Higher = fewer empty shelves, more waste. This is the business dial.",
    )
    policy = f"q={round(q, 2)}"
else:
    policy = "newsvendor"
    st.sidebar.caption(
        "Each product is ordered at its own profit-maximizing quantile "
        "q\\* = margin / price (see MATH.md)."
    )

result = run_policy(policy)
naive = run_policy("naive")
orders = result.orders

st.sidebar.divider()
st.sidebar.header("Store view")
store = st.sidebar.selectbox("Store", sorted(orders["store_id"].unique()))
dates = sorted(orders["date"].unique())
date = st.sidebar.selectbox(
    "Delivery day", dates, index=len(dates) - 1, format_func=lambda d: str(pd.Timestamp(d).date())
)

# ---------------------------------------------------------------- header
st.title("forecast-to-order")
st.caption(
    "Probabilistic demand forecasts turned into shelf-ready orders for fresh retail. "
    "Synthetic data, fully reproducible — every KPI below is a day-by-day replay of the "
    "84-day holdout under the selected policy. "
    "[Code & method](https://github.com/SalahnAI/forecast-to-order)"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Policy", policy)
c2.metric(
    "Availability (fill rate)",
    f"{result.fill_rate:.1%}",
    f"{result.fill_rate - naive.fill_rate:+.1%} vs naive reorder",
)
c3.metric(
    "Waste (share of delivered)",
    f"{result.waste_rate:.1%}",
    f"{result.waste_rate - naive.waste_rate:+.1%} vs naive reorder",
    delta_color="inverse",
)
c4.metric(
    "Profit over the holdout",
    f"{result.profit:,.0f}",
    f"{(result.profit / naive.profit - 1):+.1%} vs naive reorder",
)

# ---------------------------------------------------------------- charts
left, right = st.columns(2)

with left:
    st.subheader("Where this policy sits on the frontier")
    sweep = [run_policy(f"q={s}") for s in SWEEP]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.plot(
        [r.waste_rate * 100 for r in sweep],
        [r.fill_rate * 100 for r in sweep],
        "-o", ms=4, color="tab:blue", label="quantile sweep",
    )
    for r, s in zip(sweep, SWEEP):
        ax.annotate(f"q={s}", (r.waste_rate * 100, r.fill_rate * 100),
                    textcoords="offset points", xytext=(6, -4), fontsize=8)
    ax.plot(naive.waste_rate * 100, naive.fill_rate * 100, "s", ms=10,
            color="tab:red", label="naive reorder")
    ax.plot(result.waste_rate * 100, result.fill_rate * 100, "*", ms=18,
            color="tab:green", label=f"current ({policy})")
    ax.set_xlabel("waste (% of delivered units)")
    ax.set_ylabel("availability (% of demand served)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Forecast vs reality for one series")
    prods = sorted(orders.loc[orders["store_id"] == store, "product_id"].unique())
    prod = st.selectbox("Product", prods, key="fanchart_product")
    g = test_pred[(test_pred["store_id"] == store) & (test_pred["product_id"] == prod)]
    g = g.sort_values("date").tail(56)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.fill_between(g["date"], g[q_col(0.1)], g[q_col(0.9)], alpha=0.2, label="10–90%")
    ax.fill_between(g["date"], g[q_col(0.3)], g[q_col(0.7)], alpha=0.3, label="30–70%")
    ax.plot(g["date"], g[q_col(0.5)], lw=1.5, label="median forecast")
    ax.plot(g["date"], g["demand"], "k.", ms=5, label="actual")
    ax.axvline(pd.Timestamp(date), color="tab:green", lw=1, ls="--")
    ax.set_ylabel("units/day")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------- order sheet
st.subheader(f"Proposed orders — store {store}, {pd.Timestamp(date).date()}")

day = orders[(orders["store_id"] == store) & (orders["date"] == date)].merge(
    catalog[["product_id", "category", "price", "cost", "pack_size"]], on="product_id"
)
day = day.sort_values("order", ascending=False).reset_index(drop=True)

sheet = pd.DataFrame(
    {
        "product": day["product_id"],
        "category": day["category"],
        "forecast (median)": day["q50"].round(1),
        "80% range": [f"{a:.0f}–{b:.0f}" for a, b in zip(day["q10"], day["q90"])],
        "target level": day["target_level"].round(1),
        "on shelf": day["inventory"].astype(int),
        "pack": day["pack_size"],
        "ORDER": day["order"].astype(int),
    }
)
st.dataframe(sheet, width="stretch", hide_index=True)

st.markdown("**Why these quantities** — the explanation each line ships with:")
for _, row in day[day["order"] > 0].head(6).iterrows():
    ctx = OrderContext(
        store_id=int(row["store_id"]),
        product_id=int(row["product_id"]),
        category=str(row["category"]),
        date=str(pd.Timestamp(row["date"]).date()),
        q10=float(row["q10"]),
        q50=float(row["q50"]),
        q90=float(row["q90"]),
        target_q=float(row["target_q"]),
        target_level=float(row["target_level"]),
        inventory=float(row["inventory"]),
        pack_size=int(row["pack_size"]),
        order=int(row["order"]),
        price=float(row["price"]),
        cost=float(row["cost"]),
    )
    with st.expander(
        f"Product {ctx.product_id} ({ctx.category}) — order {ctx.order} units"
    ):
        st.write(explain_order(ctx))
