"""
Charts / Analytics Module
=========================
Reads from the SQLite warehouse + mining tables and exports PNG charts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine, text

from ..config import DB_PATH, OUTPUT_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = "Set2"
sns.set_theme(style="whitegrid", palette=PALETTE)


def _engine():
    return create_engine(f"sqlite:///{DB_PATH}", future=True)


def _save(fig: plt.Figure, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Chart 1 – Revenue by Country (Top 10)
# ---------------------------------------------------------------------------
def chart_revenue_by_country() -> Path:
    q = text("""
        SELECT dc.country, SUM(f.line_revenue) AS revenue
        FROM fact_sales_line f
        JOIN dim_customer dc ON dc.customer_key = f.customer_key
        GROUP BY dc.country ORDER BY revenue DESC LIMIT 10
    """)
    with _engine().connect() as conn:
        df = pd.read_sql(q, conn)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="country", y="revenue", palette="Blues_d", ax=ax)
    ax.set_title("Top 10 Countries by Revenue", fontsize=14, fontweight="bold")
    ax.set_xlabel("Country")
    ax.set_ylabel("Revenue (£)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v/1e3:.0f}K"))
    plt.xticks(rotation=30, ha="right")
    return _save(fig, "01_revenue_by_country.png")


# ---------------------------------------------------------------------------
# Chart 2 – Monthly Revenue Trend
# ---------------------------------------------------------------------------
def chart_monthly_revenue() -> Path:
    q = text("""
        SELECT dd.year, dd.month, dd.month_name,
               SUM(f.line_revenue) AS revenue
        FROM fact_sales_line f
        JOIN dim_date dd ON dd.date_key = f.date_key
        GROUP BY dd.year, dd.month, dd.month_name
        ORDER BY dd.year, dd.month
    """)
    with _engine().connect() as conn:
        df = pd.read_sql(q, conn)

    df["period"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["period"], df["revenue"], marker="o", linewidth=2, color="#2196F3")
    ax.fill_between(range(len(df["period"])), df["revenue"], alpha=0.15, color="#2196F3")
    ax.set_xticks(range(len(df["period"])))
    ax.set_xticklabels(df["period"], rotation=45, ha="right")
    ax.set_title("Monthly Revenue Trend", fontsize=14, fontweight="bold")
    ax.set_ylabel("Revenue (£)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v/1e3:.0f}K"))
    return _save(fig, "02_monthly_revenue.png")


# ---------------------------------------------------------------------------
# Chart 3 – Customer Segments (RFM / K-Means)
# ---------------------------------------------------------------------------
def chart_customer_segments() -> Path:
    q = text("""
        SELECT segment_label,
               COUNT(*) AS customers,
               AVG(monetary) AS avg_revenue,
               AVG(frequency) AS avg_frequency,
               AVG(recency_days) AS avg_recency
        FROM mining_customer_segment
        GROUP BY segment_label
        ORDER BY avg_revenue DESC
    """)
    with _engine().connect() as conn:
        df = pd.read_sql(q, conn)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Customer Segments (K-Means RFM)", fontsize=14, fontweight="bold")

    colors = sns.color_palette(PALETTE, len(df))

    # Pie: customer count
    axes[0].pie(df["customers"], labels=df["segment_label"], autopct="%1.1f%%",
                colors=colors, startangle=140)
    axes[0].set_title("Customer Distribution")

    # Bar: avg revenue
    sns.barplot(data=df, x="segment_label", y="avg_revenue", palette=PALETTE, ax=axes[1])
    axes[1].set_title("Avg Revenue per Customer")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Avg Revenue (£)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:.0f}"))
    plt.setp(axes[1].get_xticklabels(), rotation=15, ha="right")

    # Bar: avg frequency
    sns.barplot(data=df, x="segment_label", y="avg_frequency", palette=PALETTE, ax=axes[2])
    axes[2].set_title("Avg Purchase Frequency")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Avg Transactions")
    plt.setp(axes[2].get_xticklabels(), rotation=15, ha="right")

    plt.tight_layout()
    return _save(fig, "03_customer_segments.png")


# ---------------------------------------------------------------------------
# Chart 4 – Top 15 Products by Revenue
# ---------------------------------------------------------------------------
def chart_top_products() -> Path:
    q = text("""
        SELECT p.description, SUM(f.line_revenue) AS revenue
        FROM fact_sales_line f
        JOIN dim_product p ON p.product_key = f.product_key
        GROUP BY p.description ORDER BY revenue DESC LIMIT 15
    """)
    with _engine().connect() as conn:
        df = pd.read_sql(q, conn)

    df["description"] = df["description"].str.slice(0, 35)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=df, x="revenue", y="description", palette="rocket_r", ax=ax)
    ax.set_title("Top 15 Products by Revenue", fontsize=14, fontweight="bold")
    ax.set_xlabel("Revenue (£)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v/1e3:.0f}K"))
    return _save(fig, "04_top_products.png")


# ---------------------------------------------------------------------------
# Chart 5 – Revenue Forecast (Actual + Predicted)
# ---------------------------------------------------------------------------
def chart_revenue_forecast() -> Path:
    with _engine().connect() as conn:
        actual = pd.read_sql(text("SELECT * FROM mining_daily_revenue"), conn)
        forecast = pd.read_sql(text("SELECT * FROM mining_revenue_forecast"), conn)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual["full_date"], actual["daily_revenue"], label="Actual", color="#1565C0", linewidth=1.5)
    ax.plot(forecast["full_date"], forecast["forecasted_revenue"],
            label="Forecast (30 days)", color="#E53935", linewidth=2, linestyle="--")

    # Show only every N-th tick to avoid clutter
    n = max(1, len(actual) // 15)
    xticks = list(actual["full_date"].iloc[::n]) + list(forecast["full_date"][-1:])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=45, ha="right")

    ax.set_title("Daily Revenue: Actual vs Forecast (Linear Regression)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Revenue (£)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v/1e3:.1f}K"))
    ax.legend()
    ax.axvline(x=actual["full_date"].iloc[-1], color="gray", linestyle=":", linewidth=1)
    return _save(fig, "05_revenue_forecast.png")


# ---------------------------------------------------------------------------
# Chart 6 – Day-of-Week Revenue Heatmap
# ---------------------------------------------------------------------------
def chart_dow_heatmap() -> Path:
    q = text("""
        SELECT dd.day_name, dd.month_name, dd.month,
               SUM(f.line_revenue) AS revenue
        FROM fact_sales_line f
        JOIN dim_date dd ON dd.date_key = f.date_key
        GROUP BY dd.day_name, dd.month_name, dd.month
    """)
    with _engine().connect() as conn:
        df = pd.read_sql(q, conn)

    pivot = df.pivot_table(index="day_name", columns="month_name", values="revenue", aggfunc="sum")
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex(index=[d for d in day_order if d in pivot.index],
                          columns=[m for m in month_order if m in pivot.columns])

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot / 1e3, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, linewidths=0.5,
                cbar_kws={"label": "Revenue (£K)"})
    ax.set_title("Revenue Heatmap: Day of Week × Month", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Day of Week")
    return _save(fig, "06_dow_heatmap.png")


# ---------------------------------------------------------------------------
# Chart 7 – RFM Scatter: Recency vs Monetary coloured by Segment
# ---------------------------------------------------------------------------
def chart_rfm_scatter() -> Path:
    q = text("SELECT recency_days, monetary, frequency, segment_label FROM mining_customer_segment")
    with _engine().connect() as conn:
        df = pd.read_sql(q, conn)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, grp in df.groupby("segment_label"):
        ax.scatter(grp["recency_days"], grp["monetary"], label=label,
                   s=grp["frequency"].clip(upper=100) * 2, alpha=0.55, edgecolors="none")
    ax.set_xlabel("Recency (days since last purchase)")
    ax.set_ylabel("Total Revenue (£)")
    ax.set_title("RFM Scatter: Recency vs Monetary Value", fontsize=14, fontweight="bold")
    ax.legend(title="Segment")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v/1e3:.0f}K"))
    return _save(fig, "07_rfm_scatter.png")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_charts() -> list[Path]:
    paths = []
    for fn in [
        chart_revenue_by_country,
        chart_monthly_revenue,
        chart_customer_segments,
        chart_top_products,
        chart_revenue_forecast,
        chart_dow_heatmap,
        chart_rfm_scatter,
    ]:
        try:
            p = fn()
            paths.append(p)
        except Exception as e:
            print(f"  [WARN] {fn.__name__} failed: {e}")
    return paths
