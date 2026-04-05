"""
Data Mining Module
==================
Three mining techniques applied to the sales data warehouse:

1. RFM Segmentation   → KMeans clustering on Recency/Frequency/Monetary
2. Association Rules  → Apriori algorithm (market basket analysis)
3. Revenue Forecast   → Linear regression on daily revenue totals
"""
from __future__ import annotations

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

from .config import DB_PATH

warnings.filterwarnings("ignore")


def _engine():
    return create_engine(f"sqlite:///{DB_PATH}", future=True)


# ---------------------------------------------------------------------------
# 1. RFM Segmentation + K-Means
# ---------------------------------------------------------------------------

def run_rfm_segmentation(n_clusters: int = 4) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) scores per customer,
    cluster with K-Means, and persist results to mining_customer_segment.
    Returns the RFM DataFrame with segment labels.
    """
    eng = _engine()
    q = text("""
        SELECT dc.customer_id,
               dd.full_date,
               f.line_revenue
        FROM fact_sales_line f
        JOIN dim_customer dc ON dc.customer_key = f.customer_key
        JOIN dim_date     dd ON dd.date_key      = f.date_key
    """)
    with eng.connect() as conn:
        df = pd.read_sql(q, conn)

    df["full_date"] = pd.to_datetime(df["full_date"])
    snapshot_date = df["full_date"].max() + timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency_days=("full_date", lambda x: (snapshot_date - x.max()).days),
        frequency=("full_date", "count"),
        monetary=("line_revenue", "sum"),
    ).reset_index()

    # Score each RFM metric 1-4 using quartiles
    rfm["r_score"] = pd.qcut(rfm["recency_days"], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"], 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    # K-Means clustering on scaled RFM
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["recency_days", "frequency", "monetary"]])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    rfm["cluster"] = km.fit_predict(X)

    # Label clusters by average monetary value (highest = Champions)
    cluster_monetary = rfm.groupby("cluster")["monetary"].mean().sort_values(ascending=False)
    labels = ["Champions", "Loyal Customers", "At Risk", "Lost"]
    label_map = {cluster: labels[i] for i, cluster in enumerate(cluster_monetary.index)}
    rfm["segment_label"] = rfm["cluster"].map(label_map)

    # Persist to DB
    out = rfm[["customer_id", "recency_days", "frequency", "monetary",
               "rfm_score", "segment_label", "cluster"]]
    with eng.begin() as conn:
        out.to_sql("mining_customer_segment", conn, if_exists="replace", index=False)

    return rfm


# ---------------------------------------------------------------------------
# 2. Association Rules (Apriori / Market Basket)
# ---------------------------------------------------------------------------

def run_association_rules(
    min_support: float = 0.02,
    min_confidence: float = 0.3,
    max_basket_size: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Apriori on invoice-level baskets (UK only for tractable size).
    Returns (frequent_itemsets, rules) DataFrames.
    """
    eng = _engine()
    q = text("""
        SELECT f.invoice_no, p.description
        FROM fact_sales_line f
        JOIN dim_product  p  ON p.product_key  = f.product_key
        JOIN dim_customer dc ON dc.customer_key = f.customer_key
        WHERE dc.country = 'United Kingdom'
    """)
    with eng.connect() as conn:
        df = pd.read_sql(q, conn)

    # Keep top products to keep the basket matrix manageable
    top_products = (
        df["description"].value_counts().head(50).index.tolist()
    )
    df = df[df["description"].isin(top_products)]

    # One-hot encode baskets
    basket = (
        df.groupby(["invoice_no", "description"])["description"]
        .count()
        .unstack(fill_value=0)
        .clip(upper=1)
        .astype(bool)
    )

    if len(basket) > max_basket_size:
        basket = basket.sample(max_basket_size, random_state=42)

    try:
        freq_items = apriori(basket, min_support=min_support, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    except Exception:
        freq_items = pd.DataFrame()
        rules = pd.DataFrame()

    return freq_items, rules


# ---------------------------------------------------------------------------
# 3. Revenue Forecast (Linear Regression)
# ---------------------------------------------------------------------------

def run_revenue_forecast(forecast_days: int = 30) -> dict:
    """
    Aggregate daily revenue and fit a LinearRegression model.
    Returns a dict with trend info and 30-day forecast.
    """
    eng = _engine()
    q = text("""
        SELECT dd.full_date, SUM(f.line_revenue) AS daily_revenue
        FROM fact_sales_line f
        JOIN dim_date dd ON dd.date_key = f.date_key
        GROUP BY dd.full_date
        ORDER BY dd.full_date
    """)
    with eng.connect() as conn:
        df = pd.read_sql(q, conn)

    df["full_date"] = pd.to_datetime(df["full_date"])
    df = df.sort_values("full_date").reset_index(drop=True)
    df["day_index"] = np.arange(len(df))

    X = df[["day_index"]].values
    y = df["daily_revenue"].values

    model = LinearRegression()
    model.fit(X, y)

    # In-sample R²
    r2 = model.score(X, y)

    # Forecast next N days
    last_idx = df["day_index"].max()
    last_date = df["full_date"].max()
    future_idx = np.arange(last_idx + 1, last_idx + 1 + forecast_days).reshape(-1, 1)
    future_dates = [str((last_date + timedelta(days=i)).date()) for i in range(1, forecast_days + 1)]
    future_revenue = model.predict(future_idx)

    # Save daily actuals + forecast for chart use
    df_out = df[["full_date", "daily_revenue"]].copy()
    df_out["full_date"] = df_out["full_date"].astype(str)
    df_out.to_sql("mining_daily_revenue", _engine(), if_exists="replace", index=False)

    forecast_df = pd.DataFrame({"full_date": future_dates, "forecasted_revenue": future_revenue.round(2)})
    forecast_df.to_sql("mining_revenue_forecast", _engine(), if_exists="replace", index=False)

    return {
        "model": "LinearRegression",
        "r2_score": round(float(r2), 4),
        "slope_per_day": round(float(model.coef_[0]), 2),
        "intercept": round(float(model.intercept_), 2),
        "forecast_start": future_dates[0],
        "forecast_end": future_dates[-1],
        "avg_forecasted_daily_revenue": round(float(future_revenue.mean()), 2),
    }
