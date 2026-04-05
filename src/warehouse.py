"""
Data Warehouse Module
=====================
Loads the raw CSV into a SQLite star-schema data warehouse.

Star Schema:
  Fact table  : fact_sales_line   (one row per invoice line)
  Dimensions  : dim_date, dim_customer, dim_product
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

from .config import DATA_CSV, DB_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_engine():
    return create_engine(f"sqlite:///{DB_PATH}", future=True)


def _load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, encoding="ISO-8859-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Rename to consistent names
    df.rename(columns={
        "InvoiceNo": "invoice_no",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "UnitPrice": "unit_price",
        "CustomerID": "customer_id",
        "Country": "country",
    }, inplace=True)

    # Drop returns (negative qty) and bad rows
    df = df[df["quantity"] > 0]
    df = df[df["unit_price"] > 0]
    df.dropna(subset=["customer_id", "description"], inplace=True)

    df["customer_id"] = df["customer_id"].astype(int).astype(str)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["line_revenue"] = df["quantity"] * df["unit_price"]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS dim_date (
    date_key    INTEGER PRIMARY KEY,
    full_date   TEXT,
    year        INTEGER,
    quarter     INTEGER,
    month       INTEGER,
    month_name  TEXT,
    week        INTEGER,
    day_of_week INTEGER,
    day_name    TEXT
);

CREATE TABLE IF NOT EXISTS dim_customer (
    customer_key INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id  TEXT UNIQUE,
    country      TEXT
);

CREATE TABLE IF NOT EXISTS dim_product (
    product_key  INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code   TEXT UNIQUE,
    description  TEXT
);

CREATE TABLE IF NOT EXISTS fact_sales_line (
    fact_key        INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_no      TEXT,
    date_key        INTEGER REFERENCES dim_date(date_key),
    customer_key    INTEGER REFERENCES dim_customer(customer_key),
    product_key     INTEGER REFERENCES dim_product(product_key),
    quantity        INTEGER,
    unit_price      REAL,
    line_revenue    REAL
);

CREATE TABLE IF NOT EXISTS mining_customer_segment (
    customer_id     TEXT PRIMARY KEY,
    recency_days    INTEGER,
    frequency       INTEGER,
    monetary        REAL,
    rfm_score       INTEGER,
    segment_label   TEXT,
    cluster         INTEGER
);
"""


# ---------------------------------------------------------------------------
# Dimension builders
# ---------------------------------------------------------------------------

def _build_dim_date(df: pd.DataFrame, conn) -> pd.DataFrame:
    dates = df["invoice_date"].dt.normalize().unique()
    rows = []
    for d in dates:
        ts = pd.Timestamp(d)
        rows.append({
            "date_key": int(ts.strftime("%Y%m%d")),
            "full_date": str(ts.date()),
            "year": ts.year,
            "quarter": ts.quarter,
            "month": ts.month,
            "month_name": ts.strftime("%B"),
            "week": ts.isocalendar()[1],
            "day_of_week": ts.dayofweek,
            "day_name": ts.strftime("%A"),
        })
    dim = pd.DataFrame(rows).drop_duplicates("date_key")
    dim.to_sql("dim_date", conn, if_exists="replace", index=False)
    return dim


def _build_dim_customer(df: pd.DataFrame, conn) -> pd.DataFrame:
    dim = (
        df[["customer_id", "country"]]
        .drop_duplicates("customer_id")
        .reset_index(drop=True)
    )
    dim.index.name = "customer_key"
    dim = dim.reset_index()
    dim["customer_key"] += 1          # 1-based keys
    dim.to_sql("dim_customer", conn, if_exists="replace", index=False)
    return dim


def _build_dim_product(df: pd.DataFrame, conn) -> pd.DataFrame:
    dim = (
        df[["stock_code", "description"]]
        .drop_duplicates("stock_code")
        .reset_index(drop=True)
    )
    dim.index.name = "product_key"
    dim = dim.reset_index()
    dim["product_key"] += 1
    dim.to_sql("dim_product", conn, if_exists="replace", index=False)
    return dim


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_csv_to_warehouse() -> dict:
    """ETL: CSV → SQLite star schema. Returns summary stats dict."""
    df = _load_raw()
    engine = _get_engine()

    with engine.begin() as conn:
        # Create schema
        for stmt in DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))

        # Dimensions
        dim_date = _build_dim_date(df, conn)
        dim_customer = _build_dim_customer(df, conn)
        dim_product = _build_dim_product(df, conn)

        # Build date_key column
        df["date_key"] = df["invoice_date"].dt.normalize().apply(
            lambda t: int(pd.Timestamp(t).strftime("%Y%m%d"))
        )

        # Merge surrogate keys
        df = df.merge(dim_customer[["customer_id", "customer_key"]], on="customer_id", how="left")
        df = df.merge(dim_product[["stock_code", "product_key"]], on="stock_code", how="left")

        # Fact table
        fact = df[["invoice_no", "date_key", "customer_key", "product_key",
                   "quantity", "unit_price", "line_revenue"]].copy()
        fact.to_sql("fact_sales_line", conn, if_exists="replace", index=False)

    return {
        "total_rows_loaded": len(fact),
        "unique_customers": int(dim_customer["customer_id"].nunique()),
        "unique_products": int(dim_product["stock_code"].nunique()),
        "unique_dates": len(dim_date),
        "total_revenue": round(float(df["line_revenue"].sum()), 2),
        "date_range": f"{df['invoice_date'].min().date()} → {df['invoice_date'].max().date()}",
    }
