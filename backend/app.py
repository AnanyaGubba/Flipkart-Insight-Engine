"""FastAPI backend - full analytics REST API over the SQLite warehouse."""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("FLIPKART_DW_DB", ROOT / "flipkart_dw.db"))


def get_engine():
    return create_engine(f"sqlite:///{DB_PATH}", future=True)


app = FastAPI(title="Flipkart Sales DW API", version="2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
def health():
    return {"ok": True, "db": str(DB_PATH), "db_exists": DB_PATH.is_file()}


@app.get("/summary")
def summary():
    eng = get_engine()
    with eng.connect() as conn:
        lines  = conn.execute(text("SELECT COUNT(*) FROM fact_sales_line")).scalar_one()
        cust   = conn.execute(text("SELECT COUNT(*) FROM dim_customer")).scalar_one()
        prod   = conn.execute(text("SELECT COUNT(*) FROM dim_product")).scalar_one()
        rev    = conn.execute(text("SELECT SUM(line_revenue) FROM fact_sales_line")).scalar_one()
        d_min  = conn.execute(text("SELECT MIN(full_date) FROM dim_date")).scalar_one()
        d_max  = conn.execute(text("SELECT MAX(full_date) FROM dim_date")).scalar_one()
    return {
        "fact_lines": lines,
        "customers": cust,
        "products": prod,
        "total_revenue": round(float(rev or 0), 2),
        "date_range": f"{d_min} to {d_max}",
    }


@app.get("/top-products")
def top_products(limit: int = 10):
    q = text("""
        SELECT p.stock_code, p.description, SUM(f.line_revenue) AS revenue,
               SUM(f.quantity) AS units_sold
        FROM fact_sales_line f
        JOIN dim_product p ON p.product_key = f.product_key
        GROUP BY p.stock_code, p.description
        ORDER BY revenue DESC LIMIT :lim
    """)
    with get_engine().connect() as conn:
        rows = conn.execute(q, {"lim": limit}).mappings().all()
    return {"items": [dict(r) for r in rows]}


@app.get("/segments")
def segments():
    q = text("""
        SELECT segment_label,
               COUNT(*) AS customers,
               AVG(recency_days) AS avg_recency,
               AVG(frequency) AS avg_frequency,
               AVG(monetary) AS avg_monetary,
               SUM(monetary) AS total_monetary
        FROM mining_customer_segment
        GROUP BY segment_label
        ORDER BY total_monetary DESC
    """)
    with get_engine().connect() as conn:
        rows = conn.execute(q).mappings().all()
    return {"segments": [dict(r) for r in rows]}


@app.get("/revenue-by-country")
def revenue_by_country(limit: int = 20):
    q = text("""
        SELECT dc.country, SUM(f.line_revenue) AS revenue, COUNT(DISTINCT f.invoice_no) AS orders
        FROM fact_sales_line f
        JOIN dim_customer dc ON dc.customer_key = f.customer_key
        GROUP BY dc.country ORDER BY revenue DESC LIMIT :lim
    """)
    with get_engine().connect() as conn:
        rows = conn.execute(q, {"lim": limit}).mappings().all()
    return {"countries": [dict(r) for r in rows]}


@app.get("/monthly-revenue")
def monthly_revenue():
    q = text("""
        SELECT dd.year, dd.month, dd.month_name,
               SUM(f.line_revenue) AS revenue,
               COUNT(DISTINCT f.invoice_no) AS orders
        FROM fact_sales_line f
        JOIN dim_date dd ON dd.date_key = f.date_key
        GROUP BY dd.year, dd.month, dd.month_name
        ORDER BY dd.year, dd.month
    """)
    with get_engine().connect() as conn:
        rows = conn.execute(q).mappings().all()
    return {"monthly": [dict(r) for r in rows]}


@app.get("/forecast")
def forecast():
    try:
        with get_engine().connect() as conn:
            actual = pd.read_sql(text("SELECT * FROM mining_daily_revenue"), conn).to_dict("records")
            fc     = pd.read_sql(text("SELECT * FROM mining_revenue_forecast"), conn).to_dict("records")
        return {"actual": actual, "forecast": fc}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Run main.py first: {e}")


@app.get("/customer/{customer_id}")
def customer_detail(customer_id: str):
    q = text("""
        SELECT dc.customer_id, dc.country,
               mcs.recency_days, mcs.frequency, mcs.monetary, mcs.segment_label
        FROM dim_customer dc
        LEFT JOIN mining_customer_segment mcs ON mcs.customer_id = dc.customer_id
        WHERE dc.customer_id = :cid
    """)
    with get_engine().connect() as conn:
        row = conn.execute(q, {"cid": customer_id}).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return dict(row)


@app.get("/rules")
def association_rules_endpoint():
    """Return top association rules stored after running main.py mining step."""
    import json
    from pathlib import Path
    rules_path = ROOT / "output" / "association_rules.json"
    if not rules_path.exists():
        raise HTTPException(status_code=404, detail="Run main.py first to generate rules")
    with open(rules_path) as f:
        return {"rules": json.load(f)}
