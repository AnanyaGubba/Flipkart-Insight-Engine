"""
Flipkart Sales Analysis – Main Entry Point
==========================================
Runs the full pipeline:
  1. ETL     : Load CSV → SQLite star-schema data warehouse
  2. Mining  : RFM Segmentation, Association Rules, Revenue Forecast
  3. Charts  : Export 7 PNG charts to output/

Run:
    python main.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.charts import export_charts
from src.config import OUTPUT_DIR
from src.mining import run_association_rules, run_revenue_forecast, run_rfm_segmentation
from src.warehouse import load_csv_to_warehouse


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


def main() -> None:
    separator("STEP 1: Loading CSV -> SQLite Data Warehouse (Star Schema)")
    stats = load_csv_to_warehouse()
    print(json.dumps(stats, indent=2))

    separator("STEP 2a: Data Mining - RFM Segmentation + K-Means Clustering")
    rfm = run_rfm_segmentation()
    summary = (
        rfm.groupby("segment_label")
        .agg(customers=("customer_id", "count"), total_revenue=("monetary", "sum"))
        .sort_values("total_revenue", ascending=False)
    )
    print(summary.to_string())

    separator("STEP 2b: Data Mining - Market Basket / Association Rules (Apriori)")
    freq, rules = run_association_rules()
    print(f"Frequent itemsets : {len(freq)}")
    print(f"Association rules : {len(rules)}")
    if not rules.empty:
        cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        print("\nTop 10 rules by lift:")
        print(rules.head(10)[cols].to_string(index=False))

    separator("STEP 2c: Data Mining - Revenue Forecast (Linear Regression)")
    fc = run_revenue_forecast()
    print(json.dumps(fc, indent=2))

    separator("STEP 3: Exporting Charts")
    paths = export_charts()
    for p in paths:
        print(f"  OK {p.name}")

    separator("ALL DONE")
    print(f"Charts saved to: {OUTPUT_DIR}")
    print("To start the REST API: uvicorn backend.app:app --reload")


if __name__ == "__main__":
    main()
