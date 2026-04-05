# Flipkart Insight Engine: Sales Analytics & Forecasting
## Data Warehouse + Data Mining in Python

A fully functional sales analytics project using a **SQLite star-schema data warehouse**,
three **data mining algorithms**, a **FastAPI REST backend**, and **7 exported charts**.

---

## Project Structure

```
miniproj_dmw/
├── main.py                  ← Run this first (full pipeline)
├── requirements.txt         ← pip dependencies
├── flipkart_dw.db           ← SQLite warehouse (auto-generated on first run)
├── data/
│   └── data.csv             ← Raw sales CSV (place your file here)
├── output/                  ← 7 PNG charts are saved here
├── backend/
│   └── app.py               ← FastAPI REST API
└── src/
    ├── config.py            ← File paths (edit if you move data.csv)
    ├── warehouse.py         ← ETL: CSV → SQLite star schema
    ├── mining.py            ← Data mining algorithms
    └── analytics/
        └── charts.py        ← Chart generation (matplotlib + seaborn)
```

---

## How to Run

### Step 1 – Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 – Place your dataset

Copy `data.csv` into the `data/` folder. The file should have these columns:
```
InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
```

### Step 3 – Run the full pipeline

```bash
python main.py
```

This will:
1. **Load CSV → SQLite** warehouse (star schema with 3 dimension tables + 1 fact table)
2. **RFM Segmentation** – segments customers into Champions / Loyal / At Risk / Lost using K-Means
3. **Association Rules** – finds products frequently bought together using the Apriori algorithm
4. **Revenue Forecast** – predicts next 30 days using Linear Regression
5. **Export 7 PNG charts** to the `output/` folder

### Step 4 – Start the REST API (optional)

```bash
uvicorn backend.app:app --reload
```

Then open: [http://localhost:8000/docs](http://localhost:8000/docs)

Available API endpoints:
| Endpoint | Description |
|---|---|
| `GET /health` | Check if DB exists |
| `GET /summary` | Total rows, customers, revenue |
| `GET /top-products?limit=10` | Top N products by revenue |
| `GET /segments` | Customer segment stats |
| `GET /revenue-by-country?limit=20` | Revenue per country |
| `GET /monthly-revenue` | Month-by-month revenue |
| `GET /forecast` | Actual + forecasted daily revenue |
| `GET /customer/{id}` | Single customer detail |

---

## What Each Chart Shows

| File | Chart |
|---|---|
| `01_revenue_by_country.png` | Bar chart: Top 10 countries by total revenue |
| `02_monthly_revenue.png` | Line chart: Monthly revenue trend |
| `03_customer_segments.png` | Pie + bar charts: K-Means customer segments |
| `04_top_products.png` | Horizontal bar: Top 15 products by revenue |
| `05_revenue_forecast.png` | Line chart: Actual vs 30-day forecast |
| `06_dow_heatmap.png` | Heatmap: Revenue by day-of-week × month |
| `07_rfm_scatter.png` | Scatter: Recency vs Monetary, coloured by segment |

---

## How to Edit / Customize

### Change the dataset
Edit **`src/config.py`**:
```python
DATA_CSV = PROJECT_ROOT / "data" / "your_new_file.csv"
```
Column names are mapped in `src/warehouse.py` → `_load_raw()` (the `rename` dict).

### Change number of customer segments
Edit **`src/mining.py`** → `run_rfm_segmentation()`:
```python
def run_rfm_segmentation(n_clusters: int = 4):   # ← change 4 to 3, 5, 6…
```

### Change association rules sensitivity
Edit **`src/mining.py`** → `run_association_rules()`:
```python
def run_association_rules(
    min_support: float = 0.02,     # ← lower = more rules, slower
    min_confidence: float = 0.3,   # ← lower = weaker rules allowed
```

### Change forecast horizon
Edit **`src/mining.py`** → `run_revenue_forecast()`:
```python
def run_revenue_forecast(forecast_days: int = 30):  # ← change to 60, 90…
```

### Add a new chart
Add a function to **`src/analytics/charts.py`** following this pattern:
```python
def chart_my_new_chart() -> Path:
    with _engine().connect() as conn:
        df = pd.read_sql(text("SELECT ..."), conn)
    fig, ax = plt.subplots(figsize=(10, 5))
    # ... matplotlib/seaborn code ...
    return _save(fig, "08_my_chart.png")
```
Then add it to the `export_charts()` list at the bottom.

### Add a new API endpoint
Add a function to **`backend/app.py`**:
```python
@app.get("/my-endpoint")
def my_endpoint():
    with get_engine().connect() as conn:
        rows = conn.execute(text("SELECT ...")).mappings().all()
    return {"data": [dict(r) for r in rows]}
```

---

## Database Schema

```
dim_date          dim_customer        dim_product
─────────         ────────────        ───────────
date_key PK       customer_key PK     product_key PK
full_date         customer_id         stock_code
year              country             description
quarter
month                    ┌─────────────────────┐
month_name               │    fact_sales_line   │
week                     │─────────────────────│
day_of_week              │  fact_key PK         │
day_name                 │  invoice_no          │
                         │  date_key FK         │
                         │  customer_key FK     │
                         │  product_key FK      │
                         │  quantity            │
                         │  unit_price          │
                         │  line_revenue        │
                         └─────────────────────┘

mining_customer_segment   (populated by main.py step 2a)
───────────────────────
customer_id PK
recency_days
frequency
monetary
rfm_score
segment_label
cluster
```

---

## Technologies Used

| Layer | Technology |
|---|---|
| Data Storage | SQLite (via SQLAlchemy) |
| Data Warehouse | Star Schema (Fact + 3 Dimensions) |
| ETL Pipeline | Python / Pandas |
| Segmentation | K-Means Clustering (scikit-learn) |
| Market Basket | Apriori Algorithm (mlxtend) |
| Forecasting | Linear Regression (scikit-learn) |
| Visualization | Matplotlib + Seaborn |
| REST API | FastAPI + Uvicorn |
