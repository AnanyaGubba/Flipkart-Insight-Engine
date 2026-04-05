"""Paths and constants for the sales warehouse project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "data.csv"
DB_PATH = PROJECT_ROOT / "flipkart_dw.db"
OUTPUT_DIR = PROJECT_ROOT / "output"
