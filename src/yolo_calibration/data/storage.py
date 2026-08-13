"""Parquet-based storage layer, partitioned by year (spec section 14).

Two layers:
  - RAW checkpoint layer (data/raw/...): one file per fetched trading day,
    written as soon as it is fetched so a long historical build can resume
    without re-fetching already-downloaded days (spec section 17).
  - PROCESSED layer (data/processed/<table>/year=YYYY/part.parquet): the
    derived tables (stock_features_daily, stock_outcomes_daily,
    qqq_health_daily, qqq_health_outcomes_daily, market_universe_daily,
    reference_tickers).

Both layers are gitignored; only small curated reports are versioned.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from yolo_calibration.config import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

PROCESSED_TABLES = (
    "stock_features_daily",
    "stock_outcomes_daily",
    "qqq_health_daily",
    "qqq_health_outcomes_daily",
    "market_breadth_daily",
    "market_universe_daily",
    "reference_tickers",
    "raw_manifest",
)


# ---- RAW checkpoint layer: grouped daily bars -----------------------------

def raw_grouped_daily_path(trading_date: date) -> Path:
    return RAW_DIR / "grouped_daily" / f"year={trading_date.year}" / f"date={trading_date.isoformat()}.parquet"


def raw_grouped_daily_exists(trading_date: date) -> bool:
    return raw_grouped_daily_path(trading_date).exists()


def write_raw_grouped_daily(trading_date: date, records: list[dict]) -> Path:
    path = raw_grouped_daily_path(trading_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    df.to_parquet(path, index=False)
    return path


def read_raw_grouped_daily(trading_date: date) -> pd.DataFrame | None:
    path = raw_grouped_daily_path(trading_date)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def list_raw_grouped_daily_dates(start: date, end: date) -> list[date]:
    """Dates within [start, end] that already have a raw checkpoint file."""
    found = []
    for year_dir in RAW_DIR.glob("grouped_daily/year=*"):
        for f in year_dir.glob("date=*.parquet"):
            d = date.fromisoformat(f.stem.split("=", 1)[1])
            if start <= d <= end:
                found.append(d)
    return sorted(found)


# ---- RAW checkpoint layer: grouped daily bars, UNADJUSTED -----------------
#
# Separate from the (split-)adjusted series above. Needed specifically for
# point-in-time-correct market cap (config/market_cap_methodology.yaml):
# adjusted=true back-adjusts a historical close using ALL splits known as of
# the BUILD date, not just splits known as of that historical date — fine
# (scale-invariant) for every ratio-based feature/outcome, but wrong for
# market_cap, which must use the price actually observed on that date.

def raw_grouped_daily_unadjusted_path(trading_date: date) -> Path:
    return RAW_DIR / "grouped_daily_unadjusted" / f"year={trading_date.year}" / f"date={trading_date.isoformat()}.parquet"


def raw_grouped_daily_unadjusted_exists(trading_date: date) -> bool:
    return raw_grouped_daily_unadjusted_path(trading_date).exists()


def write_raw_grouped_daily_unadjusted(trading_date: date, records: list[dict]) -> Path:
    path = raw_grouped_daily_unadjusted_path(trading_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    df.to_parquet(path, index=False)
    return path


def read_raw_grouped_daily_unadjusted(trading_date: date) -> pd.DataFrame | None:
    path = raw_grouped_daily_unadjusted_path(trading_date)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def list_raw_grouped_daily_unadjusted_dates(start: date, end: date) -> list[date]:
    found = []
    for year_dir in RAW_DIR.glob("grouped_daily_unadjusted/year=*"):
        for f in year_dir.glob("date=*.parquet"):
            d = date.fromisoformat(f.stem.split("=", 1)[1])
            if start <= d <= end:
                found.append(d)
    return sorted(found)


# ---- RAW checkpoint layer: QQQ constituents -------------------------------

def raw_qqq_constituents_path(effective_date: date) -> Path:
    return RAW_DIR / "qqq_constituents" / f"year={effective_date.year}" / f"date={effective_date.isoformat()}.parquet"


def write_raw_qqq_constituents(effective_date: date, records: list[dict]) -> Path:
    path = raw_qqq_constituents_path(effective_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    df.to_parquet(path, index=False)
    return path


def read_raw_qqq_constituents(effective_date: date) -> pd.DataFrame | None:
    path = raw_qqq_constituents_path(effective_date)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def raw_qqq_constituents_exists(effective_date: date) -> bool:
    return raw_qqq_constituents_path(effective_date).exists()


def list_raw_qqq_constituents_dates(start: date, end: date) -> list[date]:
    found = []
    for year_dir in RAW_DIR.glob("qqq_constituents/year=*"):
        for f in year_dir.glob("date=*.parquet"):
            d = date.fromisoformat(f.stem.split("=", 1)[1])
            if start <= d <= end:
                found.append(d)
    return sorted(found)


# ---- RAW checkpoint layer: point-in-time reference tickers ---------------

def raw_reference_tickers_path(as_of_date: date) -> Path:
    return RAW_DIR / "reference_tickers" / f"year={as_of_date.year}" / f"date={as_of_date.isoformat()}.parquet"


def write_raw_reference_tickers(as_of_date: date, records: list[dict]) -> Path:
    path = raw_reference_tickers_path(as_of_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    df.to_parquet(path, index=False)
    return path


def read_raw_reference_tickers(as_of_date: date) -> pd.DataFrame | None:
    path = raw_reference_tickers_path(as_of_date)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def raw_reference_tickers_exists(as_of_date: date) -> bool:
    return raw_reference_tickers_path(as_of_date).exists()


def list_raw_reference_tickers_dates(start: date, end: date) -> list[date]:
    found = []
    for year_dir in RAW_DIR.glob("reference_tickers/year=*"):
        for f in year_dir.glob("date=*.parquet"):
            d = date.fromisoformat(f.stem.split("=", 1)[1])
            if start <= d <= end:
                found.append(d)
    return sorted(found)


# ---- PROCESSED layer -------------------------------------------------------

def _validate_table(table: str) -> None:
    if table not in PROCESSED_TABLES:
        raise ValueError(f"Unknown processed table '{table}'. Known: {PROCESSED_TABLES}")


def processed_partition_path(table: str, year: int) -> Path:
    _validate_table(table)
    return PROCESSED_DIR / table / f"year={year}" / "part.parquet"


def write_processed_partition(table: str, year: int, df: pd.DataFrame) -> Path:
    path = processed_partition_path(table, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def write_processed_by_year(table: str, df: pd.DataFrame, date_col: str = "date") -> list[Path]:
    """Split df by year of date_col and write one partition per year."""
    written = []
    years = pd.to_datetime(df[date_col]).dt.year
    for year, sub in df.groupby(years):
        written.append(write_processed_partition(table, int(year), sub))
    return written


def read_processed(table: str, years: list[int] | None = None) -> pd.DataFrame:
    _validate_table(table)
    table_dir = PROCESSED_DIR / table
    if not table_dir.exists():
        return pd.DataFrame()
    parts = sorted(table_dir.glob("year=*/part.parquet"))
    if years is not None:
        wanted = {f"year={y}" for y in years}
        parts = [p for p in parts if p.parent.name in wanted]
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def write_manifest(table: str, metadata: dict) -> Path:
    _validate_table(table)
    path = PROCESSED_DIR / table / "_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return path


def read_manifest(table: str) -> dict | None:
    _validate_table(table)
    path = PROCESSED_DIR / table / "_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
