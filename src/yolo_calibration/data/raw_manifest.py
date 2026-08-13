"""Materializes `raw_manifest`: a versioned, reproducible processed table
recording exactly which raw checkpoint files exist, for each raw data
source, over a date range (spec sections 14/17 reproducibility). One row
per (source, date): whether a checkpoint was found and its row count.

This is COVERAGE metadata only — "did we fetch a checkpoint for this
source/date, and how many rows did it contain" — not a correctness check
of the fetched content itself (see reports/data_quality.py for content-level
diagnostics). Its purpose is that a later audit or a Phase 2 re-run can
answer "was the full requested range actually fetched, for every source,
without re-scanning the raw parquet tree by hand".
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from yolo_calibration.data.storage import (
    list_raw_grouped_daily_dates,
    list_raw_grouped_daily_unadjusted_dates,
    list_raw_qqq_constituents_dates,
    list_raw_reference_tickers_dates,
    read_raw_grouped_daily,
    read_raw_grouped_daily_unadjusted,
    read_raw_qqq_constituents,
    read_raw_reference_tickers,
)

RAW_SOURCES = ("grouped_daily", "grouped_daily_unadjusted", "qqq_constituents", "reference_tickers")

_SOURCE_LISTERS = {
    "grouped_daily": (list_raw_grouped_daily_dates, read_raw_grouped_daily),
    "grouped_daily_unadjusted": (list_raw_grouped_daily_unadjusted_dates, read_raw_grouped_daily_unadjusted),
    "qqq_constituents": (list_raw_qqq_constituents_dates, read_raw_qqq_constituents),
    "reference_tickers": (list_raw_reference_tickers_dates, read_raw_reference_tickers),
}


def build_raw_manifest(start: date, end: date) -> pd.DataFrame:
    """One row per (source, date) with a checkpoint present in [start, end].
    row_count is 0 for a legitimately-empty checkpoint (e.g. a market
    holiday for grouped_daily, or a QQQ constituents day with zero
    returned holdings) — a present-but-empty checkpoint is NOT the same as
    a missing one, and both are distinguishable via this table (missing
    dates simply don't appear as a row at all)."""
    rows = []
    for source, (list_dates, read_fn) in _SOURCE_LISTERS.items():
        for d in list_dates(start, end):
            df = read_fn(d)
            rows.append({
                "source": source,
                "date": pd.Timestamp(d),
                "row_count": 0 if df is None else int(len(df)),
            })
    if not rows:
        return pd.DataFrame(columns=["source", "date", "row_count"])
    return pd.DataFrame(rows).sort_values(["source", "date"]).reset_index(drop=True)
