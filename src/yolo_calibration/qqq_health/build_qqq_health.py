"""Orchestrates `qqq_health_daily`: QQQ price structure + breadth (A/D,
RANA, MCO, MCSI, %>MA, H/L oscillator, slope/change columns), joined on
date. No health STATE labels or thresholds are assigned here (spec
sections 9, 13) — raw features only.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from yolo_calibration.data.loaders import load_grouped_daily_range
from yolo_calibration.qqq_health.breadth import compute_breadth_daily
from yolo_calibration.qqq_health.constituents import build_qqq_constituents_daily
from yolo_calibration.qqq_health.price_structure import compute_qqq_price_structure
from yolo_calibration.utils.logging import get_logger

logger = get_logger(__name__)


def build_qqq_health_daily(start: date, end: date) -> pd.DataFrame:
    logger.info("Loading point-in-time QQQ constituents...")
    constituents = build_qqq_constituents_daily(start, end)

    logger.info("Loading full-market OHLCV for breadth computation...")
    ohlcv_all = load_grouped_daily_range(start, end)
    if ohlcv_all.empty:
        raise RuntimeError("No raw grouped-daily checkpoints found for the requested range.")

    logger.info("Computing breadth features...")
    breadth = compute_breadth_daily(constituents, ohlcv_all)

    qqq_ohlcv = ohlcv_all[ohlcv_all["ticker"] == "QQQ"]
    if qqq_ohlcv.empty:
        raise RuntimeError(
            "No QQQ OHLCV rows found in the raw grouped-daily data — QQQ itself must be "
            "present in the fetched universe to compute price structure."
        )
    logger.info("Computing QQQ price structure...")
    price_structure = compute_qqq_price_structure(qqq_ohlcv)

    out = price_structure.merge(breadth, on="date", how="inner", suffixes=("_qqq", ""))
    return out.sort_values("date").reset_index(drop=True)
