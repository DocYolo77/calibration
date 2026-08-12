"""Orchestrates the `stock_features_daily` table: DATE x TICKER rows with
returns, RS percentiles (vs eligible universe only), thrust + thrust
percentiles, moving averages, and canonical ATR/ATR-extension.

Depends on `market_universe_daily` already being built (eligible flag must
be known BEFORE RS/thrust percentiles are computed — spec section 6/18).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from yolo_calibration.data.loaders import load_grouped_daily_range
from yolo_calibration.features.technical import (
    add_atr_extension,
    add_moving_averages,
    add_relative_strength_percentiles,
    add_returns,
    add_sma50_persistence,
    add_sma50_slope,
    add_thrust,
    add_thrust_percentiles,
    add_true_range_atr,
)
from yolo_calibration.utils.logging import get_logger

logger = get_logger(__name__)


def build_stock_features_daily(market_universe_daily: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if market_universe_daily.empty:
        raise RuntimeError("market_universe_daily is empty — build the universe table first.")

    ohlcv = load_grouped_daily_range(start, end)
    if ohlcv.empty:
        raise RuntimeError("No raw grouped-daily checkpoints found for the requested range.")

    df = add_returns(ohlcv)
    df = add_true_range_atr(df)
    df = add_moving_averages(df)
    df = add_atr_extension(df)
    df = add_sma50_slope(df)
    df = add_sma50_persistence(df)

    df = df.merge(
        market_universe_daily[["date", "ticker", "eligible", "market_cap", "adr20"]],
        on=["date", "ticker"],
        how="left",
    )
    df["eligible"] = df["eligible"].fillna(False)

    # RS percentiles and thrust percentiles MUST be computed against the
    # eligible-only universe, and eligibility is already resolved above —
    # this ordering is asserted by tests/test_eligible_before_rs.py.
    df = add_relative_strength_percentiles(df, eligible_col="eligible")
    df = add_thrust(df)
    df = add_thrust_percentiles(df, eligible_col="eligible")

    cols = [
        "date", "ticker", "eligible", "market_cap", "adr20",
        "close", "high", "low",
        "return_1d", "return_5d", "return_21d", "return_3m", "return_6m", "return_12m",
        "rs_percentile_1d", "rs_percentile_1w", "rs_percentile_1m",
        "rs_percentile_3m", "rs_percentile_6m", "rs_percentile_12m",
        "thrust_1d", "thrust_1w", "thrust_1m",
        "thrust_percentile_1d", "thrust_percentile_1w", "thrust_percentile_1m",
        "ema10", "ema20", "distance_ema10_pct", "distance_ema20_pct",
        "sma50", "sma50_slope_pct", "sma50_persistence_days",
        "atr14", "atr_pct", "atr_extension",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[cols].sort_values(["date", "ticker"]).reset_index(drop=True)
