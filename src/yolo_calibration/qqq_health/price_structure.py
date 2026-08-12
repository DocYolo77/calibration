"""QQQ's own price structure (spec section 11 F). Reuses the canonical ATR
implementation from features.technical (same formula, same config) rather
than redefining it — see config/features.yaml `atr`.
"""

from __future__ import annotations

import pandas as pd

from yolo_calibration.config import load_qqq_health_config
from yolo_calibration.features.technical import add_true_range_atr


def compute_qqq_price_structure(qqq_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """qqq_ohlcv: date, ticker(=='QQQ'), open, high, low, close for the QQQ
    ETF itself, sorted by date."""
    cfg = load_qqq_health_config()["qqq_price_structure"]
    adjust = cfg["ema_adjust"]

    df = add_true_range_atr(qqq_ohlcv.sort_values("date").copy())

    df["ema10"] = df["close"].ewm(span=cfg["ema_fast_days"], adjust=adjust).mean()
    df["ema20"] = df["close"].ewm(span=cfg["ema_slow_days"], adjust=adjust).mean()
    df["sma50"] = df["close"].rolling(cfg["sma_trend_days"], min_periods=cfg["sma_trend_days"]).mean()
    df["sma200"] = df["close"].rolling(cfg["sma_long_days"], min_periods=cfg["sma_long_days"]).mean()

    df["distance_ema10_pct"] = (df["close"] - df["ema10"]) / df["ema10"] * 100.0
    df["distance_ema20_pct"] = (df["close"] - df["ema20"]) / df["ema20"] * 100.0
    df["distance_sma50_pct"] = (df["close"] - df["sma50"]) / df["sma50"] * 100.0

    df["ema10_above_ema20"] = df["ema10"] > df["ema20"]
    df["close_above_ema10"] = df["close"] > df["ema10"]
    df["close_above_ema20"] = df["close"] > df["ema20"]
    df["close_above_sma50"] = df["close"] > df["sma50"]
    df["close_above_sma200"] = df["close"] > df["sma200"]

    df["gain_from_sma50_pct"] = (df["close"] - df["sma50"]) / df["sma50"] * 100.0
    df["atr_extension"] = df["gain_from_sma50_pct"] / df["atr_pct"]

    return df[[
        "date", "close", "ema10", "ema20", "sma50", "sma200",
        "distance_ema10_pct", "distance_ema20_pct", "distance_sma50_pct",
        "ema10_above_ema20", "close_above_ema10", "close_above_ema20",
        "close_above_sma50", "close_above_sma200",
        "atr14", "atr_pct", "atr_extension",
    ]]
