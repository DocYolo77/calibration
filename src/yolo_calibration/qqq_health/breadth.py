"""QQQ breadth features computed against the point-in-time constituent set
for each day (spec section 11 A-E, G). All per-ticker moving averages /
new-high-low flags are computed on each ticker's own full price history
(so warmup periods before a ticker joined QQQ are usable), then restricted
to that day's actual constituent membership before aggregating cross-
sectionally — this is what makes the breadth stats point-in-time correct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from yolo_calibration.config import load_qqq_health_config


def _per_ticker_ma_panel(ohlcv: pd.DataFrame) -> pd.DataFrame:
    cfg = load_qqq_health_config()["breadth"]["pct_above_ma"]
    hl_cfg = load_qqq_health_config()["breadth"]["high_low_oscillator"]
    adjust = cfg["ema_adjust"]

    df = ohlcv.sort_values(["ticker", "date"]).copy()
    g_close = df.groupby("ticker", sort=False)["close"]
    g_high = df.groupby("ticker", sort=False)["high"]

    df["sma5"] = g_close.transform(lambda s: s.rolling(cfg["sma5"], min_periods=cfg["sma5"]).mean())
    df["ema10"] = g_close.transform(lambda s: s.ewm(span=cfg["ema10"], adjust=adjust).mean())
    df["sma20"] = g_close.transform(lambda s: s.rolling(cfg["sma20"], min_periods=cfg["sma20"]).mean())
    df["ema21"] = g_close.transform(lambda s: s.ewm(span=cfg["ema21"], adjust=adjust).mean())
    df["sma50"] = g_close.transform(lambda s: s.rolling(cfg["sma50"], min_periods=cfg["sma50"]).mean())
    df["sma200"] = g_close.transform(lambda s: s.rolling(cfg["sma200"], min_periods=cfg["sma200"]).mean())

    lb = hl_cfg["lookback_days"]
    trailing_max_high = g_high.transform(lambda s: s.shift(1).rolling(lb, min_periods=lb).max())
    trailing_min_low_src = df.groupby("ticker", sort=False)["low"]
    trailing_min_low = trailing_min_low_src.transform(lambda s: s.shift(1).rolling(lb, min_periods=lb).min())
    df["is_new_20d_high"] = df["high"] > trailing_max_high
    df["is_new_20d_low"] = df["low"] < trailing_min_low

    df["prev_close"] = g_close.shift(1)
    df["is_advancer"] = df["close"] > df["prev_close"]
    df["is_decliner"] = df["close"] < df["prev_close"]

    return df


def compute_breadth_daily(qqq_constituents_daily: pd.DataFrame, ohlcv_all: pd.DataFrame) -> pd.DataFrame:
    """qqq_constituents_daily: date, constituent_ticker (from
    qqq_health.constituents.build_qqq_constituents_daily).
    ohlcv_all: date, ticker, open, high, low, close, volume (full-market
    grouped daily, will be restricted to ever-constituent tickers)."""
    tickers_needed = set(qqq_constituents_daily["constituent_ticker"].unique())
    panel = _per_ticker_ma_panel(ohlcv_all[ohlcv_all["ticker"].isin(tickers_needed)])

    joined = qqq_constituents_daily.merge(
        panel, left_on=["date", "constituent_ticker"], right_on=["date", "ticker"], how="left"
    )

    def _pct_above(col: str) -> pd.Series:
        valid = joined[col].notna()
        above = (joined["close"] > joined[col]) & valid
        g = joined.assign(_above=above, _valid=valid).groupby("date")
        return (g["_above"].sum() / g["_valid"].sum().replace(0, np.nan)) * 100.0

    g = joined.groupby("date")
    out = pd.DataFrame(index=sorted(joined["date"].unique()))
    out.index.name = "date"

    advancers = g["is_advancer"].sum()
    decliners = g["is_decliner"].sum()
    component_count = g["constituent_ticker"].nunique()

    out["component_count"] = component_count
    out["advancers"] = advancers
    out["decliners"] = decliners

    rana_scale = load_qqq_health_config()["breadth"]["rana_scale"]
    denom = (advancers + decliners).clip(lower=1)
    out["rana"] = ((advancers - decliners) / denom) * rana_scale

    out["pct_above_sma5"] = _pct_above("sma5")
    out["pct_above_ema10"] = _pct_above("ema10")
    out["pct_above_sma20"] = _pct_above("sma20")
    out["pct_above_ema21"] = _pct_above("ema21")
    out["pct_above_sma50"] = _pct_above("sma50")
    out["pct_above_sma200"] = _pct_above("sma200")

    new_highs = g["is_new_20d_high"].sum()
    new_lows = g["is_new_20d_low"].sum()
    out["new_20d_highs"] = new_highs
    out["new_20d_lows"] = new_lows
    out["high_low_oscillator"] = new_highs - new_lows
    out["high_low_pct"] = (new_highs - new_lows) / component_count.replace(0, np.nan) * 100.0

    out = out.reset_index().sort_values("date").reset_index(drop=True)
    return add_mco_mcsi(out)


def add_mco_mcsi(breadth_daily: pd.DataFrame) -> pd.DataFrame:
    """McClellan Oscillator + Summation Index (spec section 11 B/C).

    MCSI warmup rule (documented, mirrors the existing YOLO dashboard
    convention): the summation index is seeded at 0 on the first available
    day and is the running cumulative sum of MCO from that point forward.
    Because MCO itself needs its 39-day EMA to stabilize, the first ~39
    rows of MCSI are transitional (built from a not-yet-converged MCO) —
    this is the same warmup behavior as the source dashboard and is
    intentionally NOT patched with a synthetic seed value, to avoid
    inventing an unreviewed methodology change.
    """
    cfg = load_qqq_health_config()["breadth"]["mco"]
    mcsi_cfg = load_qqq_health_config()["breadth"]["mcsi"]
    adjust = cfg["ema_adjust"]

    out = breadth_daily.sort_values("date").copy()
    ema_short = out["rana"].ewm(span=cfg["ema_short_days"], adjust=adjust).mean()
    ema_long = out["rana"].ewm(span=cfg["ema_long_days"], adjust=adjust).mean()
    out["mco_raw"] = ema_short - ema_long

    roll_win = cfg["zscore_rolling_window_days"]
    min_p = cfg["zscore_min_periods"]
    mco_mean = out["mco_raw"].rolling(roll_win, min_periods=min_p).mean()
    mco_std = out["mco_raw"].rolling(roll_win, min_periods=min_p).std()
    out["mco_z"] = (out["mco_raw"] - mco_mean) / mco_std

    out["mcsi_raw"] = mcsi_cfg["seed_value"] + out["mco_raw"].cumsum()

    m_roll_win = mcsi_cfg["zscore_rolling_window_days"]
    m_min_p = mcsi_cfg["zscore_min_periods"]
    mcsi_mean = out["mcsi_raw"].rolling(m_roll_win, min_periods=m_min_p).mean()
    mcsi_std = out["mcsi_raw"].rolling(m_roll_win, min_periods=m_min_p).std()
    out["mcsi_z"] = (out["mcsi_raw"] - mcsi_mean) / mcsi_std
    out["mcsi_z_sma10"] = out["mcsi_z"].rolling(mcsi_cfg["sma_smooth_days"],
                                                 min_periods=mcsi_cfg["sma_smooth_days"]).mean()
    out["mcsi_z_minus_sma10"] = out["mcsi_z"] - out["mcsi_z_sma10"]

    return add_change_columns(out)


def add_change_columns(breadth_daily: pd.DataFrame) -> pd.DataFrame:
    """1D/3D/5D changes for the level-vs-trend fields (spec section 11 G)."""
    windows = load_qqq_health_config()["breadth"]["change_windows_days"]
    out = breadth_daily.copy()
    change_cols = [
        "mco_raw", "mco_z", "mcsi_z",
        "pct_above_ema10", "pct_above_ema21", "pct_above_sma50",
        "high_low_oscillator",
    ]
    for col in change_cols:
        if col not in out.columns:
            continue
        for w in windows:
            out[f"{col}_chg_{w}d"] = out[col].diff(w)
    if "high_low_oscillator" in out.columns:
        out["high_low_oscillator_5d_avg"] = out["high_low_oscillator"].rolling(5, min_periods=5).mean()
    return out
