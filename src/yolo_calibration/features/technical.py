"""Pure, side-effect-free technical calculations. Every formula here is a
direct implementation of a definition in config/features.yaml — do not
hardcode a window or formula that duplicates config; read it from there.

All functions operate on a tidy per-ticker-sorted DataFrame with columns at
least: date, ticker, open, high, low, close (grouped by ticker, sorted by
date ascending). Rolling/EMA operations are computed per-ticker via
groupby to prevent any cross-ticker leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from yolo_calibration.config import load_features_config, load_universe_config


def _by_ticker(df: pd.DataFrame) -> pd.core.groupby.generic.DataFrameGroupBy:
    return df.sort_values(["ticker", "date"]).groupby("ticker", group_keys=False, sort=False)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_features_config()["returns"]["horizons_days"]
    out = df.copy()
    grp = out.groupby("ticker", sort=False)["close"]
    for col_name, horizon in cfg.items():
        out[col_name] = grp.pct_change(periods=horizon) * 100.0
    return out


def add_true_range_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical ATR — config/features.yaml `atr` block. Simple arithmetic
    mean of True Range over the trailing window, NOT Wilder smoothing."""
    cfg = load_features_config()["atr"]
    window = cfg["window_days"]
    out = df.sort_values(["ticker", "date"]).copy()
    prev_close = out.groupby("ticker", sort=False)["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["true_range"] = tr
    out["atr14"] = (
        out.groupby("ticker", sort=False)["true_range"]
        .transform(lambda s: s.rolling(window=window, min_periods=window).mean())
    )
    out["atr_pct"] = out["atr14"] / out["close"] * 100.0
    return out


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_features_config()["moving_averages"]
    adjust = cfg["ema_adjust"]
    out = df.sort_values(["ticker", "date"]).copy()
    g = out.groupby("ticker", sort=False)["close"]
    out["ema10"] = g.transform(lambda s: s.ewm(span=cfg["ema_fast_days"], adjust=adjust).mean())
    out["ema20"] = g.transform(lambda s: s.ewm(span=cfg["ema_slow_days"], adjust=adjust).mean())
    out["sma50"] = g.transform(lambda s: s.rolling(window=cfg["sma_trend_days"],
                                                     min_periods=cfg["sma_trend_days"]).mean())
    out["distance_ema10_pct"] = (out["close"] - out["ema10"]) / out["ema10"] * 100.0
    out["distance_ema20_pct"] = (out["close"] - out["ema20"]) / out["ema20"] * 100.0
    return out


def add_sma50_slope(df: pd.DataFrame) -> pd.DataFrame:
    """sma50_slope_pct = % change of SMA50 over its own trailing N-day
    lookback (config/features.yaml `sma50_slope.lookback_days`). Requires
    `sma50` already present (see add_moving_averages). NaN until both the
    SMA50 warmup AND the slope lookback are satisfied."""
    cfg = load_features_config()["sma50_slope"]
    lookback = cfg["lookback_days"]
    out = df.sort_values(["ticker", "date"]).copy()
    sma50_lag = out.groupby("ticker", sort=False)["sma50"].shift(lookback)
    out["sma50_slope_pct"] = (out["sma50"] - sma50_lag) / sma50_lag * 100.0
    return out


def add_sma50_persistence(df: pd.DataFrame) -> pd.DataFrame:
    """sma50_persistence_days: signed streak length (trading days, ending
    on the current row) of Close being continuously above (positive) or
    below (negative) SMA50. +1/-1 the day the relationship flips; NaN
    while SMA50 itself is still in its warmup window. Requires `sma50`
    already present (see add_moving_averages). Vectorized via a
    segment-id trick (no per-row Python loop): a new "segment" starts at
    a ticker boundary, an above/below flip, or a warmup-validity change;
    the streak length is the row's position within its segment."""
    out = df.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    valid = out["sma50"].notna()
    above = out["close"] > out["sma50"]

    prev_ticker = out["ticker"].shift()
    # .shift() on a bool Series upcasts to object dtype (to hold the
    # leading NaN); ~ on an object-dtype Series of Python bools applies
    # bitwise invert per element (~True == -2, ~False == -1) instead of
    # logical NOT — both are truthy, silently breaking the flip check.
    # .astype(bool) after fillna forces a real boolean dtype so ~ behaves.
    prev_above = above.shift().fillna(False).astype(bool)
    prev_valid = valid.shift().fillna(False).astype(bool)
    new_segment = (out["ticker"] != prev_ticker) | (above != prev_above) | (~valid) | (~prev_valid)
    segment_id = new_segment.cumsum()

    streak_len = out.groupby(segment_id).cumcount() + 1
    persistence = np.where(above, streak_len, -streak_len).astype(float)
    persistence[~valid.to_numpy()] = np.nan
    out["sma50_persistence_days"] = persistence
    return out


def add_atr_extension(df: pd.DataFrame) -> pd.DataFrame:
    """Requires sma50 and atr_pct already present (see add_moving_averages,
    add_true_range_atr). gain_from_sma50_pct and atr_extension per
    config/features.yaml `atr` block formulas."""
    out = df.copy()
    out["gain_from_sma50_pct"] = (out["close"] - out["sma50"]) / out["sma50"] * 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out["atr_extension"] = out["gain_from_sma50_pct"] / out["atr_pct"]
    out.loc[~np.isfinite(out["atr_extension"]), "atr_extension"] = np.nan
    return out


def compute_adr20(df: pd.DataFrame) -> pd.DataFrame:
    """ADR20 = mean over the trailing 20 completed trading days (inclusive
    of the current row, which — in a historical batch build — is itself a
    completed day) of (High/Low - 1) * 100. See config/universe.yaml."""
    cfg = load_universe_config()["adr20"]
    window = cfg["window_days"]
    out = df.sort_values(["ticker", "date"]).copy()
    daily_range_pct = (out["high"] / out["low"] - 1.0) * 100.0
    out["_daily_range_pct"] = daily_range_pct
    out["adr20"] = (
        out.groupby("ticker", sort=False)["_daily_range_pct"]
        .transform(lambda s: s.rolling(window=window, min_periods=window).mean())
    )
    return out.drop(columns=["_daily_range_pct"])


def add_relative_strength_percentiles(df: pd.DataFrame, eligible_col: str = "eligible") -> pd.DataFrame:
    """Daily point-in-time percentile rank of each return horizon, computed
    ONLY within the eligible universe on that date (config/features.yaml
    `relative_strength`). Non-eligible rows get NaN RS percentiles."""
    cfg = load_features_config()["relative_strength"]["horizons_days"]
    return_source = {
        "rs_percentile_1d": "return_1d",
        "rs_percentile_1w": "return_5d",
        "rs_percentile_1m": "return_21d",
        "rs_percentile_3m": "return_3m",
        "rs_percentile_6m": "return_6m",
        "rs_percentile_12m": "return_12m",
    }
    out = df.copy()
    for pct_col in cfg:
        src_col = return_source[pct_col]
        out[pct_col] = np.nan
        eligible_mask = out[eligible_col].astype(bool)
        ranked = (
            out.loc[eligible_mask]
            .groupby("date", sort=False)[src_col]
            .rank(pct=True, method="average")
            * 100.0
        )
        out.loc[eligible_mask, pct_col] = ranked
    return out


def add_thrust(df: pd.DataFrame) -> pd.DataFrame:
    """Thrust = EMA(short) - EMA(long) of the daily percentage return series
    per config/features.yaml `thrust` block. Requires a `daily_return_pct`
    column (1-day % return)."""
    cfg = load_features_config()["thrust"]
    adjust = cfg["ema_adjust"]
    out = df.sort_values(["ticker", "date"]).copy()
    if "daily_return_pct" not in out.columns:
        out["daily_return_pct"] = out.groupby("ticker", sort=False)["close"].pct_change() * 100.0
    for name, horizon_cfg in cfg["horizons"].items():
        short_ema = (
            out.groupby("ticker", sort=False)["daily_return_pct"]
            .transform(lambda s: s.ewm(span=horizon_cfg["ema_short_days"], adjust=adjust).mean())
        )
        long_ema = (
            out.groupby("ticker", sort=False)["daily_return_pct"]
            .transform(lambda s: s.ewm(span=horizon_cfg["ema_long_days"], adjust=adjust).mean())
        )
        out[name] = short_ema - long_ema
    return out


def add_thrust_percentiles(df: pd.DataFrame, eligible_col: str = "eligible") -> pd.DataFrame:
    cfg = load_features_config()["thrust"]["horizons"]
    out = df.copy()
    eligible_mask = out[eligible_col].astype(bool)
    for name in cfg:
        pct_col = name.replace("thrust_", "thrust_percentile_")
        out[pct_col] = np.nan
        ranked = (
            out.loc[eligible_mask]
            .groupby("date", sort=False)[name]
            .rank(pct=True, method="average")
            * 100.0
        )
        out.loc[eligible_mask, pct_col] = ranked
    return out
