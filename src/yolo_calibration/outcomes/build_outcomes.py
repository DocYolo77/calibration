"""Stock Outcome Matrix (spec section 8): forward-looking, per DATE x
TICKER, computed for horizons 5/10/20 trading days. No future data ever
leaks INTO a feature — this module is the one place forward data is used,
and only to produce OUTCOME columns, never features.

Column naming convention: every outcome column is suffixed `_{H}d` for
horizon H in {5, 10, 20}, e.g. `mfe_pct_5d`, `reached_2atr_20d`.

Tie-break convention (documented, not a trading decision): for
`reached_plus_X_before_minus_X`, if both the +X% and -X% thresholds are
touched on the SAME forward trading day, the decline is conservatively
treated as having occurred first (a plain `first_up_day < first_down_day`
strict comparison implements this — see tests/test_outcomes.py). This
convention is flagged as an open methodological note in the Phase 1 report
since it can affect the resulting statistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from yolo_calibration.config import load_features_config
from yolo_calibration.utils.logging import get_logger

logger = get_logger(__name__)


def _forward_stack(grouped: pd.core.groupby.generic.SeriesGroupBy, horizon: int) -> pd.DataFrame:
    """Columns 0..horizon-1 hold shift(-1)..shift(-horizon): the next
    `horizon` trading days' values for that ticker, NaN once past the
    series end or across a ticker boundary (groupby.shift respects group
    boundaries)."""
    cols = {k: grouped.shift(-k) for k in range(1, horizon + 1)}
    return pd.DataFrame(cols)


def _first_hit_index(hit_matrix: np.ndarray) -> np.ndarray:
    """hit_matrix: bool array (n, H). Returns int array: 0-based index of
    first True per row, or H (sentinel = "never within window") if none."""
    n, h = hit_matrix.shape
    any_hit = hit_matrix.any(axis=1)
    first_idx = np.where(any_hit, hit_matrix.argmax(axis=1), h)
    return first_idx


def compute_new_20d_high_flag(df: pd.DataFrame, lookback_days: int) -> pd.Series:
    """Per-row boolean: is this day's high a new N-day high vs the trailing
    N days strictly BEFORE this day (no lookahead)."""
    trailing_max = (
        df.groupby("ticker", sort=False)["high"]
        .transform(lambda s: s.shift(1).rolling(window=lookback_days, min_periods=lookback_days).max())
    )
    return df["high"] > trailing_max


def build_horizon_outcomes(df: pd.DataFrame, horizon: int, *,
                            atr_col: str = "atr14") -> pd.DataFrame:
    """df must be sorted by ticker,date and contain: ticker, date, close,
    high, low, atr14, and a precomputed `is_new_20d_high` boolean column."""
    cfg = load_features_config()["outcome_thresholds"]
    pct_thresholds = cfg["pct_moves"]
    atr_multiples = cfg["atr_multiples"]

    g_high = df.groupby("ticker", sort=False)["high"]
    g_low = df.groupby("ticker", sort=False)["low"]
    g_close = df.groupby("ticker", sort=False)["close"]
    g_new_high = df.groupby("ticker", sort=False)["is_new_20d_high"]

    fwd_high = _forward_stack(g_high, horizon)
    fwd_low = _forward_stack(g_low, horizon)
    fwd_new_high = _forward_stack(g_new_high, horizon)

    window_complete = fwd_high.notna().sum(axis=1) == horizon

    close0 = df["close"].to_numpy()
    atr0 = df[atr_col].to_numpy()

    out = pd.DataFrame(index=df.index)
    suffix = f"_{horizon}d"

    # Forward close at exactly t+H
    fwd_close_h = g_close.shift(-horizon)
    out[f"forward_return_close{suffix}"] = (fwd_close_h / df["close"] - 1.0) * 100.0

    window_max_high = fwd_high.max(axis=1)
    window_min_low = fwd_low.min(axis=1)

    mfe_pct = (window_max_high / close0 - 1.0) * 100.0
    mae_pct = (window_min_low / close0 - 1.0) * 100.0
    mfe_pct = np.where(window_complete, mfe_pct, np.nan)
    mae_pct = np.where(window_complete, mae_pct, np.nan)
    out[f"mfe_pct{suffix}"] = mfe_pct
    out[f"mae_pct{suffix}"] = mae_pct

    for th in pct_thresholds:
        th_col = str(int(th)) if float(th).is_integer() else str(th)
        out[f"reached_plus_{th_col}pct{suffix}"] = np.where(window_complete, mfe_pct >= th, np.nan)

    mfe_atr_multiple = np.where(
        window_complete,
        np.divide(window_max_high - close0, atr0, out=np.full_like(close0, np.nan, dtype=float),
                  where=(atr0 != 0) & ~np.isnan(atr0)),
        np.nan,
    )
    mae_atr_multiple = np.where(
        window_complete,
        np.divide(window_min_low - close0, atr0, out=np.full_like(close0, np.nan, dtype=float),
                  where=(atr0 != 0) & ~np.isnan(atr0)),
        np.nan,
    )
    out[f"mfe_atr_multiple{suffix}"] = mfe_atr_multiple
    out[f"mae_atr_multiple{suffix}"] = mae_atr_multiple

    for mult in atr_multiples:
        mult_col = str(int(mult)) if float(mult).is_integer() else str(mult)
        out[f"reached_{mult_col}atr{suffix}"] = np.where(window_complete, mfe_atr_multiple >= mult, np.nan)

    new_high_any = fwd_new_high.astype("boolean").any(axis=1)
    out[f"new_20d_high_within_window{suffix}"] = np.where(window_complete, new_high_any, np.nan)

    # Sequential "reached +X% before -X%" per threshold
    for th in pct_thresholds:
        th_col = str(int(th)) if float(th).is_integer() else str(th)
        up_hits = (fwd_high.to_numpy() / close0[:, None] - 1.0) * 100.0 >= th
        down_hits = (fwd_low.to_numpy() / close0[:, None] - 1.0) * 100.0 <= -th
        first_up = _first_hit_index(up_hits)
        first_down = _first_hit_index(down_hits)
        reached_before = first_up < first_down
        out[f"reached_plus_{th_col}_before_minus_{th_col}{suffix}"] = np.where(
            window_complete, reached_before, np.nan
        )

    return out


def build_stock_outcomes_daily(stock_features_daily: pd.DataFrame) -> pd.DataFrame:
    """stock_features_daily must already contain: date, ticker, close, high,
    low, atr14 (the signal-day ATR — never a future ATR, per spec)."""
    required = {"date", "ticker", "close", "high", "low", "atr14"}
    missing = required - set(stock_features_daily.columns)
    if missing:
        raise ValueError(f"stock_features_daily missing required columns: {missing}")

    horizons = load_features_config()["outcome_horizons_days"]
    lookback = load_features_config()["outcome_thresholds"]["new_high_lookback_days"]

    df = stock_features_daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["is_new_20d_high"] = compute_new_20d_high_flag(df, lookback)

    pieces = [df[["date", "ticker"]]]
    for h in horizons:
        logger.info("Building %d-day outcome columns...", h)
        pieces.append(build_horizon_outcomes(df, h))

    return pd.concat(pieces, axis=1)
