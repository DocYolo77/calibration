"""`qqq_health_outcomes_daily`: INDEX outcomes only — forward return / MFE /
MAE / max drawdown / new-20D-high / EMA20 loss-reclaim, computed on QQQ
itself. Requires `qqq_health_daily` (which requires point-in-time QQQ
constituents — see `qqq_health/constituents.py`), so this table is empty
whenever that track is unavailable (currently permanent, Massive plan
limitation — see README "Bekannte Limitierungen"), same as before.

The MOMENTUM-ENVIRONMENT / "future market breadth" outcomes that used to
live in this module (`build_momentum_environment_outcomes`) were extracted
2026-08-13 into `outcomes/build_market_breadth.py` as the always-buildable
`market_breadth_daily` table: that computation never actually needed QQQ
constituents or QQQ price data (only `stock_features_daily` +
`stock_outcomes_daily`), so gating it behind this module's QQQ-health
prerequisite meant it silently never ran in practice. See that module's
docstring for the semantics fix (D+H's own actually-eligible universe,
not the D0 cohort followed forward) and the reasoning for the split.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from yolo_calibration.config import load_qqq_health_config
from yolo_calibration.outcomes.build_outcomes import build_horizon_outcomes, compute_new_20d_high_flag
from yolo_calibration.utils.logging import get_logger

logger = get_logger(__name__)


def _index_max_drawdown(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Proper forward max drawdown: within the H-day forward window, track
    the running peak of `high` seen so far (starting from the entry day's
    close as the initial reference) and take the worst (most negative)
    (low / running_peak - 1) across the window. This differs from MAE_pct
    (which only measures decline from the entry close, ignoring any new
    high made mid-window before the trough)."""
    g_high = df.groupby("ticker", sort=False)["high"]
    g_low = df.groupby("ticker", sort=False)["low"]
    close0 = df["close"].to_numpy()

    fwd_high = pd.concat([g_high.shift(-k) for k in range(1, horizon + 1)], axis=1).to_numpy()
    fwd_low = pd.concat([g_low.shift(-k) for k in range(1, horizon + 1)], axis=1).to_numpy()

    # running_peak[:, k] = max(entry close, high_1, ..., high_{k+1}) — the
    # cumulative peak available by forward day k+1, inclusive of that day's
    # own high (a same-day peak-then-reversal is still the worst case seen
    # by that day).
    combined = np.concatenate([close0.reshape(-1, 1), fwd_high], axis=1)
    cummax_combined = np.maximum.accumulate(combined, axis=1)
    running_peak = cummax_combined[:, 1:]

    with np.errstate(invalid="ignore", divide="ignore"):
        dd = (fwd_low / running_peak - 1.0) * 100.0
    window_complete = (~np.isnan(fwd_high)).sum(axis=1) == horizon
    max_dd = np.where(window_complete, np.nanmin(dd, axis=1), np.nan)
    return pd.Series(max_dd, index=df.index)


def build_qqq_index_outcomes(qqq_price_structure: pd.DataFrame) -> pd.DataFrame:
    """qqq_price_structure: date, close, high, low, atr14, close_above_ema20
    (output of qqq_health.price_structure.compute_qqq_price_structure)."""
    horizons = load_qqq_health_config()["outcome_horizons_days"]
    lookback = 20

    df = qqq_price_structure.sort_values("date").reset_index(drop=True).copy()
    df["ticker"] = "QQQ"
    df["is_new_20d_high"] = compute_new_20d_high_flag(df, lookback)

    pieces = [df[["date"]]]
    for h in horizons:
        base = build_horizon_outcomes(df, h)
        base = base.rename(columns={
            f"forward_return_close_{h}d": f"qqq_forward_return_{h}d",
            f"mfe_pct_{h}d": f"qqq_mfe_pct_{h}d",
            f"mae_pct_{h}d": f"qqq_mae_pct_{h}d",
            f"new_20d_high_within_window_{h}d": f"qqq_new_20d_high_within_window_{h}d",
        })
        keep = [c for c in base.columns if c.startswith((
            f"qqq_forward_return_{h}d", f"qqq_mfe_pct_{h}d", f"qqq_mae_pct_{h}d",
            f"qqq_new_20d_high_within_window_{h}d",
        ))]
        piece = base[keep].copy()
        piece[f"qqq_max_drawdown_pct_{h}d"] = _index_max_drawdown(df, h)

        g_ema20 = df["close_above_ema20"]
        fwd_ema20 = pd.concat([g_ema20.shift(-k) for k in range(1, h + 1)], axis=1)
        window_complete = fwd_ema20.notna().sum(axis=1) == h
        was_above = df["close_above_ema20"]
        ever_false = (~fwd_ema20.astype("boolean")).any(axis=1)
        ever_true = fwd_ema20.astype("boolean").any(axis=1)
        piece[f"qqq_ema20_lost_within_window_{h}d"] = np.where(window_complete, was_above & ever_false, np.nan)
        piece[f"qqq_ema20_reclaimed_within_window_{h}d"] = np.where(
            window_complete, (~was_above.astype("boolean")) & ever_true, np.nan
        )
        pieces.append(piece)

    return pd.concat(pieces, axis=1)


def build_qqq_health_outcomes_daily(qqq_price_structure: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building QQQ index outcomes...")
    return build_qqq_index_outcomes(qqq_price_structure)
