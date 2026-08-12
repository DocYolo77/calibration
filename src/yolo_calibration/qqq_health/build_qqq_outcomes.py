"""`qqq_health_outcomes_daily` (spec section 12): two outcome groups.

  A) INDEX outcomes: forward return / MFE / MAE / max drawdown / new-20D-high
     / EMA20 loss-reclaim, computed on QQQ itself.
  B) MOMENTUM-ENVIRONMENT outcomes: the same forward windows, but
     aggregated across the daily ELIGIBLE stock universe (median forward
     return/MFE/MAE, hit-rate shares, share positive, RS80+/90+/95+ bucket
     shares). These RS buckets are explicitly descriptive only — spec
     section 12 forbids turning them into a "leader" definition in Phase 1.

Interpretation note (flagged as an open question in the Phase 1 report,
since the spec is not fully explicit here): "zukünftige Market Breadth" /
RS80+/90+/95+ shares are computed on the SAME cohort of tickers eligible on
date D, evaluated at their own D+H future row — i.e. "of today's eligible
universe, what fraction shows RS>=80/90/95 a horizon later" — rather than
on whatever (possibly different) set of tickers is eligible on D+H itself.
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


def build_momentum_environment_outcomes(stock_features_daily: pd.DataFrame,
                                         stock_outcomes_daily: pd.DataFrame) -> pd.DataFrame:
    cfg = load_qqq_health_config()["momentum_environment_outcomes"]
    horizons = load_qqq_health_config()["outcome_horizons_days"]
    rs_buckets = cfg["rs_buckets"]

    feats = stock_features_daily.sort_values(["ticker", "date"]).copy()
    merged = feats.merge(stock_outcomes_daily, on=["date", "ticker"], how="left")

    g_rs = merged.groupby("ticker", sort=False)["rs_percentile_1m"]
    g_elig = merged.groupby("ticker", sort=False)["eligible"]

    rows = []
    for h in horizons:
        fwd_rs = g_rs.shift(-h)
        fwd_elig = g_elig.shift(-h)
        elig_mask = merged["eligible"].astype(bool)

        sub = merged.loc[elig_mask, ["date"]].copy()
        sub["forward_return"] = merged.loc[elig_mask, f"forward_return_close_{h}d"]
        sub["mfe_pct"] = merged.loc[elig_mask, f"mfe_pct_{h}d"]
        sub["mae_pct"] = merged.loc[elig_mask, f"mae_pct_{h}d"]
        sub["mfe_ge_5"] = merged.loc[elig_mask, f"reached_plus_5pct_{h}d"].astype("boolean")
        sub["mfe_ge_10"] = merged.loc[elig_mask, f"reached_plus_10pct_{h}d"].astype("boolean")
        sub["mfe_ge_2atr"] = merged.loc[elig_mask, f"reached_2atr_{h}d"].astype("boolean")
        sub["mfe_ge_3atr"] = merged.loc[elig_mask, f"reached_3atr_{h}d"].astype("boolean")
        sub["positive"] = sub["forward_return"] > 0
        sub["future_rs"] = fwd_rs.loc[elig_mask]
        sub["future_elig_valid"] = fwd_elig.loc[elig_mask].notna()

        g = sub.groupby("date")
        agg = pd.DataFrame({
            f"median_forward_return_{h}d": g["forward_return"].median(),
            f"median_mfe_pct_{h}d": g["mfe_pct"].median(),
            f"median_mae_pct_{h}d": g["mae_pct"].median(),
            f"share_mfe_ge_5pct_{h}d": g["mfe_ge_5"].mean(),
            f"share_mfe_ge_10pct_{h}d": g["mfe_ge_10"].mean(),
            f"share_mfe_ge_2atr_{h}d": g["mfe_ge_2atr"].mean(),
            f"share_mfe_ge_3atr_{h}d": g["mfe_ge_3atr"].mean(),
            f"share_positive_{h}d": g["positive"].mean(),
            f"cohort_size_{h}d": g["forward_return"].count(),
        })
        for bucket in rs_buckets:
            valid = sub[sub["future_elig_valid"]]
            share = (
                valid.assign(_hit=valid["future_rs"] >= bucket)
                .groupby("date")["_hit"].mean()
            )
            agg[f"future_rs{bucket}plus_share_{h}d"] = share
        rows.append(agg)

    out = rows[0]
    for r in rows[1:]:
        out = out.join(r, how="outer")
    return out.reset_index().sort_values("date").reset_index(drop=True)


def build_qqq_health_outcomes_daily(qqq_price_structure: pd.DataFrame,
                                     stock_features_daily: pd.DataFrame,
                                     stock_outcomes_daily: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building QQQ index outcomes...")
    index_outcomes = build_qqq_index_outcomes(qqq_price_structure)
    logger.info("Building momentum-environment outcomes...")
    env_outcomes = build_momentum_environment_outcomes(stock_features_daily, stock_outcomes_daily)
    return index_outcomes.merge(env_outcomes, on="date", how="outer").sort_values("date").reset_index(drop=True)
