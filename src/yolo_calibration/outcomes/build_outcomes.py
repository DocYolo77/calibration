"""Stock Outcome Matrix (spec section 8): forward-looking, per DATE x
TICKER, computed for horizons 5/10/20 trading days. No future data ever
leaks INTO a feature — this module is the one place forward data is used,
and only to produce OUTCOME columns, never features.

Column naming convention: every outcome column is suffixed `_{H}d` for
horizon H in {5, 10, 20}, e.g. `mfe_pct_5d`, `reached_2atr_20d`.

Tie-break convention (documented, not a trading decision — KEPT as-is per
explicit Phase 1 decision, no intraday pipeline added): for
`reached_plus_X_before_minus_X`, if both the +X% and -X% thresholds are
touched on the SAME forward trading day, the decline is conservatively
treated as having occurred first (see the first-hit-day loop below — a
strict `<` comparison between first_up_day and first_down_day implements
this). Daily OHLC alone cannot tell us the true intraday order in this
case — a same-day co-occurrence is genuinely NON-DETERMINABLE from the
data we have, not merely inconvenient to compute. Every such case is
counted and surfaced via a companion boolean diagnostic column,
`reached_plus_X_before_minus_X_tie_{H}d` (True exactly when the tie-break
rule had to be invoked to produce the corresponding `reached_plus_X_before_minus_X_{H}d`
value) — see the per-threshold loop below and
reports/data_quality.py for the aggregated count. This makes the scope of
the convention's effect on the resulting statistic auditable rather than
merely a documented caveat.

Asymmetric race pairs (added 2026-08-14, config `outcome_thresholds.asymmetric_race_pairs`,
e.g. `[10.0, 5.0]`): the same "reached before" logic but with a DIFFERENT
threshold on the up vs down side — e.g. `reached_plus_10_before_minus_5_{H}d`
answers "did it reach +10% before it gave back -5%", not the symmetric
+10%-before--10% question. Requires both thresholds to already be in
pct_moves (reuses their first-hit-day arrays, no extra O(n*horizon) pass).
Same conservative tie-break and same `_tie` diagnostic convention as the
symmetric pairs above.

Memory note (found 2026-08-13 debugging an OOM on the full 2022-2026
backfill, ~12M rows): the original implementation materialized an
(n_rows, horizon) matrix per forward-looking quantity (high, low, the
new-20d-high flag, plus per-threshold hit matrices) via
pd.concat([...shift(-k)...]). For horizon=20 and n_rows in the tens of
millions that is tens of GB of transient float/bool matrices — enough to
OOM-kill the runner outright (no clean Python traceback, just a dead
process — which is why the failure showed no usable logs and skipped the
`if: always()` cache-save step, since the whole VM went down). Rewritten
below as a single O(horizon) pass accumulating O(n_rows) running
arrays (running max/min/any, per-threshold first-hit-day) instead of
O(n_rows * horizon) matrices — same asymptotic time, a small constant
number of O(n) arrays in memory at any point instead of O(horizon) of
them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from yolo_calibration.config import load_features_config
from yolo_calibration.utils.logging import get_logger

logger = get_logger(__name__)


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
    asymmetric_race_pairs = cfg.get("asymmetric_race_pairs", [])
    for up, down in asymmetric_race_pairs:
        if up not in pct_thresholds or down not in pct_thresholds:
            raise ValueError(
                f"asymmetric_race_pairs entry [{up}, {down}] requires both thresholds to already be "
                f"listed in pct_moves ({pct_thresholds}) — first-hit-day arrays are reused, not recomputed."
            )

    n = len(df)
    close0 = df["close"].to_numpy()
    atr0 = df[atr_col].to_numpy()

    g_high = df.groupby("ticker", sort=False)["high"]
    g_low = df.groupby("ticker", sort=False)["low"]
    g_close = df.groupby("ticker", sort=False)["close"]
    g_new_high = df.groupby("ticker", sort=False)["is_new_20d_high"]

    # O(n) running accumulators, updated incrementally over k=1..horizon
    # (see module docstring — replaces the old O(n*horizon) matrices).
    running_max_high = np.full(n, -np.inf)
    running_min_low = np.full(n, np.inf)
    running_any_new_high = np.zeros(n, dtype=bool)
    seen_count = np.zeros(n, dtype=np.int32)
    up_first_hit = {th: np.full(n, horizon, dtype=np.int32) for th in pct_thresholds}
    down_first_hit = {th: np.full(n, horizon, dtype=np.int32) for th in pct_thresholds}

    for k in range(1, horizon + 1):
        fwd_high_k = g_high.shift(-k).to_numpy()
        fwd_low_k = g_low.shift(-k).to_numpy()
        fwd_new_high_k = g_new_high.shift(-k).fillna(False).astype(bool).to_numpy()

        valid_k = ~np.isnan(fwd_high_k)
        seen_count += valid_k

        running_max_high = np.where(valid_k, np.fmax(running_max_high, fwd_high_k), running_max_high)
        running_min_low = np.where(valid_k, np.fmin(running_min_low, fwd_low_k), running_min_low)
        running_any_new_high |= (valid_k & fwd_new_high_k)

        up_pct_k = (fwd_high_k / close0 - 1.0) * 100.0
        down_pct_k = (fwd_low_k / close0 - 1.0) * 100.0
        for th in pct_thresholds:
            up_hit = valid_k & (up_pct_k >= th) & (up_first_hit[th] == horizon)
            up_first_hit[th] = np.where(up_hit, k - 1, up_first_hit[th])
            down_hit = valid_k & (down_pct_k <= -th) & (down_first_hit[th] == horizon)
            down_first_hit[th] = np.where(down_hit, k - 1, down_first_hit[th])

    window_complete = seen_count == horizon

    out = pd.DataFrame(index=df.index)
    suffix = f"_{horizon}d"

    # Forward close at exactly t+H (single O(n) shift, unchanged).
    fwd_close_h = g_close.shift(-horizon)
    out[f"forward_return_close{suffix}"] = (fwd_close_h / df["close"] - 1.0) * 100.0

    mfe_pct = (running_max_high / close0 - 1.0) * 100.0
    mae_pct = (running_min_low / close0 - 1.0) * 100.0
    mfe_pct = np.where(window_complete, mfe_pct, np.nan)
    mae_pct = np.where(window_complete, mae_pct, np.nan)
    out[f"mfe_pct{suffix}"] = mfe_pct
    out[f"mae_pct{suffix}"] = mae_pct

    for th in pct_thresholds:
        th_col = str(int(th)) if float(th).is_integer() else str(th)
        out[f"reached_plus_{th_col}pct{suffix}"] = np.where(window_complete, mfe_pct >= th, np.nan)

    mfe_atr_multiple = np.where(
        window_complete,
        np.divide(running_max_high - close0, atr0, out=np.full_like(close0, np.nan, dtype=float),
                  where=(atr0 != 0) & ~np.isnan(atr0)),
        np.nan,
    )
    mae_atr_multiple = np.where(
        window_complete,
        np.divide(running_min_low - close0, atr0, out=np.full_like(close0, np.nan, dtype=float),
                  where=(atr0 != 0) & ~np.isnan(atr0)),
        np.nan,
    )
    out[f"mfe_atr_multiple{suffix}"] = mfe_atr_multiple
    out[f"mae_atr_multiple{suffix}"] = mae_atr_multiple

    for mult in atr_multiples:
        mult_col = str(int(mult)) if float(mult).is_integer() else str(mult)
        out[f"reached_{mult_col}atr{suffix}"] = np.where(window_complete, mfe_atr_multiple >= mult, np.nan)

    out[f"new_20d_high_within_window{suffix}"] = np.where(window_complete, running_any_new_high, np.nan)

    # Sequential "reached +X% before -X%" per threshold. Tie-break: a
    # strict `<` means a same-day co-occurrence resolves to "down first"
    # (down's first_hit is never made LARGER than up's in a tie), matching
    # the documented conservative convention. `_tie` companion column:
    # True exactly when up_first_hit == down_first_hit on an ACTUAL hit day
    # (< horizon, not the two "never reached" defaults coincidentally
    # matching) — i.e. the case daily OHLC genuinely cannot order.
    for th in pct_thresholds:
        th_col = str(int(th)) if float(th).is_integer() else str(th)
        up_day, down_day = up_first_hit[th], down_first_hit[th]
        reached_before = up_day < down_day
        out[f"reached_plus_{th_col}_before_minus_{th_col}{suffix}"] = np.where(
            window_complete, reached_before, np.nan
        )
        same_day_tie = (up_day == down_day) & (up_day < horizon)
        out[f"reached_plus_{th_col}_before_minus_{th_col}_tie{suffix}"] = np.where(
            window_complete, same_day_tie, np.nan
        )

    # Asymmetric "race" pairs (e.g. +10% before -5%): same tie-break
    # convention and same _tie diagnostic, just up/down thresholds may
    # differ. Reuses up_first_hit[up] / down_first_hit[down] computed above.
    for up, down in asymmetric_race_pairs:
        up_col = str(int(up)) if float(up).is_integer() else str(up)
        down_col = str(int(down)) if float(down).is_integer() else str(down)
        up_day, down_day = up_first_hit[up], down_first_hit[down]
        reached_before = up_day < down_day
        out[f"reached_plus_{up_col}_before_minus_{down_col}{suffix}"] = np.where(
            window_complete, reached_before, np.nan
        )
        same_day_tie = (up_day == down_day) & (up_day < horizon)
        out[f"reached_plus_{up_col}_before_minus_{down_col}_tie{suffix}"] = np.where(
            window_complete, same_day_tie, np.nan
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
