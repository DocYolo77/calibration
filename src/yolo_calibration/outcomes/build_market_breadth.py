"""`market_breadth_daily` (stock track): descriptive cross-sectional
breadth statistics over the daily ELIGIBLE YOLO universe, plus D0-eligible-
cohort forward performance stats. RS buckets are explicitly descriptive
only — never a leader definition or a threshold selection.

Two conceptually different things share this table, ON DIFFERENT BASES —
do not conflate them:

  A) D0-COHORT FORWARD PERFORMANCE (`median_forward_return_{h}d`,
     `median_mfe_pct_{h}d`, `median_mae_pct_{h}d`, `share_mfe_ge_*_{h}d`,
     `share_positive_{h}d`, `cohort_size_{h}d`): tracks the tickers eligible
     ON DATE D forward to their own D+H outcome row. This is "what happened
     next to today's eligible names" — a cohort followed through time.

  B) FUTURE MARKET BREADTH (`future_rs{bucket}plus_share_{h}d`): the RS
     breadth of the universe that is ACTUALLY ELIGIBLE on date D+H itself
     (that date's own cross-sectional eligible set — via
     `compute_daily_rs_breadth`), looked up H trading days ahead of row-date
     D. This is NOT the D0 cohort followed forward — a ticker that drops
     out of (or newly enters) the eligible universe by D+H is included or
     excluded on its D+H status, not its D0 status. "Of whoever will
     actually be eligible H trading days from now, what fraction shows
     RS>=bucket" rather than "of today's eligible names, what fraction
     individually shows RS>=bucket H days from now".

Fixed 2026-08-13: the previous implementation (formerly
`qqq_health/build_qqq_outcomes.py::build_momentum_environment_outcomes`)
computed (B) by following the D0 cohort forward and checking only whether
each ticker's D+H row existed (`.notna()`), NOT whether that ticker was
actually still eligible at D+H — silently mixing basis (A) into what was
meant to be an independent cross-sectional breadth read. Also, that
function lived inside the QQQ-health module and was gated behind a
successful `qqq_health_daily` build, even though it never actually needed
QQQ constituents/price data — since the QQQ constituents endpoint is
permanently unavailable on the current Massive plan (see README "Bekannte
Limitierungen"), this silently meant `future_market_breadth` never got
computed at all in practice. Extracted here as its own always-buildable
stock-track table, independent of QQQ health availability.
"""

from __future__ import annotations

import pandas as pd

from yolo_calibration.config import load_qqq_health_config
from yolo_calibration.utils.logging import get_logger

logger = get_logger(__name__)


def compute_daily_rs_breadth(stock_features_daily: pd.DataFrame, rs_buckets: list[int],
                              rs_col: str = "rs_percentile_1m") -> pd.DataFrame:
    """One row per date: for that date's OWN actually-eligible universe,
    the share of tickers with rs_col >= bucket, for each bucket in
    rs_buckets. Purely a same-date cross-sectional snapshot — this is the
    series `build_future_market_breadth` shifts to produce a forward-looking
    read; it carries no forward-looking information itself."""
    cols = ["date", "eligible_count"] + [f"breadth_rs{b}plus_share" for b in rs_buckets]
    df = stock_features_daily[stock_features_daily["eligible"].astype(bool)]
    if df.empty:
        return pd.DataFrame(columns=cols)

    out = df.groupby("date").size().rename("eligible_count").reset_index()
    for b in rs_buckets:
        hit = df[rs_col] >= b
        share = hit.groupby(df["date"]).mean()
        out = out.merge(share.rename(f"breadth_rs{b}plus_share").reset_index(), on="date", how="left")
    return out.sort_values("date").reset_index(drop=True)


def build_future_market_breadth(daily_breadth: pd.DataFrame, horizons: list[int],
                                 rs_buckets: list[int]) -> pd.DataFrame:
    """For row-date D and horizon h, `future_rs{bucket}plus_share_{h}d` is
    the breadth share of D+h's OWN eligible universe (see
    compute_daily_rs_breadth) — a plain trading-day shift of that
    date-indexed (no ticker dimension) series, since row i+h in a
    date-sorted, one-row-per-trading-day frame IS "h trading days after
    row i" by construction (same convention as outcomes/build_outcomes.py's
    ticker-wise shifts, just without a ticker groupby here). NaN once the
    D+h date falls outside the available range (incomplete window, no
    silent lookahead)."""
    daily_breadth = daily_breadth.sort_values("date").reset_index(drop=True)
    out = daily_breadth[["date"]].copy()
    for h in horizons:
        for b in rs_buckets:
            col = f"breadth_rs{b}plus_share"
            out[f"future_rs{b}plus_share_{h}d"] = daily_breadth[col].shift(-h).to_numpy()
    return out


def build_d0_cohort_forward_performance(stock_features_daily: pd.DataFrame,
                                         stock_outcomes_daily: pd.DataFrame,
                                         horizons: list[int]) -> pd.DataFrame:
    """Basis (A) from the module docstring: the D0-eligible cohort's own
    forward outcomes, aggregated per date. Unchanged from the pre-2026-08-13
    behavior — deliberately kept separate from (B)."""
    feats = stock_features_daily.sort_values(["ticker", "date"]).copy()
    merged = feats.merge(stock_outcomes_daily, on=["date", "ticker"], how="left")
    elig_mask = merged["eligible"].astype(bool)

    rows = []
    for h in horizons:
        sub = merged.loc[elig_mask, ["date"]].copy()
        sub["forward_return"] = merged.loc[elig_mask, f"forward_return_close_{h}d"]
        sub["mfe_pct"] = merged.loc[elig_mask, f"mfe_pct_{h}d"]
        sub["mae_pct"] = merged.loc[elig_mask, f"mae_pct_{h}d"]
        sub["mfe_ge_5"] = merged.loc[elig_mask, f"reached_plus_5pct_{h}d"].astype("boolean")
        sub["mfe_ge_10"] = merged.loc[elig_mask, f"reached_plus_10pct_{h}d"].astype("boolean")
        sub["mfe_ge_2atr"] = merged.loc[elig_mask, f"reached_2atr_{h}d"].astype("boolean")
        sub["mfe_ge_3atr"] = merged.loc[elig_mask, f"reached_3atr_{h}d"].astype("boolean")
        sub["positive"] = sub["forward_return"] > 0

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
        rows.append(agg)

    out = rows[0]
    for r in rows[1:]:
        out = out.join(r, how="outer")
    return out.reset_index().sort_values("date").reset_index(drop=True)


def build_market_breadth_daily(stock_features_daily: pd.DataFrame,
                                stock_outcomes_daily: pd.DataFrame) -> pd.DataFrame:
    cfg = load_qqq_health_config()["momentum_environment_outcomes"]
    horizons = load_qqq_health_config()["outcome_horizons_days"]
    rs_buckets = cfg["rs_buckets"]

    logger.info("Computing D0-eligible-cohort forward performance...")
    cohort_perf = build_d0_cohort_forward_performance(stock_features_daily, stock_outcomes_daily, horizons)

    logger.info("Computing D+H actual-eligible-universe RS breadth...")
    daily_breadth = compute_daily_rs_breadth(stock_features_daily, rs_buckets)
    future_breadth = build_future_market_breadth(daily_breadth, horizons, rs_buckets)

    out = cohort_perf.merge(future_breadth, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)
