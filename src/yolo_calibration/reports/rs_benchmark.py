"""RS Benchmark Report: purely descriptive comparison of every RS horizon
(1D/1W/1M/3M/6M/12M) against the existing 5D/10D/20D stock outcomes, for a
single benchmark year at a time.

Two bucket schemes over the RS percentile scale (0-100), both FIXED
value-range bins (not data-driven quantiles) since rs_percentile_* is
already a percentile rank — a fixed-width bin on that scale has a direct,
stable interpretation across horizons and years:

  - COARSE: 10 buckets, width 10 ("0-10", "10-20", ..., "90-100"), spanning
    the full range.
  - FINE_ABOVE_80: 8 buckets, width 2.5 ("80.0-82.5", ..., "97.5-100"),
    spanning only the top quintile, for higher resolution where RS-driven
    momentum-expansion effects are most likely to show curvature.

Per (RS horizon, bucket, outcome horizon): sample size, median/mean of
mfe_pct/mae_pct/forward_return_close, and the hit-probability of each of
reached_plus_5pct/10pct/2atr/3atr. Every number here is a plain descriptive
aggregate — this module MUST NOT select a threshold, flag a "best" bucket,
or pick a "best" RS horizon. That decision-making is explicitly out of
scope (deferred, see README "Offen für Phase 2" point 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RS_HORIZONS = (
    "rs_percentile_1d", "rs_percentile_1w", "rs_percentile_1m",
    "rs_percentile_3m", "rs_percentile_6m", "rs_percentile_12m",
)
OUTCOME_HORIZONS_DAYS = (5, 10, 20)
PCT_THRESHOLDS = (5, 10)
ATR_MULTIPLES = (2, 3)

_COARSE_BINS = list(range(0, 101, 10))
_COARSE_LABELS = [f"{lo}-{hi}" for lo, hi in zip(_COARSE_BINS[:-1], _COARSE_BINS[1:])]

_FINE_BINS = [80 + 2.5 * i for i in range(9)]
_FINE_LABELS = [f"{lo:g}-{hi:g}" for lo, hi in zip(_FINE_BINS[:-1], _FINE_BINS[1:])]


def bucket_coarse_deciles(rs: pd.Series) -> pd.Series:
    """Fixed-width value-range deciles over the full [0, 100] RS scale.
    Right-closed bins ((lo, hi], first bin additionally includes the exact
    lower edge 0) -- a value exactly on a decade boundary (e.g. 90.0)
    belongs to the LOWER bucket ("80-90", not "90-100")."""
    return pd.cut(rs, bins=_COARSE_BINS, labels=_COARSE_LABELS, include_lowest=True, right=True)


def bucket_fine_above_80(rs: pd.Series) -> pd.Series:
    """Fixed-width 2.5-point bins over [80, 100] only. Rows with rs < 80
    (or NaN) fall outside every bin and are excluded (categorical NaN).
    Same right-closed boundary convention as bucket_coarse_deciles."""
    return pd.cut(rs, bins=_FINE_BINS, labels=_FINE_LABELS, include_lowest=True, right=True)


def compute_bucket_stats(df: pd.DataFrame, bucket_col: str, horizon: int) -> pd.DataFrame:
    """One row per bucket value present in df[bucket_col] (categorical
    dtype — empty buckets are NOT silently dropped, they appear with n=0
    and NaN stats, so a sparse/absent region of the RS scale stays visible
    rather than disappearing from the table)."""
    g = df.groupby(bucket_col, observed=False)

    out = pd.DataFrame({"n": g.size()})
    for col, label in (
        (f"mfe_pct_{horizon}d", "mfe_pct"),
        (f"mae_pct_{horizon}d", "mae_pct"),
        (f"forward_return_close_{horizon}d", "forward_return"),
    ):
        out[f"{label}_median"] = g[col].median()
        out[f"{label}_mean"] = g[col].mean()

    for th in PCT_THRESHOLDS:
        col = f"reached_plus_{th}pct_{horizon}d"
        out[f"prob_plus_{th}pct"] = g[col].mean()
    for mult in ATR_MULTIPLES:
        col = f"reached_{mult}atr_{horizon}d"
        out[f"prob_{mult}atr"] = g[col].mean()

    out.index.name = "bucket"
    return out.reset_index()


def build_rs_benchmark_report(stock_features_daily: pd.DataFrame,
                               stock_outcomes_daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Returns a flat dict keyed "{rs_horizon}__{bucket_scheme}__{horizon}d"
    -> bucket-stats DataFrame, for every combination of the 6 RS horizons x
    2 bucket schemes x 3 outcome horizons (36 tables). Rows are restricted
    to the ELIGIBLE universe (RS percentiles are only meaningful there —
    non-eligible rows carry NaN RS by construction, see features/technical.py)."""
    required_feat = {"date", "ticker", "eligible", *RS_HORIZONS}
    missing_feat = required_feat - set(stock_features_daily.columns)
    if missing_feat:
        raise ValueError(f"stock_features_daily missing required columns: {missing_feat}")

    merged = stock_features_daily.merge(stock_outcomes_daily, on=["date", "ticker"], how="inner")
    merged = merged[merged["eligible"].astype(bool)].copy()

    tables: dict[str, pd.DataFrame] = {}
    for rs_col in RS_HORIZONS:
        coarse = bucket_coarse_deciles(merged[rs_col])
        fine = bucket_fine_above_80(merged[rs_col])
        for scheme_name, bucket_series in (("coarse", coarse), ("fine_above_80", fine)):
            work = merged.copy()
            work["bucket"] = bucket_series
            for horizon in OUTCOME_HORIZONS_DAYS:
                key = f"{rs_col}__{scheme_name}__{horizon}d"
                tables[key] = compute_bucket_stats(work, "bucket", horizon)

    return tables
