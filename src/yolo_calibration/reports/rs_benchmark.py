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

Per (RS horizon, bucket, outcome horizon): sample size, median/mean/p75/p90
of mfe_pct, median/mean of mae_pct and forward_return, the hit-probability
of each of reached_plus_5pct/10pct/2atr/3atr, and the two "race" outcome
probabilities P(+5% before -5%) and P(+10% before -5%) (see
outcomes/build_outcomes.py — the latter is an asymmetric threshold pair,
distinguishing genuine momentum-expansion quality from plain volatility,
which raw MFE alone can conflate). Every number here is a plain descriptive
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
# (output_field_suffix, source_column_infix) -- source columns are
# outcomes/build_outcomes.py's reached_plus_{up}_before_minus_{down}_{H}d.
# Fixed, explicit pair list (not derived from config) since these two
# specific race outcomes were named directly in the report requirements.
RACE_PAIRS = (
    ("reached_plus_5_before_minus_5_share", "reached_plus_5_before_minus_5"),
    ("reached_plus_10_before_minus_5_share", "reached_plus_10_before_minus_5"),
)

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

    mfe_col = f"mfe_pct_{horizon}d"
    out["median_mfe_pct"] = g[mfe_col].median()
    out["mean_mfe_pct"] = g[mfe_col].mean()
    out["p75_mfe_pct"] = g[mfe_col].quantile(0.75)
    out["p90_mfe_pct"] = g[mfe_col].quantile(0.90)

    mae_col = f"mae_pct_{horizon}d"
    out["median_mae_pct"] = g[mae_col].median()
    out["mean_mae_pct"] = g[mae_col].mean()

    fwd_col = f"forward_return_close_{horizon}d"
    out["median_forward_return"] = g[fwd_col].median()
    out["mean_forward_return"] = g[fwd_col].mean()

    for th in PCT_THRESHOLDS:
        col = f"reached_plus_{th}pct_{horizon}d"
        out[f"reached_plus_{th}pct_share"] = g[col].mean()
    for mult in ATR_MULTIPLES:
        col = f"reached_{mult}atr_{horizon}d"
        out[f"reached_{mult}atr_share"] = g[col].mean()

    for out_field, source_infix in RACE_PAIRS:
        col = f"{source_infix}_{horizon}d"
        if col not in df.columns:
            raise ValueError(
                f"Missing race-outcome column '{col}' — stock_outcomes_daily must be built with "
                f"config/features.yaml outcome_thresholds.asymmetric_race_pairs including the pair "
                f"this field derives from (see outcomes/build_outcomes.py)."
            )
        out[out_field] = g[col].mean()

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
