"""Phase 1 descriptive sanity-check reports (spec section 20).

These exist ONLY to sanity-check that data and relationships look
plausible (e.g. "does higher RS decile associate with better forward MFE,
in the expected direction"). They deliberately do NOT search for or emit
an "optimal threshold" — no rule selection happens here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from yolo_calibration.config import REPO_ROOT


def bucket_vs_outcomes(df: pd.DataFrame, bucket_col: str, outcome_cols: list[str],
                        n_buckets: int = 10, bucket_label: str | None = None) -> pd.DataFrame:
    """Group rows into n_buckets quantile buckets of bucket_col and report
    median/count of each outcome column per bucket. Purely descriptive."""
    label = bucket_label or bucket_col
    work = df[[bucket_col] + outcome_cols].dropna(subset=[bucket_col]).copy()
    if work.empty:
        return pd.DataFrame()
    try:
        work[f"{label}_bucket"] = pd.qcut(work[bucket_col], q=n_buckets, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    agg = work.groupby(f"{label}_bucket", observed=True)[outcome_cols].median()
    agg["n"] = work.groupby(f"{label}_bucket", observed=True).size()
    return agg.reset_index()


def generate_stock_sanity_reports(stock_features_daily: pd.DataFrame,
                                   stock_outcomes_daily: pd.DataFrame,
                                   out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    out_dir = out_dir or (REPO_ROOT / "reports" / "phase1_sanity_checks")
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = stock_features_daily.merge(stock_outcomes_daily, on=["date", "ticker"], how="inner")
    merged = merged[merged["eligible"] == True]  # noqa: E712

    outcome_cols_10d = ["mfe_pct_10d", "mae_pct_10d"]
    results = {}

    for bucket_col, label in [
        ("rs_percentile_1m", "rs_decile"),
        ("atr_extension", "atr_extension_bucket"),
        ("distance_ema20_pct", "ema20_distance_bucket"),
    ]:
        available = [c for c in outcome_cols_10d if c in merged.columns]
        table = bucket_vs_outcomes(merged, bucket_col, available, bucket_label=label)
        results[label] = table
        if not table.empty:
            table.to_csv(out_dir / f"{label}_vs_10d_mfe_mae.csv", index=False)

    return results


def generate_qqq_health_sanity_reports(qqq_health_daily: pd.DataFrame,
                                        qqq_health_outcomes_daily: pd.DataFrame,
                                        out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    out_dir = out_dir or (REPO_ROOT / "reports" / "phase1_sanity_checks")
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = qqq_health_daily.merge(qqq_health_outcomes_daily, on="date", how="inner")
    outcome_cols_10d = [c for c in [
        "median_forward_return_10d", "median_mfe_pct_10d", "median_mae_pct_10d",
        "share_positive_10d",
    ] if c in merged.columns]

    results = {}
    for bucket_col, label in [
        ("mco_z", "mco_z_decile"),
        ("mcsi_z", "mcsi_z_decile"),
        ("pct_above_ema21", "pct_above_ema21_bucket"),
        ("high_low_pct", "high_low_oscillator_bucket"),
    ]:
        if bucket_col not in merged.columns:
            continue
        table = bucket_vs_outcomes(merged, bucket_col, outcome_cols_10d, bucket_label=label)
        results[label] = table
        if not table.empty:
            table.to_csv(out_dir / f"{label}_vs_10d_environment_outcomes.csv", index=False)

    return results
