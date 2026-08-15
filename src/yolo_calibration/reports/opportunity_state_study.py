"""2024 Opportunity-State Study (Phase 2, stock track): how strong stocks
behave depending on their CURRENT technical structure (distance to EMA10 /
EMA20) and their PREVIOUS extension/leadership history ("repeat offender"
behavior), for a single benchmark year at a time.

This module produces the descriptive inputs a later, human-directed step
may use to define Opportunity States (Normal / Extended / Resetting) — it
does NOT define them itself. No threshold, "best" cell, or state
classification is computed here.

RS horizons are analyzed STRICTLY SEPARATELY (no averaging, no weighting,
no composite RS, no "best" horizon selection) for exactly the four this
study covers: RS1W, RS1M, RS3M, RS6M (config/features.yaml
`relative_strength.horizons_days`; a DELIBERATELY NARROWER set than
reports/rs_benchmark.py's six horizons — RS1D/RS12M are out of scope here).

Six tables, sharing the same 14 descriptive metrics as
reports/rs_atr_extension_study.py (reused via compute_bucket_stats,
including the conservative same-day race-outcome tie-break):

  1. rs_x_ema10 / 2. rs_x_ema20 — RS-bucket x EMA-distance-bucket, per RS
     horizon. Baselines: the full eligible universe, and the same RS
     bucket marginalized over EMA-distance (see reports/rs_atr_extension_study.py
     for the identical pattern, just swapping ATR-Extension-bucket for
     EMA-distance-bucket).
  3. prior_extension_x_ema10 / 4. prior_extension_x_ema20 — Previous-ATR-
     Extension-peak-bucket x current-EMA-distance-bucket, over the FULL
     eligible 2024 universe (no RS conditioning — this isolates the
     "previous extension" effect on its own). Baselines: the full
     universe, and the same current EMA-distance bucket marginalized over
     previous-extension history.
  5. repeat_offender_ema10 / 6. repeat_offender_ema20 — the SAME
     Previous-Extension x EMA-distance cross-tab as 3/4, but restricted
     PER RS HORIZON to rows where that horizon's RS percentile >=
     HIGH_RS_THRESHOLD (a population-definition cutoff reusing the
     existing top-RS-bucket boundary already used elsewhere in this
     codebase's RS-bucket scheme — not a tunable/optimized threshold).
     This is the direct "repeat offender" comparison: a currently-strong
     stock pulling back toward its EMA, WITH vs WITHOUT a recent strong-
     extension history. THREE baselines here (the extra one is the crux
     of the whole study): the full universe, the same RS>=HIGH_RS_THRESHOLD
     cohort marginalized over both bucket dimensions, and — the key
     comparison — the SAME current-EMA-distance bucket restricted to the
     "no prior extension" control group (previous-extension bucket "<4"),
     still within the same RS>=HIGH_RS_THRESHOLD cohort and RS horizon.
     Also carries `median_current_rs_in_cell`, a plain descriptive median
     of the row's own current RS percentile per cell — this is the
     concrete, measured answer to "does RS persist during a reset,"
     nothing inferred.

Previous-extension history (features/repeat_offender.py) is computed on
the FULL, unfiltered per-ticker feature history BEFORE any eligibility or
year filtering — atr_extension itself is computed for every ticker-day
regardless of eligibility (features/build_features.py), and restricting
the input before computing a trailing rolling window would silently
truncate genuine history into a fabricated one. Point-in-time discipline:
`year` rows are the ONLY output/analysis rows; every earlier calendar year
present in the input is used exclusively as trailing lookback (for RS3M/
6M and for the previous-extension windows) and never appears as an
outcome row itself. No later year may be present in the input at all.

Every number here is a plain descriptive aggregate — this module MUST NOT
select a threshold, flag a "best" bucket/cell, or classify a row into
Normal/Extended/Resetting. That decision-making is explicitly out of
scope (deferred to a later, human-directed Phase 2 step).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from yolo_calibration.features.repeat_offender import compute_prior_extension_history
from yolo_calibration.reports.rs_atr_extension_study import (
    RS_CROSS_BINS,
    RS_CROSS_LABELS,
    bucket_rs_cross_matrix,
)
from yolo_calibration.reports.rs_benchmark import (
    ATR_MULTIPLES,
    OUTCOME_HORIZONS_DAYS,
    PCT_THRESHOLDS,
    RACE_PAIRS,
    compute_bucket_stats,
)

RS_HORIZONS = ("rs_percentile_1w", "rs_percentile_1m", "rs_percentile_3m", "rs_percentile_6m")

# Left-closed [lo, hi) — same convention as reports/rs_atr_extension_study.py
# (a value on a boundary belongs to the bucket whose name it literally
# satisfies, e.g. -5.0 -> "-5% to -2%", 0.0 -> "0% to +2%", 10.0 -> ">=+10%").
EMA_DIST_BINS = [-np.inf, -5, -2, 0, 2, 5, 10, np.inf]
EMA_DIST_LABELS = ["<-5%", "-5% to -2%", "-2% to 0%", "0% to +2%", "+2% to +5%", "+5% to +10%", ">=+10%"]

PREV_EXT_BINS = [-np.inf, 4, 6, 8, 10, np.inf]
PREV_EXT_LABELS = ["<4", "4-6", "6-8", "8-10", ">=10"]

# Population-definition cutoff for the repeat-offender cohort — reuses the
# existing 90th-percentile RS-bucket boundary already used throughout this
# codebase's RS-bucket scheme (RS_CROSS_LABELS). Fixed so the study is
# reproducible and interpretable step-by-step (spec section 8: no giant
# automatic multi-dimensional threshold search) — NOT an optimized or
# tunable "leader" threshold.
HIGH_RS_THRESHOLD = 90.0

METRIC_COLS = (
    ["median_mfe_pct", "mean_mfe_pct", "p75_mfe_pct", "p90_mfe_pct",
     "median_mae_pct", "mean_mae_pct",
     "median_forward_return", "mean_forward_return"]
    + [f"reached_plus_{th}pct_share" for th in PCT_THRESHOLDS]
    + [f"reached_{mult}atr_share" for mult in ATR_MULTIPLES]
    + [out_field for out_field, _ in RACE_PAIRS]
)

SMALL_SAMPLE_THRESHOLD = 30


def bucket_ema_distance(distance_pct: pd.Series) -> pd.Series:
    return pd.cut(distance_pct, bins=EMA_DIST_BINS, labels=EMA_DIST_LABELS, right=False)


def bucket_previous_extension(peak_atr_extension: pd.Series) -> pd.Series:
    return pd.cut(peak_atr_extension, bins=PREV_EXT_BINS, labels=PREV_EXT_LABELS, right=False)


def _add_scalar_baseline(cross: pd.DataFrame, name: str, baseline_row: pd.Series) -> pd.DataFrame:
    out = cross.copy()
    for m in METRIC_COLS:
        out[f"baseline_{name}_{m}"] = baseline_row[m]
        out[f"delta_vs_{name}_{m}"] = out[m] - baseline_row[m]
    return out


def _add_marginal_baseline(cross: pd.DataFrame, name: str, baseline_table: pd.DataFrame, join_col: str) -> pd.DataFrame:
    """baseline_table has one row per value of join_col (e.g. one row per
    rs_bucket, marginalized over the OTHER cross-tab dimension). Joined
    onto `cross` via cross[join_col] -- see reports/rs_atr_extension_study.py
    for why the categorical .map() must go through object dtype first."""
    out = cross.copy()
    b = baseline_table.set_index(join_col)
    for m in METRIC_COLS:
        mapped = out[join_col].astype(object).map(b[m]).astype(float)
        out[f"baseline_{name}_{m}"] = mapped
        out[f"delta_vs_{name}_{m}"] = out[m] - mapped
    return out


def _universe_baseline_rows(df: pd.DataFrame) -> dict[int, pd.Series]:
    work = df.assign(_all="ALL")
    rows = {}
    for h in OUTCOME_HORIZONS_DAYS:
        row = compute_bucket_stats(work, "_all", h).drop(columns=["bucket"]).iloc[0]
        rows[h] = row
    return rows


def prepare_eligible_year_rows(stock_features_daily: pd.DataFrame, stock_outcomes_daily: pd.DataFrame,
                                year: int) -> pd.DataFrame:
    """Shared point-in-time prep, reused by reports/opportunity_state_candidate_rules.py:
    computes prior-extension history on the FULL unfiltered input, then
    restricts output rows to `year` only (see module docstring)."""
    required_feat = {"date", "ticker", "eligible", "atr_extension",
                      "distance_ema10_pct", "distance_ema20_pct", *RS_HORIZONS}
    missing = required_feat - set(stock_features_daily.columns)
    if missing:
        raise ValueError(f"stock_features_daily missing required columns: {missing}")

    # Prior-extension history needs the FULL, unfiltered per-ticker series
    # (see module docstring) -- computed BEFORE any eligibility/year
    # filtering, on whatever lookback rows the caller supplied.
    with_history = compute_prior_extension_history(stock_features_daily)

    year_mask = with_history["date"].dt.year == year
    if (with_history["date"].dt.year > year).any():
        raise ValueError(f"stock_features_daily contains rows after {year} -- this study is scoped to {year} only.")
    signal_rows = with_history.loc[year_mask].copy()

    merged = signal_rows.merge(stock_outcomes_daily, on=["date", "ticker"], how="inner")
    merged = merged[merged["eligible"].astype(bool)].copy()
    return merged


def build_opportunity_state_study(
    stock_features_daily: pd.DataFrame, stock_outcomes_daily: pd.DataFrame, year: int,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """`stock_features_daily` MUST include enough trailing history before
    `year`-01-01 for RS3M/6M (up to 126 trading days) and the
    previous-extension windows (config/features.yaml
    `prior_extension_history.peak_window_days`, 60 trading days) -- e.g.
    the full prior calendar year. Rows from `year` are the ONLY output
    rows; earlier rows are lookback-only and a later year raises."""
    merged = prepare_eligible_year_rows(stock_features_daily, stock_outcomes_daily, year)
    universe_by_h = _universe_baseline_rows(merged)

    merged["ema10_bucket"] = bucket_ema_distance(merged["distance_ema10_pct"])
    merged["ema20_bucket"] = bucket_ema_distance(merged["distance_ema20_pct"])
    merged["prev_ext_bucket"] = bucket_previous_extension(merged["atr_extension_peak"])

    rs_x_ema10_frames, rs_x_ema20_frames = [], []
    repeat_offender_ema10_frames, repeat_offender_ema20_frames = [], []
    rs_coverage: dict[str, dict] = {}
    high_rs_coverage: dict[str, dict] = {}

    for rs_col in RS_HORIZONS:
        work = merged.copy()
        work["rs_bucket"] = bucket_rs_cross_matrix(work[rs_col])
        n_valid = int(work[rs_col].notna().sum())
        rs_coverage[rs_col] = {"n": n_valid, "pct": round(100.0 * n_valid / len(merged), 2) if len(merged) else 0.0}

        cohort = work[work[rs_col] >= HIGH_RS_THRESHOLD].copy()
        high_rs_coverage[rs_col] = {
            "n": int(len(cohort)),
            "pct_of_eligible": round(100.0 * len(cohort) / len(merged), 2) if len(merged) else 0.0,
        }

        for h in OUTCOME_HORIZONS_DAYS:
            rs_bucket_h = compute_bucket_stats(work, "rs_bucket", h).rename(columns={"bucket": "rs_bucket"})

            for ema_col, frames_list in (("ema10_bucket", rs_x_ema10_frames), ("ema20_bucket", rs_x_ema20_frames)):
                cross = compute_bucket_stats(work, ["rs_bucket", ema_col], h)
                cross = _add_scalar_baseline(cross, "universe", universe_by_h[h])
                cross = _add_marginal_baseline(cross, "rs_bucket", rs_bucket_h, "rs_bucket")
                cross.insert(0, "outcome_horizon", h)
                cross.insert(0, "rs_horizon", rs_col)
                frames_list.append(cross)

            if len(cohort) == 0:
                continue
            cohort_row = compute_bucket_stats(cohort.assign(_all="ALL"), "_all", h).drop(columns=["bucket"]).iloc[0]

            for ema_col, frames_list in (("ema10_bucket", repeat_offender_ema10_frames),
                                          ("ema20_bucket", repeat_offender_ema20_frames)):
                cross = compute_bucket_stats(cohort, ["prev_ext_bucket", ema_col], h)
                control_h = compute_bucket_stats(
                    cohort[cohort["prev_ext_bucket"] == "<4"], ema_col, h
                ).rename(columns={"bucket": ema_col})

                cross = _add_scalar_baseline(cross, "universe", universe_by_h[h])
                cross = _add_scalar_baseline(cross, "rs_ge90_cohort", cohort_row)
                cross = _add_marginal_baseline(cross, "no_prior_extension_control", control_h, ema_col)

                median_rs = cohort.groupby(["prev_ext_bucket", ema_col], observed=False)[rs_col].median()
                cross = cross.set_index(["prev_ext_bucket", ema_col]).join(
                    median_rs.rename("median_current_rs_in_cell")
                ).reset_index()

                cross.insert(0, "outcome_horizon", h)
                cross.insert(0, "rs_horizon", rs_col)
                frames_list.append(cross)

    # Previous-Extension x EMA-distance over the FULL eligible universe (no
    # RS conditioning) -- computed ONCE, independent of RS horizon.
    prior_ext_x_ema10_frames, prior_ext_x_ema20_frames = [], []
    for h in OUTCOME_HORIZONS_DAYS:
        for ema_col, frames_list in (("ema10_bucket", prior_ext_x_ema10_frames),
                                      ("ema20_bucket", prior_ext_x_ema20_frames)):
            ema_bucket_h = compute_bucket_stats(merged, ema_col, h).rename(columns={"bucket": ema_col})
            cross = compute_bucket_stats(merged, ["prev_ext_bucket", ema_col], h)
            cross = _add_scalar_baseline(cross, "universe", universe_by_h[h])
            cross = _add_marginal_baseline(cross, "ema_bucket", ema_bucket_h, ema_col)
            cross.insert(0, "outcome_horizon", h)
            frames_list.append(cross)

    tables = {
        "rs_x_ema10": pd.concat(rs_x_ema10_frames, ignore_index=True),
        "rs_x_ema20": pd.concat(rs_x_ema20_frames, ignore_index=True),
        "prior_extension_x_ema10": pd.concat(prior_ext_x_ema10_frames, ignore_index=True),
        "prior_extension_x_ema20": pd.concat(prior_ext_x_ema20_frames, ignore_index=True),
        "repeat_offender_ema10": (
            pd.concat(repeat_offender_ema10_frames, ignore_index=True) if repeat_offender_ema10_frames
            else pd.DataFrame()
        ),
        "repeat_offender_ema20": (
            pd.concat(repeat_offender_ema20_frames, ignore_index=True) if repeat_offender_ema20_frames
            else pd.DataFrame()
        ),
    }

    small_cells = {}
    total_small = 0
    for key, table in tables.items():
        if table.empty or "n" not in table.columns:
            small_cells[key] = 0
            continue
        cnt = int((table["n"] < SMALL_SAMPLE_THRESHOLD).sum())
        small_cells[key] = cnt
        total_small += cnt

    n_prior_ext_valid = int(merged["atr_extension_peak"].notna().sum())
    n_partial_lookback = int((merged["n_valid_days_prior_peak_window"] <
                               merged["n_valid_days_prior_peak_window"].max()).sum()) if len(merged) else 0

    summary = {
        "year": year,
        "n_eligible_total": int(len(merged)),
        "rs_horizon_coverage": rs_coverage,
        "high_rs_threshold": HIGH_RS_THRESHOLD,
        "high_rs_coverage": high_rs_coverage,
        "prior_extension_coverage": {
            "n_with_valid_peak": n_prior_ext_valid,
            "pct_with_valid_peak": round(100.0 * n_prior_ext_valid / len(merged), 2) if len(merged) else 0.0,
            "n_with_partial_lookback_window": n_partial_lookback,
        },
        "small_sample_threshold": SMALL_SAMPLE_THRESHOLD,
        "small_sample_cell_count_by_table": small_cells,
        "small_sample_cell_count_total": total_small,
        "no_automatic_selection": True,
    }
    return tables, summary
