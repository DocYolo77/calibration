"""2024 RS x ATR Extension Study (Phase 2, stock track): a cross-tabulation
of RS-bucket x ATR-Extension-bucket against the same 5D/10D/20D outcome
metrics as reports/rs_benchmark.py, for a single benchmark year at a time.

Question this module answers descriptively (no rule/threshold chosen):
"Wann ist hohe Relative Strength produktiv, und wann ist eine Aktie trotz
hoher RS bereits zu weit gelaufen?" — i.e. does ATR Extension add
information ON TOP OF RS alone.

ATR Extension is the EXISTING canonical definition (features/technical.py::
add_atr_extension, config/features.yaml `atr` block) — this module reads
the already-computed `atr_extension` column and never recomputes it:

    gain_from_sma50_pct = (close - sma50) / sma50 * 100
    atr_extension        = gain_from_sma50_pct / atr_pct

Two bucket schemes, both FIXED value-range bins, LEFT-CLOSED ([lo, hi)) so
the literal bucket labels below read exactly as written — a value on a
boundary belongs to the bucket whose name it satisfies (e.g. RS=80.0 is
"80-90", not "<80"; ATR-Extension=10.0 is ">=10", not "8-10"):

  - RS (cross-matrix rows): "<80", "80-90", "90-95", "95-97.5", "97.5-100"
    — coarser below RS80, finer at the top, matching the study's focus on
    "how far is too far" among already-strong names.
  - ATR Extension (cross-matrix columns): "<0", "0-2", "2-4", "4-6", "6-8",
    "8-10", ">=10".

For every (RS horizon, outcome horizon) combination, THREE tables share the
same 14 descriptive metrics (median/mean/p75/p90 MFE, median/mean MAE,
median/mean forward return, 4 expansion-probability shares, 2 race-outcome
shares — identical set to reports/rs_benchmark.py::compute_bucket_stats,
reused directly, including its conservative same-day tie-break for the race
outcomes):

  1. cross_matrix: one row per (RS bucket, ATR-Extension bucket) cell.
  2. rs_bucket_baseline: one row per RS bucket, ATR-Extension-agnostic
     (the SAME population as the cross matrix's row totals).
  3. universe_baseline: ONE row per outcome horizon, over the WHOLE
     eligible 2024 universe — RS-horizon-agnostic (identical across all 6
     RS horizons' studies), the widest reference point.

The cross_matrix table carries baseline_* and delta_vs_* columns (computed
against both baselines, for every one of the 14 metrics) so each cell is
self-contained — no separate join needed to see whether ATR Extension adds
information beyond RS alone. Empty/small cells are NEVER merged or dropped
(observed=False in compute_bucket_stats) — sample size (`n`) is always the
first thing to check before reading a cell's other numbers.

This module MUST NOT select a threshold, flag a "best"/"sweet-spot" cell,
or a "best" RS horizon — purely descriptive, same scope discipline as
reports/rs_benchmark.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from yolo_calibration.reports.rs_benchmark import (
    ATR_MULTIPLES,
    OUTCOME_HORIZONS_DAYS,
    PCT_THRESHOLDS,
    RACE_PAIRS,
    RS_HORIZONS,
    compute_bucket_stats,
)

METRIC_COLS = (
    ["median_mfe_pct", "mean_mfe_pct", "p75_mfe_pct", "p90_mfe_pct",
     "median_mae_pct", "mean_mae_pct",
     "median_forward_return", "mean_forward_return"]
    + [f"reached_plus_{th}pct_share" for th in PCT_THRESHOLDS]
    + [f"reached_{mult}atr_share" for mult in ATR_MULTIPLES]
    + [out_field for out_field, _ in RACE_PAIRS]
)

RS_CROSS_BINS = [0, 80, 90, 95, 97.5, 100 + 1e-6]
RS_CROSS_LABELS = ["<80", "80-90", "90-95", "95-97.5", "97.5-100"]

ATR_EXT_BINS = [-np.inf, 0, 2, 4, 6, 8, 10, np.inf]
ATR_EXT_LABELS = ["<0", "0-2", "2-4", "4-6", "6-8", "8-10", ">=10"]

# Transparency threshold only — cells below this are still returned in
# full, just additionally listed in build_rs_atr_extension_study's summary
# so a reader notices a thin cell before trusting its statistics.
SMALL_SAMPLE_THRESHOLD = 30


def bucket_rs_cross_matrix(rs: pd.Series) -> pd.Series:
    """Left-closed [lo, hi) bins so "<80" excludes 80.0 and "97.5-100"
    includes the maximum possible percentile rank, 100.0."""
    return pd.cut(rs, bins=RS_CROSS_BINS, labels=RS_CROSS_LABELS, right=False)


def bucket_atr_extension(atr_extension: pd.Series) -> pd.Series:
    """Left-closed [lo, hi) bins so "<0" excludes 0.0 and ">=10" includes
    10.0 and everything above."""
    return pd.cut(atr_extension, bins=ATR_EXT_BINS, labels=ATR_EXT_LABELS, right=False)


def _add_baseline_deltas(cross: pd.DataFrame, rs_bucket_baseline: pd.DataFrame,
                          universe_row: pd.Series) -> pd.DataFrame:
    out = cross.copy()
    for m in METRIC_COLS:
        out[f"baseline_universe_{m}"] = universe_row[m]
        out[f"delta_vs_universe_{m}"] = out[m] - universe_row[m]

    # NOTE: out["rs_bucket"] is categorical dtype. Series.map on a
    # categorical delegates to Categorical.map, which -- when the mapping
    # happens to be one-to-one over the (possibly mostly-NaN, for sparsely
    # populated buckets) mapped values -- returns a Categorical result
    # instead of casting to the mapped values' own dtype, which breaks
    # arithmetic (`out[m] - mapped` raises TypeError: dtype category
    # cannot perform the numpy op subtract). Casting to object dtype first
    # routes through plain (non-categorical) Series.map, sidestepping that
    # dtype-preserving special case entirely; the explicit astype(float)
    # after guarantees a numeric result regardless.
    rsb = rs_bucket_baseline.set_index("rs_bucket")
    for m in METRIC_COLS:
        mapped = out["rs_bucket"].astype(object).map(rsb[m]).astype(float)
        out[f"baseline_rs_bucket_{m}"] = mapped
        out[f"delta_vs_rs_bucket_{m}"] = out[m] - mapped
    return out


def build_rs_atr_extension_study(stock_features_daily: pd.DataFrame,
                                  stock_outcomes_daily: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """Returns ({"cross_matrix", "rs_bucket_baseline", "universe_baseline"}, summary)."""
    required = {"date", "ticker", "eligible", "atr_extension", *RS_HORIZONS}
    missing = required - set(stock_features_daily.columns)
    if missing:
        raise ValueError(f"stock_features_daily missing required columns: {missing}")

    merged = stock_features_daily.merge(stock_outcomes_daily, on=["date", "ticker"], how="inner")
    merged = merged[merged["eligible"].astype(bool)].copy()
    merged["atr_bucket"] = bucket_atr_extension(merged["atr_extension"])

    # Whole-eligible-universe baseline: RS-horizon-agnostic by construction
    # (computed once here, reused identically across all 6 RS horizons below).
    uni_work = merged.assign(_all="ALL")
    universe_rows = []
    for h in OUTCOME_HORIZONS_DAYS:
        row = compute_bucket_stats(uni_work, "_all", h).drop(columns=["bucket"])
        row.insert(0, "outcome_horizon", h)
        universe_rows.append(row)
    universe_baseline = pd.concat(universe_rows, ignore_index=True)
    universe_by_h = {
        h: universe_baseline.loc[universe_baseline["outcome_horizon"] == h].iloc[0]
        for h in OUTCOME_HORIZONS_DAYS
    }

    cross_frames: list[pd.DataFrame] = []
    rs_bucket_frames: list[pd.DataFrame] = []
    coverage: dict[str, dict] = {}

    for rs_col in RS_HORIZONS:
        work = merged.copy()
        work["rs_bucket"] = bucket_rs_cross_matrix(work[rs_col])
        n_valid = int(work[rs_col].notna().sum())
        coverage[rs_col] = {
            "n": n_valid,
            "pct": round(100.0 * n_valid / len(merged), 2) if len(merged) else 0.0,
        }

        for h in OUTCOME_HORIZONS_DAYS:
            cross_h = compute_bucket_stats(work, ["rs_bucket", "atr_bucket"], h)
            rs_bucket_h = compute_bucket_stats(work, "rs_bucket", h).rename(columns={"bucket": "rs_bucket"})

            cross_h = _add_baseline_deltas(cross_h, rs_bucket_h, universe_by_h[h])
            cross_h.insert(0, "outcome_horizon", h)
            cross_h.insert(0, "rs_horizon", rs_col)
            cross_frames.append(cross_h)

            rs_bucket_out = rs_bucket_h.copy()
            rs_bucket_out.insert(0, "outcome_horizon", h)
            rs_bucket_out.insert(0, "rs_horizon", rs_col)
            rs_bucket_frames.append(rs_bucket_out)

    cross_matrix = pd.concat(cross_frames, ignore_index=True)
    rs_bucket_baseline = pd.concat(rs_bucket_frames, ignore_index=True)

    small_cells = cross_matrix.loc[
        cross_matrix["n"] < SMALL_SAMPLE_THRESHOLD,
        ["rs_horizon", "outcome_horizon", "rs_bucket", "atr_bucket", "n"],
    ]
    summary = {
        "n_eligible_2024_total": int(len(merged)),
        "rs_horizon_coverage": coverage,
        "small_sample_threshold": SMALL_SAMPLE_THRESHOLD,
        "small_sample_cell_count": int(len(small_cells)),
        "small_sample_cells": small_cells.to_dict(orient="records"),
        "no_automatic_selection": True,
    }
    return {
        "cross_matrix": cross_matrix,
        "rs_bucket_baseline": rs_bucket_baseline,
        "universe_baseline": universe_baseline,
    }, summary
