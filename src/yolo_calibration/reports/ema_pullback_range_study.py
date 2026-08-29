"""EMA Pullback Range Study (Phase 2, stock track): calibrates ONLY the
future dashboard labels "EMA10 Pullback" / "EMA20 Pullback" for a single
benchmark year. Answers, separately for EMA10 and EMA20, separately for
RS1W/1M/3M/6M (never averaged or combined into a composite RS):

    "At what percentage distance to EMA10 / EMA20 do strong stocks
    typically sit before a new productive push begins within the next
    trading days?"

This is NOT the same question rs_atr_extension_study.py or
opportunity_state_study.py asked -- those looked at productivity BY
EMA-distance bucket, including previously-extended ("repeat offender")
setups. This module explicitly EXCLUDES Resetting rows (the existing
frozen Candidate Rule, reports/opportunity_state_candidate_rules.py::
classify_candidate_state) from every table, because Resetting is already
its own history-dependent state -- mixing it back in here would blur a
normal pre-push pullback with a post-extension reset. Resetting is
counted and reported, never silently dropped. Extended
(atr_extension >= 8) rows are the opposite: NEVER excluded here, only
separately broken out (module's build_ema_pullback_range_study output
`extended_among_successful_pushes`) -- the question of whether far-above-
EMA starts still produce a relevant share of successful pushes must stay
visible, not be quietly filtered out of the pullback distribution.

Population: the eligible universe, restricted to RS >= HIGH_RS_THRESHOLD
(reports/opportunity_state_study.py's existing 90th-percentile cutoff,
reused verbatim -- not a new number), reported both as that single
combined slice ("RS>=90") AND separately for its three sub-buckets
(90-95, 95-97.5, 97.5-100, from reports/rs_atr_extension_study.py's
existing RS-bucket scheme) -- sub-buckets are never merged just to
inflate n; thin ones are flagged (n<30), not hidden.

"A productive push" (module docstring, not a Resetting/Extended
condition) = reached_plus_5_before_minus_5_{h}d is True at some outcome
horizon h (the PRIMARY definition, h=10 trading days -- a fresh swing
push, not an intraday event this daily-bar pipeline could ever observe
directly). reached_plus_10_before_minus_5_{h}d is tracked as a SEPARATE,
stricter secondary push definition -- the two are never combined into one
score (spec section 5). 5D/20D are reported as sensitivity checks
alongside the primary 10D.

Four kinds of output, per EMA (10/20) and RS horizon:

  1. push_distance_distribution -- among successful-push rows only, the
     percentile distribution (P2.5..P97.5) of that row's OWN EMA-distance
     at the signal day, plus the derived (non-forced-symmetric) central
     80%/90%/95% ranges (P10-P90 / P5-P95 / P2.5-P97.5). This is the
     direct empirical answer to "where do successful pushes start."
  2. distance_bucket_stats -- the standard 14-metric descriptive
     aggregation (compute_bucket_stats, reused verbatim) over 14 FIXED,
     literal EMA-distance buckets (not derived from question 1's
     percentiles) -- shows whether narrow bands are also individually
     productive, not just numerically common.
  3. candidate_comparison -- reports/opportunity_state_candidate_rules.py's
     existing "-5% to +5%" reference range, PLUS a transparent grid of
     round-number alternative ranges (compare_pullback_range, a fully
     generic function taking an explicit lo/hi -- there is no code path
     that picks a "winner" among them; a human chooses from the visible
     table, matching this codebase's established never-auto-select-best
     discipline). Scoped to RS>=90 and the primary 10D horizon only, to
     keep the grid tractable -- see module docstring section above for
     the finer per-sub-bucket/per-horizon detail available in tables 1/2.
  4. extended_among_successful_pushes -- a small, separate breakdown
     (not a filter) of how many successful-push observations were
     ALSO Extended at the signal day.

No candidate range is selected, frozen, or written back into
opportunity_state_candidate_rules.py's thresholds by this module --
that remains an explicit, separate, human-approved step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from yolo_calibration.reports.opportunity_state_candidate_rules import (
    EXTENDED_ATR_THRESHOLD,
    classify_candidate_state,
)
from yolo_calibration.reports.opportunity_state_study import (
    HIGH_RS_THRESHOLD,
    RS_HORIZONS,
    SMALL_SAMPLE_THRESHOLD,
    prepare_eligible_year_rows,
)
from yolo_calibration.reports.rs_atr_extension_study import bucket_rs_cross_matrix
from yolo_calibration.reports.rs_benchmark import OUTCOME_HORIZONS_DAYS, compute_bucket_stats

# Left-closed [lo, hi), literal per spec section 7.
EMA_FINE_BINS = [-np.inf, -10, -7.5, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7.5, 10, np.inf]
EMA_FINE_LABELS = [
    "<-10%", "-10 to -7.5%", "-7.5 to -5%", "-5 to -3%", "-3 to -2%", "-2 to -1%", "-1 to 0%",
    "0 to +1%", "+1 to +2%", "+2 to +3%", "+3 to +5%", "+5 to +7.5%", "+7.5 to +10%", ">=+10%",
]

PERCENTILES = (2.5, 5, 10, 25, 50, 75, 90, 95, 97.5)
PERCENTILE_COL_NAMES = {2.5: "p2_5", 5: "p5", 10: "p10", 25: "p25", 50: "median",
                         75: "p75", 90: "p90", 95: "p95", 97.5: "p97_5"}

PUSH_TYPES = (
    ("plus5_before_minus5", "reached_plus_5_before_minus_5"),
    ("plus10_before_minus5", "reached_plus_10_before_minus_5"),
)

V1_DASHBOARD_RANGE = ("v1_dashboard_-5_to_+5", -5.0, 5.0)
# A transparent grid of round-number alternatives -- NOT a search that
# picks a winner (compare_pullback_range never ranks or flags a "best"
# row); every combination is computed and shown, a human reads it.
CANDIDATE_GRID_LO = (-8, -7, -6, -5, -4, -3, -2)
CANDIDATE_GRID_HI = (2, 3, 4, 5, 6)


def bucket_ema_distance_fine(distance_pct: pd.Series) -> pd.Series:
    return pd.cut(distance_pct, bins=EMA_FINE_BINS, labels=EMA_FINE_LABELS, right=False)


def _rs_population_masks(work: pd.DataFrame, rs_col: str) -> dict[str, pd.Series]:
    rs_bucket = bucket_rs_cross_matrix(work[rs_col])
    return {
        "RS>=90": work[rs_col] >= HIGH_RS_THRESHOLD,
        "90-95": rs_bucket == "90-95",
        "95-97.5": rs_bucket == "95-97.5",
        "97.5-100": rs_bucket == "97.5-100",
    }


def _percentile_row(distances: pd.Series) -> dict:
    n = int(distances.notna().sum())
    row = {"n": n}
    if n == 0:
        for p in PERCENTILES:
            row[PERCENTILE_COL_NAMES[p]] = np.nan
    else:
        q = distances.quantile([p / 100.0 for p in PERCENTILES])
        for p in PERCENTILES:
            row[PERCENTILE_COL_NAMES[p]] = float(q.loc[p / 100.0])
    row["range80_lo"], row["range80_hi"] = row.get("p10"), row.get("p90")
    row["range90_lo"], row["range90_hi"] = row.get("p5"), row.get("p95")
    row["range95_lo"], row["range95_hi"] = row.get("p2_5"), row.get("p97_5")
    return row


def build_push_distance_distribution(work: pd.DataFrame, ema: str, rs_col: str) -> pd.DataFrame:
    """`work` must already have Resetting rows excluded for this `ema`
    (see build_ema_pullback_range_study) and a `distance_ema{ema}_pct`
    column. One row per (rs_population_slice, outcome_horizon, push_type)
    -- the percentile distribution of successful-push rows' OWN
    EMA-distance, among ONLY the rows meeting that push definition."""
    dist_col = f"distance_ema{ema}_pct"
    masks = _rs_population_masks(work, rs_col)
    rows = []
    for slice_name, slice_mask in masks.items():
        for h in OUTCOME_HORIZONS_DAYS:
            for push_key, push_prefix in PUSH_TYPES:
                push_col = f"{push_prefix}_{h}d"
                success_mask = slice_mask & (work[push_col] == 1.0)
                row = _percentile_row(work.loc[success_mask, dist_col])
                row.update({"rs_population_slice": slice_name, "outcome_horizon": h, "push_type": push_key})
                rows.append(row)
    cols = ["rs_population_slice", "outcome_horizon", "push_type", "n"] + \
           [PERCENTILE_COL_NAMES[p] for p in PERCENTILES] + \
           ["range80_lo", "range80_hi", "range90_lo", "range90_hi", "range95_lo", "range95_hi"]
    return pd.DataFrame(rows)[cols]


def build_distance_bucket_stats(work: pd.DataFrame, ema: str, rs_col: str) -> pd.DataFrame:
    """Standard 14-metric descriptive aggregation over the 14 FIXED
    EMA-distance buckets, per rs_population_slice and outcome horizon."""
    dist_col = f"distance_ema{ema}_pct"
    masks = _rs_population_masks(work, rs_col)
    frames = []
    for slice_name, slice_mask in masks.items():
        sliced = work.loc[slice_mask].copy()
        sliced["distance_bucket"] = bucket_ema_distance_fine(sliced[dist_col])
        for h in OUTCOME_HORIZONS_DAYS:
            stats = compute_bucket_stats(sliced, "distance_bucket", h)
            stats.insert(0, "outcome_horizon", h)
            stats.insert(0, "rs_population_slice", slice_name)
            frames.append(stats)
    return pd.concat(frames, ignore_index=True)


def compare_pullback_range(work: pd.DataFrame, ema: str, rs_col: str, lo: float, hi: float,
                            label: str, is_v1: bool) -> dict:
    """Fully generic: takes an EXPLICIT (lo, hi) -- there is no threshold
    baked into this function's logic, so it cannot itself "select" a
    range. Scoped to the RS>=90 slice and the primary 10D outcome
    horizon (see module docstring). Inclusive bounds [lo, hi], matching
    how a dashboard range like "-5% to +5%" literally reads."""
    dist_col = f"distance_ema{ema}_pct"
    strong = work[work[rs_col] >= HIGH_RS_THRESHOLD]
    in_range = strong[(strong[dist_col] >= lo) & (strong[dist_col] <= hi)]

    out = {
        "range_label": label, "range_lo": lo, "range_hi": hi, "is_v1_reference": is_v1,
        "n_in_range": int(len(in_range)),
    }
    for push_key, push_prefix in PUSH_TYPES:
        push_col = f"{push_prefix}_10d"
        total_success = int((strong[push_col] == 1.0).sum())
        success_in_range = int((in_range[push_col] == 1.0).sum())
        out[f"n_successful_{push_key}_total"] = total_success
        out[f"share_of_successful_{push_key}_in_range"] = (
            round(100.0 * success_in_range / total_success, 2) if total_success else np.nan
        )
        out[f"{push_key}_share_in_range"] = float(in_range[push_col].mean()) if len(in_range) else np.nan

    for col, name in (("mfe_pct_10d", "median_mfe_pct"), ("mae_pct_10d", "median_mae_pct"),
                       ("forward_return_close_10d", "median_forward_return")):
        out[name] = float(in_range[col].median()) if len(in_range) else np.nan
    return out


def build_candidate_comparison_table(work: pd.DataFrame, ema: str) -> pd.DataFrame:
    rows = []
    for rs_col in RS_HORIZONS:
        label, lo, hi = V1_DASHBOARD_RANGE
        row = compare_pullback_range(work, ema, rs_col, lo, hi, label, is_v1=True)
        row["rs_horizon"] = rs_col
        rows.append(row)
        for grid_lo in CANDIDATE_GRID_LO:
            for grid_hi in CANDIDATE_GRID_HI:
                lbl = f"{grid_lo:+d}% to {grid_hi:+d}%"
                row = compare_pullback_range(work, ema, rs_col, float(grid_lo), float(grid_hi), lbl, is_v1=False)
                row["rs_horizon"] = rs_col
                rows.append(row)
    df = pd.DataFrame(rows)
    front = ["rs_horizon", "range_label", "range_lo", "range_hi", "is_v1_reference", "n_in_range"]
    return df[front + [c for c in df.columns if c not in front]]


def _extended_breakdown(work: pd.DataFrame, ema: str, rs_col: str) -> dict:
    strong = work[work[rs_col] >= HIGH_RS_THRESHOLD]
    push_col = "reached_plus_5_before_minus_5_10d"
    successful = strong[strong[push_col] == 1.0]
    n_total = int(len(successful))
    n_extended = int((successful["atr_extension"] >= EXTENDED_ATR_THRESHOLD).sum())
    return {
        "n_successful_pushes_10d": n_total,
        "n_extended_among_them": n_extended,
        "pct_extended_among_them": round(100.0 * n_extended / n_total, 2) if n_total else 0.0,
    }


def build_ema_pullback_range_study(
    stock_features_daily: pd.DataFrame, stock_outcomes_daily: pd.DataFrame, year: int,
) -> tuple[dict[str, pd.DataFrame], dict]:
    merged = prepare_eligible_year_rows(stock_features_daily, stock_outcomes_daily, year)

    resetting_exclusion = {}
    extended_breakdown = {}
    distribution_frames = {"10": [], "20": []}
    bucket_frames = {"10": [], "20": []}

    for ema in ("10", "20"):
        state_col = f"candidate_state_ema{ema}"
        merged[state_col] = classify_candidate_state(merged, ema)
        is_resetting = merged[state_col] == "Resetting"
        work = merged.loc[~is_resetting].copy()
        resetting_exclusion[f"ema{ema}"] = {
            "n_resetting_excluded": int(is_resetting.sum()),
            "n_remaining": int(len(work)),
        }

        for rs_col in RS_HORIZONS:
            dist = build_push_distance_distribution(work, ema, rs_col)
            dist.insert(0, "rs_horizon", rs_col)
            distribution_frames[ema].append(dist)

            buckets = build_distance_bucket_stats(work, ema, rs_col)
            buckets.insert(0, "rs_horizon", rs_col)
            bucket_frames[ema].append(buckets)

            extended_breakdown[f"ema{ema}__{rs_col}"] = _extended_breakdown(work, ema, rs_col)

    comparison_frames = []
    for ema in ("10", "20"):
        state_col = f"candidate_state_ema{ema}"
        work = merged.loc[merged[state_col] != "Resetting"].copy()
        cmp_table = build_candidate_comparison_table(work, ema)
        cmp_table.insert(0, "ema", ema)
        comparison_frames.append(cmp_table)

    tables = {
        "ema10_push_distance_distribution": pd.concat(distribution_frames["10"], ignore_index=True),
        "ema20_push_distance_distribution": pd.concat(distribution_frames["20"], ignore_index=True),
        "ema10_push_distance_buckets": pd.concat(bucket_frames["10"], ignore_index=True),
        "ema20_push_distance_buckets": pd.concat(bucket_frames["20"], ignore_index=True),
        "ema_pullback_candidate_comparison": pd.concat(comparison_frames, ignore_index=True),
    }

    small_cells = {}
    total_small = 0
    for key, table in tables.items():
        if "n" in table.columns:
            n_col = table["n"]
        elif "n_in_range" in table.columns:
            n_col = table["n_in_range"]
        else:
            small_cells[key] = 0
            continue
        cnt = int((n_col < SMALL_SAMPLE_THRESHOLD).sum())
        small_cells[key] = cnt
        total_small += cnt

    rs_coverage = {}
    strong_rs_coverage = {}
    missing_values = {
        "distance_ema10_pct": int(merged["distance_ema10_pct"].isna().sum()),
        "distance_ema20_pct": int(merged["distance_ema20_pct"].isna().sum()),
        "atr_extension": int(merged["atr_extension"].isna().sum()),
    }
    for rs_col in RS_HORIZONS:
        n_valid = int(merged[rs_col].notna().sum())
        rs_coverage[rs_col] = {"n": n_valid, "pct": round(100.0 * n_valid / len(merged), 2) if len(merged) else 0.0}
        n_strong = int((merged[rs_col] >= HIGH_RS_THRESHOLD).sum())
        strong_rs_coverage[rs_col] = {
            "n": n_strong, "pct_of_eligible": round(100.0 * n_strong / len(merged), 2) if len(merged) else 0.0,
        }
        missing_values[rs_col] = int(merged[rs_col].isna().sum())

    summary = {
        "year": year,
        "n_eligible_total": int(len(merged)),
        "high_rs_threshold": HIGH_RS_THRESHOLD,
        "v1_dashboard_range": {"lo": V1_DASHBOARD_RANGE[1], "hi": V1_DASHBOARD_RANGE[2]},
        "rs_horizon_coverage": rs_coverage,
        "strong_rs_coverage": strong_rs_coverage,
        "resetting_exclusion": resetting_exclusion,
        "extended_among_successful_pushes": extended_breakdown,
        "missing_values": missing_values,
        "small_sample_threshold": SMALL_SAMPLE_THRESHOLD,
        "small_sample_cell_count_by_table": small_cells,
        "small_sample_cell_count_total": total_small,
        "no_automatic_range_selection": True,
    }
    return tables, summary
