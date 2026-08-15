"""Opportunity-State Candidate Rules v1 (Phase 2, stock track): a FROZEN,
deliberately simple classification of every eligible DATE x TICKER row
into Normal / Extended / Resetting, derived from the descriptive findings
of reports/rs_atr_extension_study.py and reports/opportunity_state_study.py
-- and reusing bucket boundaries those two studies already established
(8 and 4 ATR-Extension, 0% EMA distance) rather than fitting new numbers.

This is the first module in this codebase that actually classifies rows.
That is intentional and explicitly authorized (unlike rs_atr_extension_study.py
and opportunity_state_study.py, which deliberately never do this) -- but the
classification is a CANDIDATE, not a production rule: no decimal-place
optimization, no per-call threshold override, and -- critically -- the SAME
frozen thresholds are used for the 2023 robustness check as for the 2024
report that produced them (config section 6: "Regeln einfrieren, exakt
unverändert auf 2023 anwenden, keine Schwellen anhand von 2023
nachoptimieren"). There is no code path in this module through which a
caller could pass a different threshold; recalibrating means editing the
constants below and re-reading this docstring's justification.

Candidate Rules v1
-------------------
Extended:   atr_extension >= EXTENDED_ATR_THRESHOLD (8.0)
            Current canonical ATR Extension (features/technical.py::
            add_atr_extension) at or above 8 -- reports/rs_atr_extension_study.py
            showed race-outcome reliability (P(+5% before -5%)) visibly
            lower in the "8-10" and ">=10" ATR-Extension buckets than in
            "<0"/"0-2" across every RS horizon and RS bucket tested, while
            MFE itself did not fall off -- i.e. still explosive, but less
            reliably tradeable. 8 is an EXISTING bucket boundary in both
            prior studies' bucket schemes (ATR_EXT_LABELS /
            PREV_EXT_LABELS), not a new number chosen for this rule.

Resetting:  (a previous strong-extension peak) AND (that extension has
            genuinely unwound) AND (price at or below the given EMA):
              atr_extension_peak            >= RESET_PEAK_ATR_THRESHOLD (8.0)
              atr_extension                  < RESET_CURRENT_ATR_THRESHOLD (4.0)
              distance_ema{10,20}_pct        <= RESET_EMA_DISTANCE_THRESHOLD (0.0)
            The repeat-offender history requirement IS the peak condition
            (reports/opportunity_state_study.py's repeat_offender_* tables
            showed the previous-extension effect on MFE/race-outcomes
            concentrated at peak>=8, strongest for RS1M, materially weaker
            by RS6M -- see that study's report). Reusing the SAME 8.0
            threshold as Extended keeps the rule internally consistent:
            "Resetting" means "was Extended, is not anymore." The 4.0
            "no longer extended" cutoff and the 0% "at/below the EMA"
            cutoff are both existing bucket edges (PREV_EXT_LABELS'
            "<4" bucket, EMA_DIST_LABELS' 0%/2% boundary), again not new
            numbers. EMA10 and EMA20 are two SEPARATE candidate
            classifications (candidate_state_ema10, candidate_state_ema20),
            never merged into one -- the two studies found EMA10 and EMA20
            pullbacks behave differently, and averaging them would hide
            exactly the distinction the study surfaced. A row lacking
            previous-extension history (atr_extension_peak is NaN --
            insufficient trailing data) can never satisfy Resetting; it can
            still be Extended (current-value-only) or Normal.

Normal:     everything else -- neither Extended nor Resetting under that
            EMA's definition. By construction (4.0 < 8.0) Extended and
            Resetting are mutually exclusive without needing an explicit
            "not Extended" guard.

RS horizons are analyzed STRICTLY SEPARATELY (RS1W/1M/3M/6M, same subset
as reports/opportunity_state_study.py) -- never averaged or combined into
a composite RS. For each RS horizon, outcome horizon, RS bucket and
candidate state, the same 14 descriptive metrics as the other Phase-2
studies are reported (compute_bucket_stats, including the conservative
same-day race-outcome tie-break), with TWO baselines: the full eligible
universe, and -- the one requested explicitly -- the SAME RS bucket with
no state filter (i.e. every state in that RS bucket combined). No "best"
state, RS horizon, or bucket is ever marked; this module still does not
pick a winner, it only classifies according to the frozen rule above and
reports the resulting descriptive numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from yolo_calibration.reports.opportunity_state_study import (
    RS_HORIZONS,
    SMALL_SAMPLE_THRESHOLD,
    prepare_eligible_year_rows,
)
from yolo_calibration.reports.rs_atr_extension_study import bucket_rs_cross_matrix
from yolo_calibration.reports.rs_benchmark import (
    ATR_MULTIPLES,
    OUTCOME_HORIZONS_DAYS,
    PCT_THRESHOLDS,
    RACE_PAIRS,
    compute_bucket_stats,
)

EXTENDED_ATR_THRESHOLD = 8.0
RESET_PEAK_ATR_THRESHOLD = 8.0
RESET_CURRENT_ATR_THRESHOLD = 4.0
RESET_EMA_DISTANCE_THRESHOLD = 0.0

CANDIDATE_STATES = ("Normal", "Extended", "Resetting")

METRIC_COLS = (
    ["median_mfe_pct", "mean_mfe_pct", "p75_mfe_pct", "p90_mfe_pct",
     "median_mae_pct", "mean_mae_pct",
     "median_forward_return", "mean_forward_return"]
    + [f"reached_plus_{th}pct_share" for th in PCT_THRESHOLDS]
    + [f"reached_{mult}atr_share" for mult in ATR_MULTIPLES]
    + [out_field for out_field, _ in RACE_PAIRS]
)


def classify_candidate_state(df: pd.DataFrame, ema: str) -> pd.Series:
    """`df` must carry atr_extension, atr_extension_peak (see
    features/repeat_offender.py), and distance_ema{ema}_pct. `ema` is
    literally "10" or "20" -- EMA10 and EMA20 resets are two separate
    classifications by design (see module docstring), never merged."""
    if ema not in ("10", "20"):
        raise ValueError(f"ema must be '10' or '20', got {ema!r}")

    current = df["atr_extension"]
    peak = df["atr_extension_peak"]
    dist = df[f"distance_ema{ema}_pct"]

    extended = current >= EXTENDED_ATR_THRESHOLD
    resetting = (peak >= RESET_PEAK_ATR_THRESHOLD) & (current < RESET_CURRENT_ATR_THRESHOLD) & (dist <= RESET_EMA_DISTANCE_THRESHOLD)

    raw = np.where(extended, "Extended", np.where(resetting, "Resetting", "Normal"))
    out = pd.Series(raw, index=df.index, dtype="object")
    out[current.isna()] = np.nan  # cannot classify at all without a current ATR-Extension reading
    return out.astype(pd.CategoricalDtype(categories=list(CANDIDATE_STATES)))


def _add_scalar_baseline(cross: pd.DataFrame, name: str, baseline_row: pd.Series) -> pd.DataFrame:
    out = cross.copy()
    for m in METRIC_COLS:
        out[f"baseline_{name}_{m}"] = baseline_row[m]
        out[f"delta_vs_{name}_{m}"] = out[m] - baseline_row[m]
    return out


def _add_marginal_baseline(cross: pd.DataFrame, name: str, baseline_table: pd.DataFrame, join_col: str) -> pd.DataFrame:
    out = cross.copy()
    b = baseline_table.set_index(join_col)
    for m in METRIC_COLS:
        mapped = out[join_col].astype(object).map(b[m]).astype(float)
        out[f"baseline_{name}_{m}"] = mapped
        out[f"delta_vs_{name}_{m}"] = out[m] - mapped
    return out


def _universe_baseline_rows(df: pd.DataFrame) -> dict[int, pd.Series]:
    work = df.assign(_all="ALL")
    return {
        h: compute_bucket_stats(work, "_all", h).drop(columns=["bucket"]).iloc[0]
        for h in OUTCOME_HORIZONS_DAYS
    }


def build_candidate_rules_report(
    stock_features_daily: pd.DataFrame, stock_outcomes_daily: pd.DataFrame, year: int,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Returns ({"candidate_states_by_rs_bucket_ema10", "...ema20"}, summary).
    Applies the FROZEN Candidate Rules v1 (module docstring) to every
    eligible `year` row, then cross-tabulates state x RS-bucket x outcome
    horizon, separately for each of RS1W/1M/3M/6M -- never averaged."""
    merged = prepare_eligible_year_rows(stock_features_daily, stock_outcomes_daily, year)
    for ema in ("10", "20"):
        merged[f"candidate_state_ema{ema}"] = classify_candidate_state(merged, ema)

    universe_by_h = _universe_baseline_rows(merged)

    frames = {"10": [], "20": []}
    rs_coverage: dict[str, dict] = {}

    for rs_col in RS_HORIZONS:
        work = merged.copy()
        work["rs_bucket"] = bucket_rs_cross_matrix(work[rs_col])
        n_valid = int(work[rs_col].notna().sum())
        rs_coverage[rs_col] = {"n": n_valid, "pct": round(100.0 * n_valid / len(merged), 2) if len(merged) else 0.0}

        for h in OUTCOME_HORIZONS_DAYS:
            rs_bucket_marginal_h = compute_bucket_stats(work, "rs_bucket", h).rename(columns={"bucket": "rs_bucket"})

            for ema in ("10", "20"):
                state_col = f"candidate_state_ema{ema}"
                cross = compute_bucket_stats(work, ["rs_bucket", state_col], h).rename(columns={state_col: "candidate_state"})
                cross = _add_scalar_baseline(cross, "universe", universe_by_h[h])
                cross = _add_marginal_baseline(cross, "rs_bucket", rs_bucket_marginal_h, "rs_bucket")
                cross.insert(0, "outcome_horizon", h)
                cross.insert(0, "rs_horizon", rs_col)
                frames[ema].append(cross)

    tables = {
        "candidate_states_by_rs_bucket_ema10": pd.concat(frames["10"], ignore_index=True),
        "candidate_states_by_rs_bucket_ema20": pd.concat(frames["20"], ignore_index=True),
    }

    small_cells = {}
    total_small = 0
    for key, table in tables.items():
        cnt = int((table["n"] < SMALL_SAMPLE_THRESHOLD).sum())
        small_cells[key] = cnt
        total_small += cnt

    state_population = {}
    for ema in ("10", "20"):
        col = merged[f"candidate_state_ema{ema}"]
        state_population[f"ema{ema}"] = {state: int((col == state).sum()) for state in CANDIDATE_STATES}
        state_population[f"ema{ema}"]["unclassified_missing_current_extension"] = int(col.isna().sum())

    summary = {
        "year": year,
        "n_eligible_total": int(len(merged)),
        "rs_horizon_coverage": rs_coverage,
        "candidate_rule_thresholds": {
            "extended_atr_threshold": EXTENDED_ATR_THRESHOLD,
            "reset_peak_atr_threshold": RESET_PEAK_ATR_THRESHOLD,
            "reset_current_atr_threshold": RESET_CURRENT_ATR_THRESHOLD,
            "reset_ema_distance_threshold": RESET_EMA_DISTANCE_THRESHOLD,
        },
        "state_population": state_population,
        "small_sample_threshold": SMALL_SAMPLE_THRESHOLD,
        "small_sample_cell_count_by_table": small_cells,
        "small_sample_cell_count_total": total_small,
        "no_automatic_best_state_selection": True,
    }
    return tables, summary
