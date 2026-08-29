"""reports/ema_pullback_range_study.py: calibrates ONLY the EMA10/EMA20
Pullback dashboard labels. No candidate range is ever auto-selected by
this module's code -- tests assert the descriptive percentile/bucket/
comparison numbers and the Resetting-exclusion / Extended-retention
guardrails."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

import numpy as np
import pandas as pd
import pytest

_sys.path.insert(0, str(_Path(__file__).parent))
from test_opportunity_state_study import _outcome_row, _repeat_offender_fixture  # noqa: E402

from yolo_calibration.reports.ema_pullback_range_study import (
    CANDIDATE_GRID_HI,
    CANDIDATE_GRID_LO,
    EMA_FINE_LABELS,
    V1_DASHBOARD_RANGE,
    bucket_ema_distance_fine,
    build_ema_pullback_range_study,
    compare_pullback_range,
)


def test_bucket_ema_distance_fine_boundaries():
    # Left-closed [lo, hi) -- a value exactly on a boundary belongs to the
    # bucket whose name it literally satisfies (-10.0 -> "-10 to -7.5%",
    # not "<-10%"; 10.0 -> ">=+10%").
    s = pd.Series([-20.0, -10.0, -9.9, -7.5, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 50.0, np.nan])
    out = bucket_ema_distance_fine(s)
    assert out.astype(str).tolist()[:-1] == [
        "<-10%", "-10 to -7.5%", "-10 to -7.5%", "-7.5 to -5%", "-5 to -3%", "-3 to -2%", "-2 to -1%", "-1 to 0%",
        "0 to +1%", "+1 to +2%", "+2 to +3%", "+3 to +5%", "+5 to +7.5%", "+7.5 to +10%", ">=+10%", ">=+10%",
    ]
    assert pd.isna(out.iloc[-1])
    assert len(EMA_FINE_LABELS) == 14


def _percentile_fixture():
    """20 lookback + 1 signal day per ticker (atr_extension constant 1.0,
    far below both Extended(>=8) and any plausible Resetting peak) so
    every ticker below is cleanly "Normal", never Resetting -- isolates
    the percentile-distribution math from the exclusion logic (that is
    tested separately in test_resetting_excluded_from_distribution).
    Ten tickers carry the exact distances used in the hand-computed
    pandas .quantile() check in this file's development notes:
    [-6,-4,-3,-2,-1,0,1,2,3,5] -> P10=-4.2, median=-0.5, P90=3.2 (linear
    interpolation, pandas default)."""
    dates = pd.bdate_range("2023-10-02", "2024-01-15")
    signal_date = dates[-1]
    distances = [-6, -4, -3, -2, -1, 0, 1, 2, 3, 5]
    feat_rows, out_rows = [], []
    for i, d10 in enumerate(distances):
        t = f"P{i:02d}"
        for day in dates:
            feat_rows.append({
                "date": day, "ticker": t, "eligible": True,
                "atr_extension": 1.0,
                "distance_ema10_pct": float(d10) if day == signal_date else 0.0,
                "distance_ema20_pct": float(d10) if day == signal_date else 0.0,
                "rs_percentile_1w": 50.0, "rs_percentile_1m": 50.0, "rs_percentile_3m": 50.0,
                "rs_percentile_6m": 95.0 if day == signal_date else 50.0,
            })
        out_rows.append(_outcome_row(signal_date, t, mfe=6.0, race5=1.0, race10=0.0))  # all "successful"
    features = pd.DataFrame(feat_rows)
    outcomes = pd.DataFrame(out_rows)
    return features, outcomes, signal_date


def test_percentile_distribution_matches_hand_computation():
    features, outcomes, signal_date = _percentile_fixture()
    tables, summary = build_ema_pullback_range_study(features, outcomes, 2024)
    dist = tables["ema10_push_distance_distribution"]
    row = dist[(dist.rs_horizon == "rs_percentile_6m") & (dist.rs_population_slice == "RS>=90") &
               (dist.outcome_horizon == 10) & (dist.push_type == "plus5_before_minus5")].iloc[0]

    assert row["n"] == 10
    assert row["p10"] == pytest.approx(-4.2)
    assert row["median"] == pytest.approx(-0.5)
    assert row["p90"] == pytest.approx(3.2)
    assert row["range80_lo"] == pytest.approx(row["p10"])
    assert row["range80_hi"] == pytest.approx(row["p90"])
    assert row["range95_lo"] == pytest.approx(row["p2_5"])
    assert row["range95_hi"] == pytest.approx(row["p97_5"])
    # No forced symmetry: this distribution is not centered on 0.
    assert row["range80_lo"] != -row["range80_hi"]


def test_only_successful_push_rows_enter_the_distribution():
    features, outcomes, signal_date = _percentile_fixture()
    # Flip one ticker's outcome to "not successful" -- it must vanish
    # from the percentile population (n drops), not just get down-weighted.
    outcomes.loc[outcomes["ticker"] == "P00", [c for c in outcomes.columns if c.startswith("reached_plus_5_before_minus_5")]] = 0.0
    tables, _ = build_ema_pullback_range_study(features, outcomes, 2024)
    dist = tables["ema10_push_distance_distribution"]
    row = dist[(dist.rs_horizon == "rs_percentile_6m") & (dist.rs_population_slice == "RS>=90") &
               (dist.outcome_horizon == 10) & (dist.push_type == "plus5_before_minus5")].iloc[0]
    assert row["n"] == 9


def test_resetting_excluded_but_extended_retained():
    features, outcomes, signal_date = _repeat_offender_fixture()
    tables, summary = build_ema_pullback_range_study(features, outcomes, 2024)

    # Fixture recap: R1-3/L1-2 spike to 12.0 ATR ~22 trading days before
    # the signal day -> classified Resetting under both EMA definitions;
    # C1-3 never spike -> Normal. All 5 R/L rows must be excluded from
    # BOTH ema10 and ema20 populations, 3 C rows remain.
    assert summary["resetting_exclusion"]["ema10"]["n_resetting_excluded"] == 5
    assert summary["resetting_exclusion"]["ema10"]["n_remaining"] == 3
    assert summary["resetting_exclusion"]["ema20"]["n_resetting_excluded"] == 5

    # Bucket table must be built only from the 3 remaining (C1-3) rows --
    # total n across all 14 buckets for RS6M/RS>=90/10D must equal 3, not 8.
    buckets = tables["ema10_push_distance_buckets"]
    sub = buckets[(buckets.rs_horizon == "rs_percentile_6m") & (buckets.rs_population_slice == "RS>=90") &
                  (buckets.outcome_horizon == 10)]
    assert sub["n"].sum() == 3


def test_extended_rows_are_never_filtered_out():
    features, outcomes, signal_date = _percentile_fixture()
    # Make one ticker's CURRENT atr_extension Extended (>=8) on the signal
    # day -- it must still appear in the bucket table and count toward
    # extended_among_successful_pushes, never silently dropped.
    features.loc[(features["ticker"] == "P09") & (features["date"] == signal_date), "atr_extension"] = 9.0
    tables, summary = build_ema_pullback_range_study(features, outcomes, 2024)

    buckets = tables["ema10_push_distance_buckets"]
    sub = buckets[(buckets.rs_horizon == "rs_percentile_6m") & (buckets.rs_population_slice == "RS>=90") &
                  (buckets.outcome_horizon == 10)]
    assert sub["n"].sum() == 10  # P09 (Extended) still counted

    ext = summary["extended_among_successful_pushes"]["ema10__rs_percentile_6m"]
    assert ext["n_successful_pushes_10d"] == 10
    assert ext["n_extended_among_them"] == 1
    assert ext["pct_extended_among_them"] == pytest.approx(10.0)


def test_compare_pullback_range_arithmetic():
    features, outcomes, signal_date = _percentile_fixture()
    merged_tables, _ = build_ema_pullback_range_study(features, outcomes, 2024)
    # Rebuild a `work` frame the same way the module does internally, to
    # call compare_pullback_range directly with a known range.
    from yolo_calibration.reports.opportunity_state_study import prepare_eligible_year_rows
    from yolo_calibration.reports.opportunity_state_candidate_rules import classify_candidate_state

    merged = prepare_eligible_year_rows(features, outcomes, 2024)
    merged["candidate_state_ema10"] = classify_candidate_state(merged, "10")
    work = merged.loc[merged["candidate_state_ema10"] != "Resetting"].copy()

    # Range [-3, 3] should catch distances {-3,-2,-1,0,1,2,3} = 7 of 10 rows.
    row = compare_pullback_range(work, "10", "rs_percentile_6m", -3.0, 3.0, "test_range", is_v1=False)
    assert row["n_in_range"] == 7
    assert row["n_successful_plus5_before_minus5_total"] == 10  # all 10 rows are successful pushes
    assert row["share_of_successful_plus5_before_minus5_in_range"] == pytest.approx(70.0)
    assert row["plus5_before_minus5_share_in_range"] == pytest.approx(1.0)  # all rows here ARE the successful ones
    assert row["median_mfe_pct"] == pytest.approx(6.0)


def test_v1_range_is_exactly_minus5_to_plus5():
    assert V1_DASHBOARD_RANGE[1] == -5.0
    assert V1_DASHBOARD_RANGE[2] == 5.0


def test_candidate_comparison_table_includes_v1_and_a_grid_no_winner_marked():
    features, outcomes, signal_date = _percentile_fixture()
    tables, _ = build_ema_pullback_range_study(features, outcomes, 2024)
    cmp = tables["ema_pullback_candidate_comparison"]

    n_expected_ranges = 1 + len(CANDIDATE_GRID_LO) * len(CANDIDATE_GRID_HI)
    assert len(cmp[cmp.ema == "10"]) == n_expected_ranges * 4  # x4 RS horizons
    assert cmp["is_v1_reference"].sum() == 2 * 4  # both EMAs x 4 RS horizons

    forbidden = ("best", "optimal", "recommend", "selected", "winner", "final")
    for col in cmp.columns:
        assert not any(tok in col.lower() for tok in forbidden), col


def test_missing_required_columns_raises():
    with pytest.raises(ValueError):
        build_ema_pullback_range_study(pd.DataFrame({"date": [1], "ticker": ["A"]}), pd.DataFrame(), 2024)


def test_future_year_raises():
    features, outcomes, signal_date = _percentile_fixture()
    bad = pd.concat([features, features.tail(1).assign(date=pd.Timestamp("2025-01-02"))], ignore_index=True)
    with pytest.raises(ValueError):
        build_ema_pullback_range_study(bad, outcomes, 2024)


def test_small_sample_cells_never_merged_or_dropped():
    features, outcomes, signal_date = _repeat_offender_fixture()
    tables, summary = build_ema_pullback_range_study(features, outcomes, 2024)
    buckets = tables["ema10_push_distance_buckets"]
    sub = buckets[(buckets.rs_horizon == "rs_percentile_6m") & (buckets.rs_population_slice == "RS>=90") &
                  (buckets.outcome_horizon == 10)]
    assert len(sub) == 14  # all 14 fixed buckets present even though only 1-2 are populated
    assert summary["small_sample_cell_count_total"] > 0


def test_cli_requires_lookback_year_before_target_year(tmp_path, monkeypatch):
    import argparse

    import yolo_calibration.cli as cli
    import yolo_calibration.data.storage as storage

    monkeypatch.setattr(storage, "PROCESSED_DIR", tmp_path / "processed")
    args = argparse.Namespace(start=pd.Timestamp("2024-01-01").date(), end=pd.Timestamp("2024-12-31").date())
    assert cli.cmd_build_ema_pullback_range_study(args) == 1


def test_cli_writes_outputs_under_a_new_report_dir(tmp_path, monkeypatch):
    import argparse

    import yolo_calibration.cli as cli
    import yolo_calibration.data.storage as storage

    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(storage, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

    features, outcomes, signal_date = _repeat_offender_fixture()
    storage.write_processed_by_year("stock_features_daily", features)
    storage.write_processed_by_year("stock_outcomes_daily", outcomes)

    args = argparse.Namespace(start=pd.Timestamp("2023-10-02").date(), end=pd.Timestamp("2024-01-15").date())
    rc = cli.cmd_build_ema_pullback_range_study(args)
    assert rc == 0

    out_dir = tmp_path / "reports" / "ema_pullback_range_2024"
    for name in ("ema10_push_distance_distribution", "ema20_push_distance_distribution",
                 "ema10_push_distance_buckets", "ema20_push_distance_buckets",
                 "ema_pullback_candidate_comparison"):
        assert (out_dir / f"{name}.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "METHODOLOGY.md").exists()

    assert not (tmp_path / "reports" / "opportunity_state_2024").exists()
    assert not (tmp_path / "reports" / "opportunity_state_candidate_rules_2024").exists()
