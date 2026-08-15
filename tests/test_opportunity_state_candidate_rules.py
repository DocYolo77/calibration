"""reports/opportunity_state_candidate_rules.py: the FROZEN Candidate
Rules v1 classification (Normal/Extended/Resetting). Unlike the other
Phase-2 modules, this one legitimately classifies rows -- tests instead
guard that the thresholds are truly frozen (no override path), that
EMA10/EMA20 stay two separate classifications, and that no "best state"
language or logic slips in."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

import numpy as np
import pandas as pd
import pytest

_sys.path.insert(0, str(_Path(__file__).parent))
from test_opportunity_state_study import _repeat_offender_fixture  # noqa: E402

from yolo_calibration.reports.opportunity_state_candidate_rules import (
    CANDIDATE_STATES,
    EXTENDED_ATR_THRESHOLD,
    RESET_CURRENT_ATR_THRESHOLD,
    RESET_EMA_DISTANCE_THRESHOLD,
    RESET_PEAK_ATR_THRESHOLD,
    build_candidate_rules_report,
    classify_candidate_state,
)


def _row(atr_extension, atr_extension_peak, distance_ema10_pct, distance_ema20_pct=None):
    return {
        "atr_extension": atr_extension,
        "atr_extension_peak": atr_extension_peak,
        "distance_ema10_pct": distance_ema10_pct,
        "distance_ema20_pct": distance_ema20_pct if distance_ema20_pct is not None else distance_ema10_pct,
    }


def test_extended_boundary():
    df = pd.DataFrame([
        _row(atr_extension=7.999, atr_extension_peak=np.nan, distance_ema10_pct=5.0),
        _row(atr_extension=8.0, atr_extension_peak=np.nan, distance_ema10_pct=5.0),
        _row(atr_extension=50.0, atr_extension_peak=np.nan, distance_ema10_pct=5.0),
    ])
    out = classify_candidate_state(df, "10")
    assert out.tolist() == ["Normal", "Extended", "Extended"]


def test_resetting_requires_all_three_conditions():
    df = pd.DataFrame([
        # Peak high enough, current unwound, at/below EMA10 -> Resetting.
        _row(atr_extension=3.9, atr_extension_peak=8.0, distance_ema10_pct=0.0),
        # Peak just short of threshold -> not Resetting (falls to Normal).
        _row(atr_extension=3.9, atr_extension_peak=7.999, distance_ema10_pct=0.0),
        # Current extension not yet unwound (>= 4) -> not Resetting.
        _row(atr_extension=4.0, atr_extension_peak=8.0, distance_ema10_pct=0.0),
        # Still above the EMA (> 0) -> not Resetting.
        _row(atr_extension=3.9, atr_extension_peak=8.0, distance_ema10_pct=0.1),
    ])
    out = classify_candidate_state(df, "10")
    assert out.tolist() == ["Resetting", "Normal", "Normal", "Normal"]


def test_extended_and_resetting_are_mutually_exclusive_by_construction():
    # RESET_CURRENT_ATR_THRESHOLD (4.0) < EXTENDED_ATR_THRESHOLD (8.0) makes
    # this a structural guarantee, not a runtime special case -- assert the
    # invariant the module docstring claims still holds for these constants.
    assert RESET_CURRENT_ATR_THRESHOLD < EXTENDED_ATR_THRESHOLD
    assert RESET_PEAK_ATR_THRESHOLD == EXTENDED_ATR_THRESHOLD  # "was Extended, is not anymore"


def test_no_prior_history_can_never_be_resetting():
    df = pd.DataFrame([
        _row(atr_extension=1.0, atr_extension_peak=np.nan, distance_ema10_pct=-3.0),
        _row(atr_extension=9.0, atr_extension_peak=np.nan, distance_ema10_pct=-3.0),
    ])
    out = classify_candidate_state(df, "10")
    assert out.tolist() == ["Normal", "Extended"]  # never "Resetting" without a real peak


def test_missing_current_extension_is_unclassified_not_normal():
    df = pd.DataFrame([_row(atr_extension=np.nan, atr_extension_peak=8.0, distance_ema10_pct=-1.0)])
    out = classify_candidate_state(df, "10")
    assert pd.isna(out.iloc[0])


def test_ema10_and_ema20_are_independent_classifications():
    # Same row: below EMA10 but still above EMA20 -> Resetting under EMA10
    # only, Normal under EMA20 -- the two must never be merged/averaged.
    df = pd.DataFrame([_row(atr_extension=2.0, atr_extension_peak=9.0, distance_ema10_pct=-0.5, distance_ema20_pct=3.0)])
    out10 = classify_candidate_state(df, "10")
    out20 = classify_candidate_state(df, "20")
    assert out10.tolist() == ["Resetting"]
    assert out20.tolist() == ["Normal"]


def test_invalid_ema_argument_raises():
    df = pd.DataFrame([_row(atr_extension=1.0, atr_extension_peak=1.0, distance_ema10_pct=0.0)])
    with pytest.raises(ValueError):
        classify_candidate_state(df, "15")


def test_candidate_states_constant_matches_classification_output():
    df = pd.DataFrame([
        _row(atr_extension=1.0, atr_extension_peak=1.0, distance_ema10_pct=5.0),   # Normal
        _row(atr_extension=9.0, atr_extension_peak=1.0, distance_ema10_pct=5.0),   # Extended
        _row(atr_extension=1.0, atr_extension_peak=9.0, distance_ema10_pct=-1.0),  # Resetting
    ])
    out = classify_candidate_state(df, "10")
    assert set(out.dropna().unique().tolist()) <= set(CANDIDATE_STATES)
    assert set(out.tolist()) == {"Normal", "Extended", "Resetting"}


def test_state_populations_and_cross_tab_values():
    features, outcomes, signal_date = _repeat_offender_fixture()
    tables, summary = build_candidate_rules_report(features, outcomes, 2024)

    # Fixture recap: C1-3 never spike (always Normal); R1-3 and L1-2 spike
    # to 12.0 ATR ~22 trading days before the signal day then sit at 1.0
    # with distance_ema10/20_pct == -1.0 on the signal day -> Resetting
    # under BOTH EMA10 and EMA20 (peak=12>=8, current=1<4, dist=-1<=0).
    assert summary["state_population"]["ema10"] == {
        "Normal": 3, "Extended": 0, "Resetting": 5, "unclassified_missing_current_extension": 0,
    }
    assert summary["state_population"]["ema20"] == summary["state_population"]["ema10"]

    cross = tables["candidate_states_by_rs_bucket_ema10"]
    sub = cross[(cross.rs_horizon == "rs_percentile_6m") & (cross.outcome_horizon == 10)]

    control = sub[(sub.rs_bucket == "95-97.5") & (sub.candidate_state == "Normal")].iloc[0]
    assert control["n"] == 3
    assert control["median_mfe_pct"] == 4.0

    reset_high_rs = sub[(sub.rs_bucket == "90-95") & (sub.candidate_state == "Resetting")].iloc[0]
    assert reset_high_rs["n"] == 3
    assert reset_high_rs["median_mfe_pct"] == 10.0

    reset_low_rs = sub[(sub.rs_bucket == "<80") & (sub.candidate_state == "Resetting")].iloc[0]
    assert reset_low_rs["n"] == 2
    assert reset_low_rs["median_mfe_pct"] == 7.0


def test_baseline_vs_same_rs_bucket_no_state_filter():
    features, outcomes, signal_date = _repeat_offender_fixture()
    tables, _ = build_candidate_rules_report(features, outcomes, 2024)
    cross = tables["candidate_states_by_rs_bucket_ema10"]
    sub = cross[(cross.rs_horizon == "rs_percentile_6m") & (cross.outcome_horizon == 10)]

    # 90-95 bucket contains ONLY the Resetting rows (R1-3) in this fixture
    # -> "same RS bucket without state filter" baseline must equal the
    # Resetting cell itself here (delta == 0), a useful self-check.
    reset_cell = sub[(sub.rs_bucket == "90-95") & (sub.candidate_state == "Resetting")].iloc[0]
    assert reset_cell["baseline_rs_bucket_median_mfe_pct"] == pytest.approx(10.0)
    assert reset_cell["delta_vs_rs_bucket_median_mfe_pct"] == pytest.approx(0.0)

    for m in ("median_mfe_pct", "reached_plus_5_before_minus_5_share"):
        assert reset_cell[f"delta_vs_universe_{m}"] == pytest.approx(reset_cell[m] - reset_cell[f"baseline_universe_{m}"])


def test_race_outcomes_correctly_aggregated():
    features, outcomes, signal_date = _repeat_offender_fixture()
    tables, _ = build_candidate_rules_report(features, outcomes, 2024)
    cross = tables["candidate_states_by_rs_bucket_ema10"]
    sub = cross[(cross.rs_horizon == "rs_percentile_6m") & (cross.outcome_horizon == 20)]

    control = sub[(sub.rs_bucket == "95-97.5") & (sub.candidate_state == "Normal")].iloc[0]
    reset_high_rs = sub[(sub.rs_bucket == "90-95") & (sub.candidate_state == "Resetting")].iloc[0]
    assert control["reached_plus_5_before_minus_5_share"] == 1.0
    assert reset_high_rs["reached_plus_5_before_minus_5_share"] == 0.0


def test_small_and_empty_cells_never_merged_or_dropped():
    features, outcomes, signal_date = _repeat_offender_fixture()
    tables, summary = build_candidate_rules_report(features, outcomes, 2024)
    cross = tables["candidate_states_by_rs_bucket_ema10"]
    sub = cross[(cross.rs_horizon == "rs_percentile_6m") & (cross.outcome_horizon == 10)]
    # 5 rs_buckets x 3 states = 15 rows, even though most are empty here.
    assert len(sub) == 15
    assert (sub["n"] == 0).sum() == 12  # only 3 of the 15 cells are populated in this fixture
    assert summary["small_sample_cell_count_total"] > 0


def test_no_best_state_selection_or_forbidden_language():
    features, outcomes, signal_date = _repeat_offender_fixture()
    tables, summary = build_candidate_rules_report(features, outcomes, 2024)

    forbidden = ("best", "optimal", "recommend", "selected", "sweet_spot", "final", "production", "score")
    for name, table in tables.items():
        for col in table.columns:
            assert not any(tok in col.lower() for tok in forbidden), f"{name}.{col}"
        # candidate_state VALUES are exactly the frozen three -- no extra
        # label (e.g. a silently-added "Leader") ever leaks in.
        assert set(table["candidate_state"].astype(str).unique()) <= set(CANDIDATE_STATES)

    for key in summary:
        if key in ("no_automatic_best_state_selection", "small_sample_threshold", "candidate_rule_thresholds"):
            continue
        assert not any(tok in key.lower() for tok in forbidden), key
    assert summary["no_automatic_best_state_selection"] is True


def test_thresholds_are_frozen_module_constants_not_call_parameters():
    import inspect

    sig = inspect.signature(classify_candidate_state)
    assert list(sig.parameters) == ["df", "ema"]  # no threshold override parameter exists at all

    sig2 = inspect.signature(build_candidate_rules_report)
    assert list(sig2.parameters) == ["stock_features_daily", "stock_outcomes_daily", "year"]

    # The exact frozen values this docstring/report is built around --
    # a change here is a deliberate recalibration, never an accidental drift.
    assert EXTENDED_ATR_THRESHOLD == 8.0
    assert RESET_PEAK_ATR_THRESHOLD == 8.0
    assert RESET_CURRENT_ATR_THRESHOLD == 4.0
    assert RESET_EMA_DISTANCE_THRESHOLD == 0.0


def test_only_target_year_rows_analyzed_and_no_future_leakage():
    features, outcomes, signal_date = _repeat_offender_fixture()
    bad = pd.concat([features, features.tail(1).assign(date=pd.Timestamp("2025-01-02"))], ignore_index=True)
    with pytest.raises(ValueError):
        build_candidate_rules_report(bad, outcomes, 2024)


def test_missing_required_columns_raises():
    with pytest.raises(ValueError):
        build_candidate_rules_report(pd.DataFrame({"date": [1], "ticker": ["A"]}), pd.DataFrame(), 2024)


def test_cli_requires_lookback_year_before_target_year(tmp_path, monkeypatch):
    import argparse

    import yolo_calibration.cli as cli
    import yolo_calibration.data.storage as storage

    monkeypatch.setattr(storage, "PROCESSED_DIR", tmp_path / "processed")
    args = argparse.Namespace(start=pd.Timestamp("2024-01-01").date(), end=pd.Timestamp("2024-12-31").date())
    assert cli.cmd_build_opportunity_state_candidate_rules_report(args) == 1


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
    rc = cli.cmd_build_opportunity_state_candidate_rules_report(args)
    assert rc == 0

    out_dir = tmp_path / "reports" / "opportunity_state_candidate_rules_2024"
    for name in ("candidate_states_by_rs_bucket_ema10", "candidate_states_by_rs_bucket_ema20"):
        assert (out_dir / f"{name}.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "METHODOLOGY.md").exists()

    assert not (tmp_path / "reports" / "opportunity_state_2024").exists()
    assert not (tmp_path / "reports" / "rs_atr_extension_2024").exists()
