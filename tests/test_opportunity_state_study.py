"""reports/opportunity_state_study.py: purely descriptive RS x EMA-distance
and previous-extension ("repeat offender") cross-tabulations. No Normal/
Extended/Resetting classification, no threshold selection anywhere here --
tests assert the descriptive numbers and structural guardrails only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yolo_calibration.reports.opportunity_state_study import (
    HIGH_RS_THRESHOLD,
    RS_HORIZONS,
    SMALL_SAMPLE_THRESHOLD,
    bucket_ema_distance,
    bucket_previous_extension,
    build_opportunity_state_study,
)


def test_rs_horizons_is_exactly_the_four_horizon_subset():
    assert RS_HORIZONS == ("rs_percentile_1w", "rs_percentile_1m", "rs_percentile_3m", "rs_percentile_6m")
    assert "rs_percentile_1d" not in RS_HORIZONS
    assert "rs_percentile_12m" not in RS_HORIZONS


def test_bucket_ema_distance_boundaries():
    s = pd.Series([-10.0, -5.0, -4.9, -2.0, -1.9, 0.0, 0.1, 2.0, 4.9, 5.0, 9.9, 10.0, 50.0, np.nan])
    out = bucket_ema_distance(s)
    assert out.astype(str).tolist()[:-1] == [
        "<-5%", "-5% to -2%", "-5% to -2%", "-2% to 0%", "-2% to 0%",
        "0% to +2%", "0% to +2%", "+2% to +5%", "+2% to +5%",
        "+5% to +10%", "+5% to +10%", ">=+10%", ">=+10%",
    ]
    assert pd.isna(out.iloc[-1])


def test_bucket_previous_extension_boundaries():
    s = pd.Series([0.0, 3.9, 4.0, 5.9, 6.0, 7.9, 8.0, 9.9, 10.0, 50.0, np.nan])
    out = bucket_previous_extension(s)
    assert out.astype(str).tolist()[:-1] == [
        "<4", "<4", "4-6", "4-6", "6-8", "6-8", "8-10", "8-10", ">=10", ">=10",
    ]
    assert pd.isna(out.iloc[-1])


def _outcome_row(date_, ticker, *, mfe, mae=-3.0, fwd=1.0, race5=1.0, race10=0.0):
    row = {"date": date_, "ticker": ticker}
    for h in (5, 10, 20):
        row[f"mfe_pct_{h}d"] = mfe
        row[f"mae_pct_{h}d"] = mae
        row[f"forward_return_close_{h}d"] = fwd
        row[f"reached_plus_5pct_{h}d"] = 1.0
        row[f"reached_plus_10pct_{h}d"] = 0.0
        row[f"reached_2atr_{h}d"] = 1.0
        row[f"reached_3atr_{h}d"] = 0.0
        row[f"reached_plus_5_before_minus_5_{h}d"] = race5
        row[f"reached_plus_10_before_minus_5_{h}d"] = race10
    return row


def _repeat_offender_fixture():
    """~75 trading days of lookback (2023-10-02 .. 2024-01-12) ending on a
    single 2024 signal day (2024-01-15), for three cohorts sharing the
    SAME current EMA10/EMA20-distance bucket ("-2% to 0%") and the SAME
    high current RS6M (>=90 except the low-RS control group), differing
    ONLY in prior ATR-extension history:

      - control (C1-C3): atr_extension constant 1.0 the whole time ->
        previous-extension bucket "<4" (no meaningful prior extension).
      - repeat_offender (R1-R3): atr_extension spikes to 12.0 for a block
        ~20 trading days before the signal day, else 1.0 -> previous-
        extension bucket ">=10", high current RS6M.
      - low_rs (L1-L2): SAME spike pattern as repeat_offender (bucket
        ">=10") but current RS6M is only 50 (< HIGH_RS_THRESHOLD) -- must
        be excluded from the high-RS repeat-offender tables but still
        present in the RS-agnostic universe-level tables.
    """
    dates = pd.bdate_range("2023-10-02", "2024-01-15")
    signal_date = dates[-1]
    assert signal_date.year == 2024
    spike_start = len(dates) - 1 - 22
    spike_end = spike_start + 5

    def make_ticker(ticker, ext_base, ext_spike, rs6m_signal):
        ext = np.full(len(dates), ext_base)
        if ext_spike is not None:
            ext[spike_start:spike_end] = ext_spike
        d10 = np.zeros(len(dates))
        d10[-1] = -1.0  # signal day only: "-2% to 0%" bucket
        rows = []
        for i, d in enumerate(dates):
            rows.append({
                "date": d, "ticker": ticker, "eligible": True,
                "atr_extension": float(ext[i]),
                "distance_ema10_pct": float(d10[i]), "distance_ema20_pct": float(d10[i]),
                "rs_percentile_1w": 50.0, "rs_percentile_1m": 50.0, "rs_percentile_3m": 50.0,
                "rs_percentile_6m": rs6m_signal if i == len(dates) - 1 else 50.0,
            })
        return rows

    feat_rows = []
    for t in ("C1", "C2", "C3"):
        feat_rows += make_ticker(t, ext_base=1.0, ext_spike=None, rs6m_signal=95.0)
    for t in ("R1", "R2", "R3"):
        feat_rows += make_ticker(t, ext_base=1.0, ext_spike=12.0, rs6m_signal=92.0)
    for t in ("L1", "L2"):
        feat_rows += make_ticker(t, ext_base=1.0, ext_spike=12.0, rs6m_signal=50.0)
    features = pd.DataFrame(feat_rows)

    out_rows = []
    for t in ("C1", "C2", "C3"):
        out_rows.append(_outcome_row(signal_date, t, mfe=4.0, race5=1.0, race10=0.0))
    for t in ("R1", "R2", "R3"):
        out_rows.append(_outcome_row(signal_date, t, mfe=10.0, race5=0.0, race10=0.0))
    for t in ("L1", "L2"):
        out_rows.append(_outcome_row(signal_date, t, mfe=7.0, race5=1.0, race10=1.0))
    outcomes = pd.DataFrame(out_rows)

    return features, outcomes, signal_date


def test_previous_extension_bucket_assignment_matches_atr_history():
    features, outcomes, signal_date = _repeat_offender_fixture()
    tables, summary = build_opportunity_state_study(features, outcomes, 2024)
    ro = tables["repeat_offender_ema10"]
    sub = ro[(ro.rs_horizon == "rs_percentile_6m") & (ro.outcome_horizon == 10)]

    control_cell = sub[(sub.prev_ext_bucket == "<4") & (sub.ema10_bucket == "-2% to 0%")].iloc[0]
    assert control_cell["n"] == 3
    assert control_cell["median_mfe_pct"] == 4.0

    ro_cell = sub[(sub.prev_ext_bucket == ">=10") & (sub.ema10_bucket == "-2% to 0%")].iloc[0]
    assert ro_cell["n"] == 3  # R1-R3 only, NOT L1/L2 (excluded by the RS>=90 cohort filter)
    assert ro_cell["median_mfe_pct"] == 10.0
    assert ro_cell["median_current_rs_in_cell"] == 92.0


def test_baseline_deltas_isolate_the_repeat_offender_effect():
    features, outcomes, signal_date = _repeat_offender_fixture()
    tables, _ = build_opportunity_state_study(features, outcomes, 2024)
    ro = tables["repeat_offender_ema10"]
    sub = ro[(ro.rs_horizon == "rs_percentile_6m") & (ro.outcome_horizon == 10)]
    ro_cell = sub[(sub.prev_ext_bucket == ">=10") & (sub.ema10_bucket == "-2% to 0%")].iloc[0]

    # The crux of the whole study: repeat-offender cell vs the SAME
    # current-EMA-bucket "no prior extension" control, within the same
    # RS>=90 cohort -- must equal exactly 10.0 - 4.0.
    assert ro_cell["delta_vs_no_prior_extension_control_median_mfe_pct"] == pytest.approx(6.0)
    assert ro_cell["baseline_no_prior_extension_control_median_mfe_pct"] == pytest.approx(4.0)

    for m in ("median_mfe_pct", "reached_plus_5_before_minus_5_share"):
        assert ro_cell[f"delta_vs_universe_{m}"] == pytest.approx(ro_cell[m] - ro_cell[f"baseline_universe_{m}"])
        assert ro_cell[f"delta_vs_rs_ge90_cohort_{m}"] == pytest.approx(ro_cell[m] - ro_cell[f"baseline_rs_ge90_cohort_{m}"])
        assert ro_cell[f"delta_vs_no_prior_extension_control_{m}"] == pytest.approx(
            ro_cell[m] - ro_cell[f"baseline_no_prior_extension_control_{m}"]
        )


def test_race_outcomes_correctly_aggregated():
    features, outcomes, signal_date = _repeat_offender_fixture()
    tables, _ = build_opportunity_state_study(features, outcomes, 2024)
    ro = tables["repeat_offender_ema10"]
    sub = ro[(ro.rs_horizon == "rs_percentile_6m") & (ro.outcome_horizon == 20)]

    control_cell = sub[(sub.prev_ext_bucket == "<4") & (sub.ema10_bucket == "-2% to 0%")].iloc[0]
    ro_cell = sub[(sub.prev_ext_bucket == ">=10") & (sub.ema10_bucket == "-2% to 0%")].iloc[0]
    assert control_cell["reached_plus_5_before_minus_5_share"] == 1.0
    assert ro_cell["reached_plus_5_before_minus_5_share"] == 0.0


def test_high_rs_cohort_excludes_low_rs_rows_but_universe_tables_include_them():
    features, outcomes, signal_date = _repeat_offender_fixture()
    tables, summary = build_opportunity_state_study(features, outcomes, 2024)

    ro = tables["repeat_offender_ema10"]
    sub = ro[(ro.rs_horizon == "rs_percentile_6m") & (ro.outcome_horizon == 10)]
    total_n_ro_table = sub["n"].sum()
    assert total_n_ro_table == 6  # C1-3 + R1-3 only; L1/L2 excluded

    universe = tables["prior_extension_x_ema10"]
    usub = universe[universe.outcome_horizon == 10]
    ro_cell_universe = usub[(usub.prev_ext_bucket == ">=10") & (usub.ema10_bucket == "-2% to 0%")].iloc[0]
    assert ro_cell_universe["n"] == 5  # R1-3 AND L1-2 -- this table is RS-agnostic

    assert summary["high_rs_coverage"]["rs_percentile_6m"]["n"] == 6


def test_small_sample_cells_never_merged_or_dropped():
    features, outcomes, signal_date = _repeat_offender_fixture()
    tables, summary = build_opportunity_state_study(features, outcomes, 2024)
    ro = tables["repeat_offender_ema10"]
    sub = ro[(ro.rs_horizon == "rs_percentile_6m") & (ro.outcome_horizon == 10)]
    # 5 prev-ext buckets x 7 ema buckets = 35 rows, even though only 2 of
    # them have any data at all in this fixture.
    assert len(sub) == 35
    assert (sub["n"] == 0).sum() == 33
    assert summary["small_sample_cell_count_total"] > 0


def test_no_automatic_selection_or_state_classification_anywhere():
    features, outcomes, signal_date = _repeat_offender_fixture()
    tables, summary = build_opportunity_state_study(features, outcomes, 2024)

    forbidden = ("best", "optimal", "recommend", "selected", "sweet_spot", "rule",
                 "normal", "extended", "resetting", "opportunity_state", "leader")
    for name, table in tables.items():
        for col in table.columns:
            assert not any(tok in col.lower() for tok in forbidden), f"{name}.{col}"

    allowed_threshold_keys = {"small_sample_threshold", "high_rs_threshold"}
    for key in summary:
        if key in ("no_automatic_selection", *allowed_threshold_keys):
            continue
        assert not any(tok in key.lower() for tok in forbidden), key
    assert summary["no_automatic_selection"] is True


def test_future_year_in_features_raises():
    features, outcomes, signal_date = _repeat_offender_fixture()
    bad = pd.concat([features, features.tail(1).assign(date=pd.Timestamp("2025-01-02"))], ignore_index=True)
    with pytest.raises(ValueError):
        build_opportunity_state_study(bad, outcomes, 2024)


def test_only_target_year_rows_are_analyzed():
    features, outcomes, signal_date = _repeat_offender_fixture()
    # A stray 2023 row in the outcomes table (should never happen from the
    # real pipeline, but must not silently leak into the 2024 analysis
    # even if it did) must not change any cross-tab total.
    stray = _outcome_row(pd.Timestamp("2023-11-01"), "C1", mfe=999.0)
    outcomes_with_stray = pd.concat([outcomes, pd.DataFrame([stray])], ignore_index=True)
    tables_a, _ = build_opportunity_state_study(features, outcomes, 2024)
    tables_b, _ = build_opportunity_state_study(features, outcomes_with_stray, 2024)
    pd.testing.assert_frame_equal(tables_a["repeat_offender_ema10"], tables_b["repeat_offender_ema10"])


def test_missing_required_columns_raises():
    with pytest.raises(ValueError):
        build_opportunity_state_study(pd.DataFrame({"date": [1], "ticker": ["A"]}), pd.DataFrame(), 2024)


def test_cli_requires_lookback_year_before_target_year(tmp_path, monkeypatch):
    import argparse

    import yolo_calibration.cli as cli
    import yolo_calibration.data.storage as storage

    monkeypatch.setattr(storage, "PROCESSED_DIR", tmp_path / "processed")
    args = argparse.Namespace(start=pd.Timestamp("2024-01-01").date(), end=pd.Timestamp("2024-12-31").date())
    assert cli.cmd_build_opportunity_state_study(args) == 1


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
    rc = cli.cmd_build_opportunity_state_study(args)
    assert rc == 0

    out_dir = tmp_path / "reports" / "opportunity_state_2024"
    for name in ("rs_x_ema10", "rs_x_ema20", "prior_extension_x_ema10", "prior_extension_x_ema20",
                 "repeat_offender_ema10", "repeat_offender_ema20"):
        assert (out_dir / f"{name}.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "METHODOLOGY.md").exists()

    assert not (tmp_path / "reports" / "rs_benchmark_2024").exists()
    assert not (tmp_path / "reports" / "rs_atr_extension_2024").exists()
