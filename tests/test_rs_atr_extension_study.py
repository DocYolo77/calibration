"""reports/rs_atr_extension_study.py: purely descriptive RS x ATR-Extension
cross-tabulation. Same scope discipline as test_rs_benchmark.py -- no
threshold selection, no "best" cell/RS-horizon logic anywhere here."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yolo_calibration.reports.rs_atr_extension_study import (
    ATR_EXT_LABELS,
    METRIC_COLS,
    RS_CROSS_LABELS,
    SMALL_SAMPLE_THRESHOLD,
    bucket_atr_extension,
    bucket_rs_cross_matrix,
    build_rs_atr_extension_study,
)


def test_bucket_rs_cross_matrix_boundaries():
    # Left-closed [lo, hi) -- a value exactly on a boundary belongs to the
    # bucket whose name it literally satisfies (80.0 -> "80-90", not "<80";
    # 100.0, the maximum possible percentile rank, -> "97.5-100").
    rs = pd.Series([0.0, 79.999, 80.0, 89.999, 90.0, 95.0, 97.5, 99.999, 100.0, np.nan])
    out = bucket_rs_cross_matrix(rs)
    assert out.astype(str).tolist()[:-1] == [
        "<80", "<80", "80-90", "80-90", "90-95", "95-97.5", "97.5-100", "97.5-100", "97.5-100",
    ]
    assert pd.isna(out.iloc[-1])


def test_bucket_atr_extension_boundaries():
    # Left-closed [lo, hi) -- "<0" excludes 0.0, ">=10" includes 10.0 and
    # everything above (unbounded on both ends via +/-inf bin edges).
    atr = pd.Series([-100.0, -0.001, 0.0, 1.999, 2.0, 9.999, 10.0, 500.0, np.nan])
    out = bucket_atr_extension(atr)
    assert out.astype(str).tolist()[:-1] == ["<0", "<0", "0-2", "0-2", "2-4", "8-10", ">=10", ">=10"]
    assert pd.isna(out.iloc[-1])


def _outcome_row_stub(date_, ticker, *, mfe=5.0, mae=-3.0, fwd=1.0, race5=1.0, race10=0.0):
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


def _controlled_fixture():
    """One date, one RS horizon of interest (rs_percentile_1d), all six
    tickers share the SAME RS bucket ("90-95") but split across TWO ATR
    Extension buckets ("0-2" for T1-T3, "6-8" for T4-T6) with deliberately
    different mfe/race outcomes, so the RS-bucket baseline (ATR-agnostic)
    and universe baseline can both be checked against a hand-computed
    value. Other RS horizons are left fully NaN (not this fixture's
    concern) so only rs_percentile_1d's cross tables are exercised below."""
    d = pd.Timestamp("2024-03-01")
    tickers_low_ext = ["T1", "T2", "T3"]
    tickers_high_ext = ["T4", "T5", "T6"]
    feat_rows, out_rows = [], []
    for t in tickers_low_ext:
        feat_rows.append({
            "date": d, "ticker": t, "eligible": True, "atr_extension": 1.0,
            "rs_percentile_1d": 92.0, "rs_percentile_1w": np.nan, "rs_percentile_1m": np.nan,
            "rs_percentile_3m": np.nan, "rs_percentile_6m": np.nan, "rs_percentile_12m": np.nan,
        })
        out_rows.append(_outcome_row_stub(d, t, mfe=4.0, race5=1.0, race10=0.0))
    for t in tickers_high_ext:
        feat_rows.append({
            "date": d, "ticker": t, "eligible": True, "atr_extension": 7.0,
            "rs_percentile_1d": 93.0, "rs_percentile_1w": np.nan, "rs_percentile_1m": np.nan,
            "rs_percentile_3m": np.nan, "rs_percentile_6m": np.nan, "rs_percentile_12m": np.nan,
        })
        out_rows.append(_outcome_row_stub(d, t, mfe=20.0, race5=0.0, race10=0.0))
    return pd.DataFrame(feat_rows), pd.DataFrame(out_rows)


def test_cross_matrix_cell_values_match_hand_computation():
    features, outcomes = _controlled_fixture()
    tables, summary = build_rs_atr_extension_study(features, outcomes)
    cm = tables["cross_matrix"]

    row = cm[
        (cm["rs_horizon"] == "rs_percentile_1d") & (cm["outcome_horizon"] == 5)
        & (cm["rs_bucket"] == "90-95") & (cm["atr_bucket"] == "0-2")
    ].iloc[0]
    assert row["n"] == 3
    assert row["median_mfe_pct"] == 4.0
    assert row["reached_plus_5_before_minus_5_share"] == 1.0

    row_hi = cm[
        (cm["rs_horizon"] == "rs_percentile_1d") & (cm["outcome_horizon"] == 5)
        & (cm["rs_bucket"] == "90-95") & (cm["atr_bucket"] == "6-8")
    ].iloc[0]
    assert row_hi["n"] == 3
    assert row_hi["median_mfe_pct"] == 20.0
    assert row_hi["reached_plus_5_before_minus_5_share"] == 0.0

    # Every other RS x ATR cell for this RS horizon is empty (n=0) but
    # still present -- transparency guardrail, see
    # test_small_and_empty_cells_are_never_merged_or_dropped below.
    assert set(RS_CROSS_LABELS) == set(cm.loc[cm["rs_horizon"] == "rs_percentile_1d", "rs_bucket"].astype(str).unique())
    assert set(ATR_EXT_LABELS) == set(cm.loc[cm["rs_horizon"] == "rs_percentile_1d", "atr_bucket"].astype(str).unique())


def test_baseline_deltas_are_correct():
    features, outcomes = _controlled_fixture()
    tables, summary = build_rs_atr_extension_study(features, outcomes)
    cm = tables["cross_matrix"]

    cell = cm[
        (cm["rs_horizon"] == "rs_percentile_1d") & (cm["outcome_horizon"] == 5)
        & (cm["rs_bucket"] == "90-95") & (cm["atr_bucket"] == "0-2")
    ].iloc[0]

    # RS-bucket baseline: ALL 6 rows share rs_bucket "90-95" (92.0 and 93.0
    # both fall in [90,95)), independent of atr_bucket -> median of
    # [4,4,4,20,20,20] = 12.0.
    assert cell["baseline_rs_bucket_median_mfe_pct"] == 12.0
    assert cell["delta_vs_rs_bucket_median_mfe_pct"] == pytest.approx(4.0 - 12.0)

    # Universe baseline: identical population here (only these 6 rows are
    # eligible in 2024), so it must match the RS-bucket baseline exactly
    # for this fixture.
    assert cell["baseline_universe_median_mfe_pct"] == 12.0
    assert cell["delta_vs_universe_median_mfe_pct"] == pytest.approx(4.0 - 12.0)

    # Arithmetic identity holds for every metric column, not just MFE.
    for m in METRIC_COLS:
        assert cell[f"delta_vs_rs_bucket_{m}"] == pytest.approx(cell[m] - cell[f"baseline_rs_bucket_{m}"], nan_ok=True)
        assert cell[f"delta_vs_universe_{m}"] == pytest.approx(cell[m] - cell[f"baseline_universe_{m}"], nan_ok=True)


def test_race_outcomes_correctly_aggregated_per_cell():
    features, outcomes = _controlled_fixture()
    tables, _ = build_rs_atr_extension_study(features, outcomes)
    cm = tables["cross_matrix"]

    low = cm[(cm["rs_horizon"] == "rs_percentile_1d") & (cm["outcome_horizon"] == 10)
             & (cm["rs_bucket"] == "90-95") & (cm["atr_bucket"] == "0-2")].iloc[0]
    high = cm[(cm["rs_horizon"] == "rs_percentile_1d") & (cm["outcome_horizon"] == 10)
              & (cm["rs_bucket"] == "90-95") & (cm["atr_bucket"] == "6-8")].iloc[0]
    assert low["reached_plus_5_before_minus_5_share"] == 1.0
    assert high["reached_plus_5_before_minus_5_share"] == 0.0
    assert low["reached_plus_10_before_minus_5_share"] == 0.0
    assert high["reached_plus_10_before_minus_5_share"] == 0.0


def test_small_and_empty_cells_are_never_merged_or_dropped():
    features, outcomes = _controlled_fixture()
    tables, summary = build_rs_atr_extension_study(features, outcomes)
    cm = tables["cross_matrix"]

    # All 5 RS buckets x 7 ATR buckets must appear for every (RS horizon,
    # outcome horizon) pair -- 35 rows per combination, none silently
    # merged away because they happened to be empty or tiny.
    one_combo = cm[(cm["rs_horizon"] == "rs_percentile_1d") & (cm["outcome_horizon"] == 5)]
    assert len(one_combo) == len(RS_CROSS_LABELS) * len(ATR_EXT_LABELS)

    empty_cells = one_combo[one_combo["n"] == 0]
    assert len(empty_cells) == len(RS_CROSS_LABELS) * len(ATR_EXT_LABELS) - 2  # only the two populated cells above
    assert empty_cells["median_mfe_pct"].isna().all()

    # All 6 rows in this fixture land in cells with n=3 < SMALL_SAMPLE_THRESHOLD
    # (30) -- summary must surface them transparently, not merge/hide them.
    assert summary["small_sample_cell_count"] >= 2
    small_n_values = {c["n"] for c in summary["small_sample_cells"]}
    assert small_n_values <= {0, 3}


def test_no_threshold_or_best_cell_selection_anywhere():
    """Guardrail: no column or summary key may look like an automatic
    threshold/best-cell/best-RS-horizon decision."""
    features, outcomes = _controlled_fixture()
    tables, summary = build_rs_atr_extension_study(features, outcomes)

    forbidden_tokens = ("best", "optimal", "recommend", "selected", "sweet_spot", "rule")
    for table in tables.values():
        for col in table.columns:
            assert not any(tok in col.lower() for tok in forbidden_tokens), col

    for key in summary:
        if key == "no_automatic_selection":
            continue  # documents the guardrail itself, not a violation of it
        assert not any(tok in key.lower() for tok in forbidden_tokens), key
    assert summary["no_automatic_selection"] is True

    # "threshold" is expected to appear exactly once, in the transparency
    # (small-sample) metadata key -- never attached to a chosen value.
    threshold_keys = [k for k in summary if "threshold" in k.lower()]
    assert threshold_keys == ["small_sample_threshold"]
    assert summary["small_sample_threshold"] == SMALL_SAMPLE_THRESHOLD


def test_missing_atr_extension_column_raises():
    features, outcomes = _controlled_fixture()
    features = features.drop(columns=["atr_extension"])
    with pytest.raises(ValueError):
        build_rs_atr_extension_study(features, outcomes)


def test_uses_existing_atr_extension_column_without_recomputing_it():
    """The module must read the already-computed `atr_extension` column
    verbatim (features/technical.py::add_atr_extension) and never derive
    its own extension formula from close/sma50/atr_pct -- proven here by
    the fact that the study succeeds even when none of those raw columns
    are present at all."""
    features, outcomes = _controlled_fixture()
    assert not {"close", "sma50", "atr_pct"} & set(features.columns)
    tables, _ = build_rs_atr_extension_study(features, outcomes)
    assert tables["cross_matrix"]["n"].sum() > 0


def _large_synthetic_fixture():
    """Larger synthetic fixture (mirrors test_rs_benchmark.py's
    _synthetic_features_outcomes shape) exercising all 6 RS horizons'
    coverage bookkeeping and a non-trivial atr_extension spread."""
    dates = pd.bdate_range("2024-01-02", periods=15)
    tickers = [f"T{i:02d}" for i in range(10)]
    rng = np.random.default_rng(11)
    feat_rows, out_rows = [], []
    for d in dates:
        for t in tickers:
            rs = rng.uniform(0, 100)
            feat_rows.append({
                "date": d, "ticker": t, "eligible": True,
                "atr_extension": rng.normal(3, 4),
                "rs_percentile_1d": rs, "rs_percentile_1w": rs, "rs_percentile_1m": rs,
                "rs_percentile_3m": rs if d >= dates[5] else np.nan,
                "rs_percentile_6m": np.nan, "rs_percentile_12m": np.nan,
            })
            out_rows.append(_outcome_row_stub(
                d, t, mfe=rng.normal(5, 2), mae=-rng.normal(3, 1), fwd=rng.normal(1, 3),
                race5=float(rng.random() < 0.4), race10=float(rng.random() < 0.15),
            ))
    return pd.DataFrame(feat_rows), pd.DataFrame(out_rows)


def test_rs_horizon_coverage_reflects_populated_rows_only():
    features, outcomes = _large_synthetic_fixture()
    _, summary = build_rs_atr_extension_study(features, outcomes)

    n_total = summary["n_eligible_2024_total"]
    assert summary["rs_horizon_coverage"]["rs_percentile_1d"]["n"] == n_total
    assert summary["rs_horizon_coverage"]["rs_percentile_1d"]["pct"] == 100.0
    assert summary["rs_horizon_coverage"]["rs_percentile_6m"]["n"] == 0
    assert summary["rs_horizon_coverage"]["rs_percentile_6m"]["pct"] == 0.0
    # rs_percentile_3m populated only from dates[5:] onward (10 of 15 dates).
    assert summary["rs_horizon_coverage"]["rs_percentile_3m"]["n"] == 10 * 10


def test_cli_rejects_cross_year_range(tmp_path, monkeypatch):
    import argparse

    import yolo_calibration.cli as cli
    import yolo_calibration.data.storage as storage

    monkeypatch.setattr(storage, "PROCESSED_DIR", tmp_path / "processed")
    args = argparse.Namespace(start=pd.Timestamp("2024-01-01").date(), end=pd.Timestamp("2025-01-05").date())
    assert cli.cmd_build_rs_atr_extension_study(args) == 1


def test_cli_writes_outputs_for_a_single_year_under_a_new_report_dir(tmp_path, monkeypatch):
    import argparse

    import yolo_calibration.cli as cli
    import yolo_calibration.data.storage as storage

    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(storage, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

    features, outcomes = _large_synthetic_fixture()
    storage.write_processed_by_year("stock_features_daily", features)
    storage.write_processed_by_year("stock_outcomes_daily", outcomes)

    args = argparse.Namespace(start=pd.Timestamp("2024-01-02").date(), end=pd.Timestamp("2024-12-31").date())
    rc = cli.cmd_build_rs_atr_extension_study(args)
    assert rc == 0

    out_dir = tmp_path / "reports" / "rs_atr_extension_2024"
    assert (out_dir / "cross_matrix.csv").exists()
    assert (out_dir / "rs_bucket_baseline.csv").exists()
    assert (out_dir / "universe_baseline.csv").exists()
    assert (out_dir / "summary.json").exists()

    # Must never touch/overwrite the existing (different-purpose) RS
    # benchmark report directories.
    assert not (tmp_path / "reports" / "rs_benchmark_2024").exists()
    assert not (tmp_path / "reports" / "rs_benchmark_2024_common_sample").exists()
