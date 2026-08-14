"""reports/rs_benchmark.py: purely descriptive RS-horizon-vs-outcome bucket
stats. No threshold selection, no "best" RS horizon logic anywhere here —
tests assert the descriptive numbers only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yolo_calibration.reports.rs_benchmark import (
    bucket_coarse_deciles,
    bucket_fine_above_80,
    build_rs_benchmark_report,
    compute_bucket_stats,
)


def test_bucket_coarse_deciles_boundaries():
    # right-closed bins ((lo, hi], first bin additionally includes the
    # exact lower edge 0 via include_lowest) -- a value exactly on a
    # decade boundary (10.0, 90.0) belongs to the LOWER bucket.
    rs = pd.Series([0.0, 5.0, 9.9, 10.0, 45.0, 89.9, 90.0, 100.0])
    out = bucket_coarse_deciles(rs)
    assert list(out.astype(str)) == ["0-10", "0-10", "0-10", "0-10", "40-50", "80-90", "80-90", "90-100"]


def test_bucket_fine_above_80_excludes_below_80():
    rs = pd.Series([79.9, 80.0, 81.0, 82.5, 97.5, 100.0, np.nan])
    out = bucket_fine_above_80(rs)
    assert pd.isna(out.iloc[0])  # 79.9 -> below the fine range entirely
    assert str(out.iloc[1]) == "80-82.5"
    assert str(out.iloc[2]) == "80-82.5"
    assert str(out.iloc[3]) == "80-82.5"  # 82.5 lands in the lower bucket (right-closed boundary)
    assert str(out.iloc[4]) == "95-97.5"
    assert str(out.iloc[5]) == "97.5-100"
    assert pd.isna(out.iloc[6])


def test_compute_bucket_stats_values_and_probabilities():
    df = pd.DataFrame({
        "bucket": pd.Categorical(["A", "A", "A", "B"], categories=["A", "B", "C"]),
        "mfe_pct_10d": [10.0, 20.0, 30.0, 5.0],
        "mae_pct_10d": [-1.0, -2.0, -3.0, -0.5],
        "forward_return_close_10d": [1.0, 2.0, 3.0, 0.5],
        "reached_plus_5pct_10d": [1.0, 1.0, 0.0, 0.0],
        "reached_plus_10pct_10d": [1.0, 0.0, 0.0, 0.0],
        "reached_2atr_10d": [1.0, 1.0, 1.0, 0.0],
        "reached_3atr_10d": [0.0, 0.0, 0.0, 0.0],
        "reached_plus_5_before_minus_5_10d": [1.0, 1.0, 0.0, 0.0],
        "reached_plus_10_before_minus_5_10d": [1.0, 0.0, 0.0, 0.0],
    })
    out = compute_bucket_stats(df, "bucket", 10)

    row_a = out[out["bucket"] == "A"].iloc[0]
    assert row_a["n"] == 3
    assert row_a["median_mfe_pct"] == 20.0
    assert abs(row_a["mean_mfe_pct"] - 20.0) < 1e-9
    assert row_a["p75_mfe_pct"] == pd.Series([10.0, 20.0, 30.0]).quantile(0.75)
    assert row_a["p90_mfe_pct"] == pd.Series([10.0, 20.0, 30.0]).quantile(0.90)
    assert abs(row_a["reached_plus_5pct_share"] - (2 / 3)) < 1e-9
    assert abs(row_a["reached_plus_10pct_share"] - (1 / 3)) < 1e-9
    assert row_a["reached_2atr_share"] == 1.0
    assert row_a["reached_3atr_share"] == 0.0
    assert abs(row_a["reached_plus_5_before_minus_5_share"] - (2 / 3)) < 1e-9
    assert abs(row_a["reached_plus_10_before_minus_5_share"] - (1 / 3)) < 1e-9

    row_b = out[out["bucket"] == "B"].iloc[0]
    assert row_b["n"] == 1

    # Category "C" has zero rows -- must still appear (visible gap), not be dropped.
    row_c = out[out["bucket"] == "C"].iloc[0]
    assert row_c["n"] == 0
    assert pd.isna(row_c["median_mfe_pct"])


def test_compute_bucket_stats_raises_when_race_outcome_column_missing():
    df = pd.DataFrame({
        "bucket": pd.Categorical(["A"], categories=["A"]),
        "mfe_pct_5d": [1.0], "mae_pct_5d": [-1.0], "forward_return_close_5d": [1.0],
        "reached_plus_5pct_5d": [1.0], "reached_plus_10pct_5d": [1.0],
        "reached_2atr_5d": [1.0], "reached_3atr_5d": [1.0],
        # race-outcome columns intentionally omitted
    })
    with pytest.raises(ValueError):
        compute_bucket_stats(df, "bucket", 5)


def test_no_threshold_selection_or_best_horizon_columns():
    """Guardrail: the report module must never emit a column that looks
    like an automatic threshold/best-horizon decision."""
    df = pd.DataFrame({
        "bucket": pd.Categorical(["A"], categories=["A"]),
        "mfe_pct_5d": [1.0], "mae_pct_5d": [-1.0], "forward_return_close_5d": [1.0],
        "reached_plus_5pct_5d": [1.0], "reached_plus_10pct_5d": [1.0],
        "reached_2atr_5d": [1.0], "reached_3atr_5d": [1.0],
        "reached_plus_5_before_minus_5_5d": [1.0], "reached_plus_10_before_minus_5_5d": [1.0],
    })
    out = compute_bucket_stats(df, "bucket", 5)
    forbidden_tokens = ("best", "optimal", "threshold", "recommend", "selected")
    for col in out.columns:
        assert not any(tok in col.lower() for tok in forbidden_tokens), col


def _synthetic_features_outcomes():
    dates = pd.bdate_range("2024-01-02", periods=30)
    tickers = [f"T{i:02d}" for i in range(20)]
    rng = np.random.default_rng(7)
    feat_rows, out_rows = [], []
    for d in dates:
        for t in tickers:
            rs = rng.uniform(0, 100)
            feat_rows.append({
                "date": d, "ticker": t, "eligible": True,
                "rs_percentile_1d": rs, "rs_percentile_1w": rs, "rs_percentile_1m": rs,
                "rs_percentile_3m": rs if d >= dates[10] else np.nan,
                "rs_percentile_6m": np.nan, "rs_percentile_12m": np.nan,
            })
            out_rows.append({
                "date": d, "ticker": t,
                "mfe_pct_5d": rng.normal(5, 2), "mae_pct_5d": -rng.normal(3, 1),
                "forward_return_close_5d": rng.normal(1, 3),
                "reached_plus_5pct_5d": float(rng.random() < 0.5),
                "reached_plus_10pct_5d": float(rng.random() < 0.2),
                "reached_2atr_5d": float(rng.random() < 0.3),
                "reached_3atr_5d": float(rng.random() < 0.1),
                "reached_plus_5_before_minus_5_5d": float(rng.random() < 0.35),
                "reached_plus_10_before_minus_5_5d": float(rng.random() < 0.15),
                "mfe_pct_10d": rng.normal(8, 3), "mae_pct_10d": -rng.normal(4, 1),
                "forward_return_close_10d": rng.normal(2, 4),
                "reached_plus_5pct_10d": float(rng.random() < 0.6),
                "reached_plus_10pct_10d": float(rng.random() < 0.3),
                "reached_2atr_10d": float(rng.random() < 0.4),
                "reached_3atr_10d": float(rng.random() < 0.15),
                "reached_plus_5_before_minus_5_10d": float(rng.random() < 0.45),
                "reached_plus_10_before_minus_5_10d": float(rng.random() < 0.25),
                "mfe_pct_20d": rng.normal(12, 4), "mae_pct_20d": -rng.normal(5, 2),
                "forward_return_close_20d": rng.normal(3, 5),
                "reached_plus_5pct_20d": float(rng.random() < 0.7),
                "reached_plus_10pct_20d": float(rng.random() < 0.4),
                "reached_2atr_20d": float(rng.random() < 0.5),
                "reached_3atr_20d": float(rng.random() < 0.2),
                "reached_plus_5_before_minus_5_20d": float(rng.random() < 0.55),
                "reached_plus_10_before_minus_5_20d": float(rng.random() < 0.35),
            })
    # A few non-eligible rows that must be excluded entirely.
    feat_rows.append({
        "date": dates[0], "ticker": "INELIGIBLE", "eligible": False,
        "rs_percentile_1d": 99.0, "rs_percentile_1w": 99.0, "rs_percentile_1m": 99.0,
        "rs_percentile_3m": np.nan, "rs_percentile_6m": np.nan, "rs_percentile_12m": np.nan,
    })
    out_rows.append({
        "date": dates[0], "ticker": "INELIGIBLE",
        **{c: 0.0 for c in out_rows[0] if c not in ("date", "ticker")},
    })
    return pd.DataFrame(feat_rows), pd.DataFrame(out_rows)


def test_build_rs_benchmark_report_shape_and_eligibility_filter():
    features, outcomes = _synthetic_features_outcomes()
    tables = build_rs_benchmark_report(features, outcomes)

    assert len(tables) == 6 * 2 * 3  # 6 RS horizons x 2 bucket schemes x 3 outcome horizons
    assert "rs_percentile_1d__coarse__10d" in tables
    assert "rs_percentile_12m__fine_above_80__20d" in tables

    # Ineligible row's extreme RS (99.0) must not leak into any bucket's n.
    coarse_1d_5d = tables["rs_percentile_1d__coarse__5d"]
    assert coarse_1d_5d["n"].sum() == 20 * 30  # eligible rows only, INELIGIBLE excluded

    # rs_percentile_12m is entirely NaN in this synthetic fixture -> every
    # bucket must show n=0, not be silently dropped from the table.
    rs12_table = tables["rs_percentile_12m__coarse__10d"]
    assert set(rs12_table["n"].unique()) == {0}

    # rs_percentile_3m only populated for the second half of dates -> partial coverage visible via n.
    rs3m_table = tables["rs_percentile_3m__coarse__5d"]
    assert rs3m_table["n"].sum() == 20 * 20  # 20 of 30 dates (dates[10:]) x 20 tickers


def test_build_rs_benchmark_report_missing_columns_raises():
    with pytest.raises(ValueError):
        build_rs_benchmark_report(pd.DataFrame({"date": [1], "ticker": ["A"]}), pd.DataFrame())


def test_cli_rejects_cross_year_range(tmp_path, monkeypatch):
    import argparse
    import yolo_calibration.cli as cli
    import yolo_calibration.data.storage as storage

    monkeypatch.setattr(storage, "PROCESSED_DIR", tmp_path / "processed")
    args = argparse.Namespace(start=pd.Timestamp("2024-01-01").date(), end=pd.Timestamp("2025-01-05").date())
    assert cli.cmd_build_rs_benchmark_report(args) == 1


def test_cli_writes_csv_tables_for_a_single_year(tmp_path, monkeypatch):
    import argparse
    import yolo_calibration.cli as cli
    import yolo_calibration.data.storage as storage

    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(storage, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

    features, outcomes = _synthetic_features_outcomes()
    storage.write_processed_by_year("stock_features_daily", features)
    storage.write_processed_by_year("stock_outcomes_daily", outcomes)

    args = argparse.Namespace(start=pd.Timestamp("2024-01-02").date(), end=pd.Timestamp("2024-12-31").date())
    rc = cli.cmd_build_rs_benchmark_report(args)
    assert rc == 0

    out_dir = tmp_path / "reports" / "rs_benchmark_2024"
    csvs = list(out_dir.glob("*.csv"))
    assert len(csvs) == 6 * 2 * 3
    assert (out_dir / "rs_percentile_1d__coarse__10d.csv").exists()
