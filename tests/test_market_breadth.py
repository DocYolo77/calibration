"""outcomes/build_market_breadth.py: verifies future_rs{bucket}plus_share_{h}d
is computed on D+H's OWN actually-eligible universe, NOT the D0 cohort
followed forward (the bug fixed 2026-08-13 — see module docstring)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from yolo_calibration.outcomes.build_market_breadth import (
    build_d0_cohort_forward_performance,
    build_future_market_breadth,
    compute_daily_rs_breadth,
)


def test_compute_daily_rs_breadth_uses_that_dates_own_eligible_universe():
    dates = pd.bdate_range("2023-01-02", periods=3)
    df = pd.DataFrame([
        {"date": dates[0], "ticker": "A", "eligible": True, "rs_percentile_1m": 50.0},
        {"date": dates[0], "ticker": "B", "eligible": True, "rs_percentile_1m": 90.0},
        {"date": dates[0], "ticker": "C", "eligible": False, "rs_percentile_1m": 99.0},  # excluded: not eligible
        {"date": dates[1], "ticker": "A", "eligible": True, "rs_percentile_1m": 85.0},
        {"date": dates[1], "ticker": "B", "eligible": True, "rs_percentile_1m": 85.0},
    ])
    out = compute_daily_rs_breadth(df, rs_buckets=[80, 90])

    row0 = out[out["date"] == dates[0]].iloc[0]
    assert row0["eligible_count"] == 2
    assert row0["breadth_rs80plus_share"] == 0.5   # B only (C excluded: not eligible)
    assert row0["breadth_rs90plus_share"] == 0.5   # B only (>=90)

    row1 = out[out["date"] == dates[1]].iloc[0]
    assert row1["eligible_count"] == 2
    assert row1["breadth_rs80plus_share"] == 1.0   # both A and B >= 80
    assert row1["breadth_rs90plus_share"] == 0.0   # neither >= 90


def test_future_breadth_reflects_dPlusH_universe_not_d0_cohort():
    """Regression test for the 2026-08-13 fix: ticker B is eligible (and
    RS<80) at D0, then drops OUT of eligibility by D+5; ticker C is
    ineligible at D0 but becomes eligible (with RS>=80) by D+5. The OLD
    (buggy) cohort-following logic would have produced a D0-row share of
    0/2=0.0 (B's own D+5 RS checked, C never considered since it wasn't in
    the D0 cohort). The correct D+5-universe-based share is 1/2=0.5 (A: no,
    C: yes; B is excluded because it is no longer eligible at D+5)."""
    dates = pd.bdate_range("2023-01-02", periods=15)
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "ticker": "A", "eligible": True, "rs_percentile_1m": 50.0})
        rows.append({"date": d, "ticker": "B", "eligible": i < 5, "rs_percentile_1m": 70.0})
        rows.append({"date": d, "ticker": "C", "eligible": i >= 5, "rs_percentile_1m": 95.0})
    features = pd.DataFrame(rows)

    daily_breadth = compute_daily_rs_breadth(features, rs_buckets=[80])
    future = build_future_market_breadth(daily_breadth, horizons=[5], rs_buckets=[80])

    row0 = future[future["date"] == dates[0]].iloc[0]
    assert row0["future_rs80plus_share_5d"] == 0.5  # NEW (correct): D+5 universe = {A, C}

    # Sanity: prove this is NOT what the old D0-cohort-following approach
    # would have produced (0.0 — B's own D+5 RS of 70 is < 80, C never
    # considered since it wasn't eligible at D0).
    old_style_cohort = features[(features["date"] == dates[0]) & (features["eligible"])]["ticker"].tolist()
    assert set(old_style_cohort) == {"A", "B"}
    old_style_hits = features[
        (features["date"] == dates[5]) & (features["ticker"].isin(old_style_cohort))
    ]
    old_style_share = (old_style_hits["rs_percentile_1m"] >= 80).mean()
    assert old_style_share == 0.0
    assert row0["future_rs80plus_share_5d"] != old_style_share


def test_future_breadth_nan_when_window_incomplete():
    dates = pd.bdate_range("2023-01-02", periods=5)
    daily_breadth = pd.DataFrame({
        "date": dates, "eligible_count": [2] * 5, "breadth_rs80plus_share": [0.5] * 5,
    })
    future = build_future_market_breadth(daily_breadth, horizons=[3], rs_buckets=[80])
    # Last 3 rows have no D+3 date available -> NaN, not a silently wrong 0.
    assert future["future_rs80plus_share_3d"].iloc[:2].notna().all()
    assert future["future_rs80plus_share_3d"].iloc[2:].isna().all()


def test_d0_cohort_forward_performance_only_includes_eligible_rows():
    dates = pd.bdate_range("2023-01-02", periods=3)
    features = pd.DataFrame([
        {"date": dates[0], "ticker": "A", "eligible": True},
        {"date": dates[0], "ticker": "B", "eligible": False},
    ])
    outcomes = pd.DataFrame([
        {"date": dates[0], "ticker": "A", "forward_return_close_5d": 10.0, "mfe_pct_5d": 12.0,
         "mae_pct_5d": -3.0, "reached_plus_5pct_5d": True, "reached_plus_10pct_5d": True,
         "reached_2atr_5d": False, "reached_3atr_5d": False},
        {"date": dates[0], "ticker": "B", "forward_return_close_5d": -50.0, "mfe_pct_5d": 1.0,
         "mae_pct_5d": -50.0, "reached_plus_5pct_5d": False, "reached_plus_10pct_5d": False,
         "reached_2atr_5d": False, "reached_3atr_5d": False},
    ])
    out = build_d0_cohort_forward_performance(features, outcomes, horizons=[5])
    row = out[out["date"] == dates[0]].iloc[0]
    assert row["cohort_size_5d"] == 1  # B excluded: not eligible at D0
    assert row["median_forward_return_5d"] == 10.0
