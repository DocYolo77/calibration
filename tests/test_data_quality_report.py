import numpy as np
import pandas as pd

from yolo_calibration.reports.data_quality import build_data_quality_report


def _base_universe():
    dates = pd.bdate_range("2023-01-02", periods=5)
    rows = []
    for d in dates:
        rows.append({"date": d, "ticker": "NORMAL", "close": 100.0, "high": 101.0, "low": 99.0,
                      "eligible": True, "market_cap": 5e9})
    return pd.DataFrame(rows)


def test_implausible_price_flagged_but_not_dropped():
    """A close price above $1,000,000 (a level no legitimate US common
    stock has ever traded at) must be surfaced as a warning/diagnostic
    count, never silently corrected or removed from the report inputs."""
    universe = _base_universe()
    bad_row = pd.DataFrame([{
        "date": universe["date"].iloc[0], "ticker": "GLITCH", "close": 5_000_000_000.0,
        "high": 5_100_000_000.0, "low": 4_900_000_000.0, "eligible": False, "market_cap": None,
    }])
    universe = pd.concat([universe, bad_row], ignore_index=True)

    report = build_data_quality_report(
        market_universe_daily=universe,
        stock_features_daily=pd.DataFrame({"a": [1]}),
        stock_outcomes_daily=pd.DataFrame({"a": [1]}),
        qqq_health_daily=None,
        qqq_health_outcomes_daily=None,
        qqq_constituent_component_counts=None,
        qqq_health_error="unavailable in this test",
    )
    assert report["implausible_price_row_count"] == 1
    assert report["implausible_price_ticker_count"] == 1
    assert any("implausible" in w.lower() or "1,000,000" in w for w in report["warnings"])


def test_market_cap_sensitivity_flags_near_threshold_and_volatile_tickers():
    """A ticker sitting within the near-threshold band whose market_cap
    swings >3x month-over-month (the MULN-style shares-outstanding
    instability pattern) must be flagged by the sensitivity diagnostic,
    while a stable near-threshold ticker and a far-from-threshold volatile
    ticker must NOT trigger the volatile-ticker flag."""
    rows = []
    # STABLE_NEAR: consistently near the $1B line, no big swings.
    for i, d in enumerate(pd.bdate_range("2023-01-02", periods=40)):
        rows.append({"date": d, "ticker": "STABLE_NEAR", "close": 10.0, "high": 10.1, "low": 9.9,
                      "eligible": (i % 2 == 0), "market_cap": 1.02e9})
    # VOLATILE_NEAR: near the threshold AND swings >3x month over month.
    for i, d in enumerate(pd.bdate_range("2023-01-02", periods=25)):
        rows.append({"date": d, "ticker": "VOLATILE_NEAR", "close": 1.0, "high": 1.05, "low": 0.95,
                      "eligible": False, "market_cap": 9.0e8})
    for i, d in enumerate(pd.bdate_range("2023-03-01", periods=25)):
        rows.append({"date": d, "ticker": "VOLATILE_NEAR", "close": 1.0, "high": 1.05, "low": 0.95,
                      "eligible": True, "market_cap": 5.0e9})
    # VOLATILE_FAR: swings a lot but never anywhere near $1B -> must not be flagged.
    for i, d in enumerate(pd.bdate_range("2023-01-02", periods=25)):
        rows.append({"date": d, "ticker": "VOLATILE_FAR", "close": 1.0, "high": 1.05, "low": 0.95,
                      "eligible": False, "market_cap": 1.0e6})
    for i, d in enumerate(pd.bdate_range("2023-03-01", periods=25)):
        rows.append({"date": d, "ticker": "VOLATILE_FAR", "close": 1.0, "high": 1.05, "low": 0.95,
                      "eligible": False, "market_cap": 5.0e7})
    universe = pd.DataFrame(rows)

    report = build_data_quality_report(
        market_universe_daily=universe,
        stock_features_daily=pd.DataFrame({"a": [1]}),
        stock_outcomes_daily=pd.DataFrame({"a": [1]}),
        qqq_health_daily=None,
        qqq_health_outcomes_daily=None,
        qqq_constituent_component_counts=None,
        qqq_health_error="unavailable in this test",
    )
    sens = report["market_cap_eligibility_sensitivity"]
    assert sens["near_threshold_ticker_count"] == 2
    assert sens["near_threshold_volatile_ticker_count"] == 1
    assert sens["near_threshold_volatile_tickers_example"] == ["VOLATILE_NEAR"]
    assert any("near" in w.lower() and "1B" in w for w in report["warnings"])


def test_tie_break_diagnostics_counts_same_day_ties_from_outcomes():
    outcomes = pd.DataFrame([
        {"date": pd.Timestamp("2023-01-02"), "ticker": "AAA",
         "reached_plus_5_before_minus_5_5d": False, "reached_plus_5_before_minus_5_tie_5d": True},
        {"date": pd.Timestamp("2023-01-03"), "ticker": "AAA",
         "reached_plus_5_before_minus_5_5d": True, "reached_plus_5_before_minus_5_tie_5d": False},
        {"date": pd.Timestamp("2023-01-04"), "ticker": "AAA",
         "reached_plus_5_before_minus_5_5d": np.nan, "reached_plus_5_before_minus_5_tie_5d": np.nan},
    ])
    report = build_data_quality_report(
        market_universe_daily=pd.DataFrame(),
        stock_features_daily=pd.DataFrame({"a": [1]}),
        stock_outcomes_daily=outcomes,
        qqq_health_daily=None,
        qqq_health_outcomes_daily=None,
        qqq_constituent_component_counts=None,
        qqq_health_error="unavailable in this test",
    )
    diag = report["reached_before_tie_break_diagnostics"]
    assert diag["total_ties"] == 1
    col_stats = diag["columns"]["reached_plus_5_before_minus_5_tie_5d"]
    assert col_stats["tie_count"] == 1
    assert col_stats["window_complete_count"] == 2  # NaN (incomplete window) row excluded
    assert any("same-day" in w.lower() or "not determinable" in w.lower() for w in report["warnings"])


def test_no_implausible_prices_means_zero_count_no_extra_warning():
    universe = _base_universe()
    report = build_data_quality_report(
        market_universe_daily=universe,
        stock_features_daily=pd.DataFrame({"a": [1]}),
        stock_outcomes_daily=pd.DataFrame({"a": [1]}),
        qqq_health_daily=None,
        qqq_health_outcomes_daily=None,
        qqq_constituent_component_counts=None,
        qqq_health_error="unavailable in this test",
    )
    assert report["implausible_price_row_count"] == 0
    assert not any("1,000,000" in w for w in report["warnings"])
