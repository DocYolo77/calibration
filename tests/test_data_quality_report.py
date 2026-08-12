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
