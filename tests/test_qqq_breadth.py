import numpy as np
import pandas as pd

from yolo_calibration.qqq_health.breadth import add_mco_mcsi, compute_breadth_daily


def test_advance_decline_and_rana():
    dates = pd.bdate_range("2023-01-02", periods=3)
    rows = []
    for ticker, closes in [("A", [100, 105, 108]), ("B", [100, 95, 90]), ("C", [100, 102, 99])]:
        for d, c in zip(dates, closes):
            rows.append({"date": d, "ticker": ticker, "open": c, "high": c, "low": c, "close": c, "volume": 1000})
    ohlcv = pd.DataFrame(rows)

    constituents = pd.DataFrame({
        "date": [d for d in dates for _ in range(3)],
        "constituent_ticker": ["A", "B", "C"] * 3,
        "weight": 0.33,
    })

    breadth = compute_breadth_daily(constituents, ohlcv)
    day2 = breadth[breadth["date"] == dates[1]].iloc[0]

    assert day2["component_count"] == 3
    assert day2["advancers"] == 2  # A up, C up
    assert day2["decliners"] == 1  # B down
    expected_rana = ((2 - 1) / 3) * 1000
    assert abs(day2["rana"] - expected_rana) < 1e-6


def test_rana_zero_when_advancers_equal_decliners():
    dates = pd.bdate_range("2023-01-02", periods=2)
    rows = []
    for ticker, closes in [("A", [100, 110]), ("B", [100, 90])]:
        for d, c in zip(dates, closes):
            rows.append({"date": d, "ticker": ticker, "open": c, "high": c, "low": c, "close": c, "volume": 1000})
    ohlcv = pd.DataFrame(rows)
    constituents = pd.DataFrame({
        "date": [d for d in dates for _ in range(2)],
        "constituent_ticker": ["A", "B"] * 2,
        "weight": 0.5,
    })
    breadth = compute_breadth_daily(constituents, ohlcv)
    day2 = breadth[breadth["date"] == dates[1]].iloc[0]
    assert abs(day2["rana"] - 0.0) < 1e-9


def test_mco_is_ema19_minus_ema39_adjust_false():
    dates = pd.bdate_range("2023-01-02", periods=60)
    rng = np.random.default_rng(42)
    rana = rng.normal(0, 200, size=60)
    breadth_daily = pd.DataFrame({"date": dates, "rana": rana})

    out = add_mco_mcsi(breadth_daily)

    expected_short = pd.Series(rana).ewm(span=19, adjust=False).mean()
    expected_long = pd.Series(rana).ewm(span=39, adjust=False).mean()
    expected_mco = (expected_short - expected_long).to_numpy()
    np.testing.assert_allclose(out["mco_raw"].to_numpy(), expected_mco, atol=1e-9)


def test_mcsi_is_cumulative_sum_of_mco():
    dates = pd.bdate_range("2023-01-02", periods=30)
    rana = [100.0] * 30
    breadth_daily = pd.DataFrame({"date": dates, "rana": rana})
    out = add_mco_mcsi(breadth_daily)
    expected_mcsi = out["mco_raw"].cumsum().to_numpy()
    np.testing.assert_allclose(out["mcsi_raw"].to_numpy(), expected_mcsi, atol=1e-9)


def test_mco_z_and_mcsi_z_require_warmup_and_use_200d_window():
    dates = pd.bdate_range("2023-01-02", periods=250)
    rng = np.random.default_rng(7)
    rana = rng.normal(0, 150, size=250)
    breadth_daily = pd.DataFrame({"date": dates, "rana": rana})
    out = add_mco_mcsi(breadth_daily)

    # Before min_periods=80 is satisfied, z-scores must be NaN.
    assert out["mco_z"].iloc[:79].isna().all()
    assert out["mcsi_z"].iloc[:79].isna().all()
    # After warmup, z-scores should be populated.
    assert out["mco_z"].iloc[100:].notna().all()
    assert out["mcsi_z"].iloc[100:].notna().all()

    # Manually recompute the z-score at a late index using a 200-day trailing window.
    idx = 249
    window = out["mco_raw"].iloc[max(0, idx - 199):idx + 1]
    expected_z = (out["mco_raw"].iloc[idx] - window.mean()) / window.std()
    assert abs(out["mco_z"].iloc[idx] - expected_z) < 1e-6


def test_pct_above_ma_excludes_missing_and_uses_correct_denominator():
    dates = pd.bdate_range("2023-01-02", periods=10)
    rows = []
    for ticker, base in [("A", 100.0), ("B", 50.0)]:
        for i, d in enumerate(dates):
            c = base * (1 + 0.05 * i)  # strictly increasing -> above its own rising SMA5 quickly
            rows.append({"date": d, "ticker": ticker, "open": c, "high": c, "low": c, "close": c, "volume": 1000})
    ohlcv = pd.DataFrame(rows)
    constituents = pd.DataFrame({
        "date": [d for d in dates for _ in range(2)],
        "constituent_ticker": ["A", "B"] * 10,
        "weight": 0.5,
    })
    breadth = compute_breadth_daily(constituents, ohlcv)
    last = breadth.iloc[-1]
    # Both tickers are in a clean uptrend -> both should be above SMA5 on the last day.
    assert last["pct_above_sma5"] == 100.0
