import pandas as pd

from yolo_calibration.features.technical import compute_adr20


def test_adr20_exact_formula():
    dates = pd.bdate_range("2023-01-02", periods=25)
    # Constant (High/Low - 1)*100 = 5.0% every day -> ADR20 must be exactly 5.0
    df = pd.DataFrame({
        "date": dates,
        "ticker": "AAA",
        "high": [105.0] * 25,
        "low": [100.0] * 25,
        "close": [102.0] * 25,
    })
    out = compute_adr20(df)
    valid = out.dropna(subset=["adr20"])
    assert len(valid) == 25 - 19  # first full 20-day window completes at row index 19
    assert (valid["adr20"].round(6) == 5.0).all()


def test_adr20_uses_trailing_20_days_only():
    dates = pd.bdate_range("2023-01-02", periods=25)
    high_low_pct = [2.0] * 20 + [10.0] * 5  # range jumps on day 21+
    df = pd.DataFrame({
        "date": dates,
        "ticker": "AAA",
        "high": [100 * (1 + p / 100) for p in high_low_pct],
        "low": [100.0] * 25,
        "close": [101.0] * 25,
    })
    out = compute_adr20(df)
    # Row 19 (0-indexed, the 20th day) = mean of the first 20 days' 2.0% range
    row19 = out.iloc[19]
    assert abs(row19["adr20"] - 2.0) < 1e-6
    # Last row's window is days 6..25 -> 15 days at 2.0% + 5 days at 10.0%
    last = out.iloc[-1]
    expected = (15 * 2.0 + 5 * 10.0) / 20
    assert abs(last["adr20"] - expected) < 1e-6


def test_adr20_respects_ticker_boundaries(two_ticker_ohlcv):
    out = compute_adr20(two_ticker_ohlcv)
    # No adr20 value for AAA should be computed from BBB's rows and vice versa.
    for ticker in ["AAA", "BBB"]:
        sub = out[out["ticker"] == ticker].sort_values("date")
        assert sub["adr20"].notna().sum() == len(sub) - 19
