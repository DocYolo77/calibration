import numpy as np
import pandas as pd

from yolo_calibration.features.technical import add_atr_extension, add_moving_averages, add_true_range_atr


def test_true_range_formula():
    dates = pd.bdate_range("2023-01-02", periods=3)
    df = pd.DataFrame({
        "date": dates,
        "ticker": "AAA",
        "high": [10.0, 12.0, 9.0],
        "low": [8.0, 9.0, 7.0],
        "close": [9.0, 11.5, 7.5],
    })
    out = add_true_range_atr(df)
    # Row 0: no previous close -> TR = High-Low = 2.0
    assert abs(out.iloc[0]["true_range"] - 2.0) < 1e-9
    # Row 1: prev close = 9.0. TR = max(12-9, |12-9|, |9-9|) = max(3,3,0) = 3.0
    assert abs(out.iloc[1]["true_range"] - 3.0) < 1e-9
    # Row 2: prev close = 11.5. TR = max(9-7, |9-11.5|, |7-11.5|) = max(2, 2.5, 4.5) = 4.5
    assert abs(out.iloc[2]["true_range"] - 4.5) < 1e-9


def test_atr14_is_simple_mean_not_wilder():
    dates = pd.bdate_range("2023-01-02", periods=20)
    # Constant true range of exactly 3.0 every day (high-low=3, no gaps)
    highs = [100 + 3.0] * 20
    lows = [100.0] * 20
    closes = [101.5] * 20
    df = pd.DataFrame({"date": dates, "ticker": "AAA", "high": highs, "low": lows, "close": closes})
    out = add_true_range_atr(df)
    row13 = out.iloc[13]  # 14th row: first fully-populated 14-day ATR window
    assert abs(row13["atr14"] - 3.0) < 1e-9

    # Now introduce one large TR spike on day 14 (index 13's predecessor window) and confirm
    # the simple mean (not exponentially-decaying Wilder smoothing) drops it out cleanly
    # once it exits the trailing-14 window.
    highs2 = [100 + 3.0] * 5 + [130.0] + [100 + 3.0] * 14
    lows2 = [100.0] * 5 + [100.0] + [100.0] * 14
    closes2 = [101.5] * 5 + [101.5] + [101.5] * 14
    df2 = pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=20), "ticker": "AAA",
                        "high": highs2, "low": lows2, "close": closes2})
    out2 = add_true_range_atr(df2)
    # Window for the LAST row (index 19) spans indices 6..19 -> does not include the spike (index 5)
    last_atr = out2.iloc[-1]["atr14"]
    assert abs(last_atr - 3.0) < 1e-9  # spike has fully rolled out under simple-mean ATR


def test_atr_pct_and_extension_formulas():
    dates = pd.bdate_range("2023-01-02", periods=60)
    df = pd.DataFrame({
        "date": dates,
        "ticker": "AAA",
        "high": np.full(60, 105.0),
        "low": np.full(60, 95.0),
        "close": np.full(60, 100.0),
    })
    df = add_true_range_atr(df)
    df = add_moving_averages(df)
    df = add_atr_extension(df)
    row = df.dropna(subset=["atr14", "sma50"]).iloc[0]
    expected_atr_pct = row["atr14"] / row["close"] * 100.0
    assert abs(row["atr_pct"] - expected_atr_pct) < 1e-9
    expected_gain = (row["close"] - row["sma50"]) / row["sma50"] * 100.0
    expected_ext = expected_gain / expected_atr_pct
    assert abs(row["atr_extension"] - expected_ext) < 1e-9
