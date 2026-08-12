import numpy as np
import pandas as pd

from yolo_calibration.outcomes.build_outcomes import (
    build_horizon_outcomes,
    build_stock_outcomes_daily,
    compute_new_20d_high_flag,
)


def _simple_series():
    # 10 trading days of a single ticker with hand-picked highs/lows so
    # forward-window math can be checked by hand.
    dates = pd.bdate_range("2023-01-02", periods=10)
    close = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    high = [100, 108, 103, 96, 112, 101, 101, 101, 101, 101]
    low = [100, 99, 97, 94, 99, 99, 99, 99, 99, 99]
    df = pd.DataFrame({
        "date": dates, "ticker": "AAA", "close": close, "high": high, "low": low,
        "atr14": [2.0] * 10,
    })
    df["is_new_20d_high"] = False
    return df


def test_forward_return_close():
    df = _simple_series()
    out = build_horizon_outcomes(df, 3)
    # Signal day 0: close[0]=100, close[3]=100 -> forward_return = 0%
    assert abs(out.iloc[0]["forward_return_close_3d"] - 0.0) < 1e-9


def test_mfe_mae_are_window_extremes_not_just_endpoint():
    df = _simple_series()
    out = build_horizon_outcomes(df, 4)
    # Signal day 0, window = days 1..4 -> highs [108,103,96,112], lows [99,97,94,99]
    row0 = out.iloc[0]
    assert abs(row0["mfe_pct_4d"] - 12.0) < 1e-9   # max high 112 vs close 100 -> +12%
    assert abs(row0["mae_pct_4d"] - -6.0) < 1e-9    # min low 94 vs close 100 -> -6%


def test_atr_multiples():
    df = _simple_series()
    out = build_horizon_outcomes(df, 4)
    row0 = out.iloc[0]
    # mfe in price terms = 112-100=12, atr=2.0 -> 6.0x
    assert abs(row0["mfe_atr_multiple_4d"] - 6.0) < 1e-9
    assert row0["reached_2atr_4d"] == True  # noqa: E712
    assert row0["reached_3atr_4d"] == True  # noqa: E712


def test_no_future_leakage_incomplete_windows_are_nan():
    df = _simple_series()
    out = build_horizon_outcomes(df, 5)
    # Last row (index 9) has no future days at all -> everything NaN, not 0 or fabricated.
    last = out.iloc[9]
    assert pd.isna(last["forward_return_close_5d"])
    assert pd.isna(last["mfe_pct_5d"])
    assert pd.isna(last["mae_pct_5d"])


def test_reached_plus_before_minus_tie_break_and_ordering():
    dates = pd.bdate_range("2023-01-02", periods=4)
    # Day0 signal, close=100. Day1: up move to +6% first (high), down never happens.
    df_up_first = pd.DataFrame({
        "date": dates, "ticker": "AAA",
        "close": [100, 100, 100, 100],
        "high": [100, 106, 100, 100],
        "low": [100, 100, 100, 100],
        "atr14": [2.0] * 4,
        "is_new_20d_high": [False] * 4,
    })
    out = build_horizon_outcomes(df_up_first, 3)
    assert out.iloc[0]["reached_plus_5_before_minus_5_3d"] == True  # noqa: E712

    # Down move happens on day1, up move happens on day2 -> down came first.
    df_down_first = pd.DataFrame({
        "date": dates, "ticker": "AAA",
        "close": [100, 100, 100, 100],
        "high": [100, 100, 106, 100],
        "low": [100, 94, 94, 94],
        "atr14": [2.0] * 4,
        "is_new_20d_high": [False] * 4,
    })
    out2 = build_horizon_outcomes(df_down_first, 3)
    assert out2.iloc[0]["reached_plus_5_before_minus_5_3d"] == False  # noqa: E712


def test_new_20d_high_flag_no_lookahead():
    dates = pd.bdate_range("2023-01-02", periods=25)
    high = [100.0] * 20 + [90.0, 90.0, 150.0, 90.0, 90.0]  # spike on day 22 (index 22)
    low = [95.0] * 25
    df = pd.DataFrame({"date": dates, "ticker": "AAA", "high": high, "low": low})
    flags = compute_new_20d_high_flag(df, 20)
    assert flags.iloc[22] == True  # noqa: E712  150 > trailing max of prior 20 days (100)
    assert flags.iloc[20] == False  # noqa: E712  90 < trailing max (100)
    # First 20 rows have no full trailing window -> cannot be a "new" high by definition
    assert flags.iloc[:20].sum() == 0


def test_signal_day_atr_used_not_future_atr():
    """The ATR fed into mfe_atr_multiple must be the ATR AS OF the signal
    day, never an ATR computed from data after it."""
    dates = pd.bdate_range("2023-01-02", periods=5)
    df = pd.DataFrame({
        "date": dates, "ticker": "AAA",
        "close": [100] * 5,
        "high": [100, 110, 100, 100, 100],
        "low": [100, 100, 100, 100, 100],
        "atr14": [1.0, 999.0, 1.0, 1.0, 1.0],  # signal day (0) ATR = 1.0, NOT the inflated future value
        "is_new_20d_high": [False] * 5,
    })
    out = build_horizon_outcomes(df, 2)
    row0 = out.iloc[0]
    # mfe = 110-100=10, atr at signal day (0) = 1.0 -> multiple = 10.0, not 10/999
    assert abs(row0["mfe_atr_multiple_2d"] - 10.0) < 1e-6


def test_build_stock_outcomes_daily_end_to_end_multi_ticker(two_ticker_ohlcv):
    df = two_ticker_ohlcv.copy()
    df["atr14"] = 2.0  # stub signal-day ATR
    out = build_stock_outcomes_daily(df)
    assert set(out["ticker"].unique()) == {"AAA", "BBB"}
    assert "mfe_pct_5d" in out.columns and "mae_pct_20d" in out.columns
    # No row should have a forward_return_close_20d for the last 20 rows of each ticker.
    for ticker in ["AAA", "BBB"]:
        sub = out[out["ticker"] == ticker].sort_values("date")
        assert sub["forward_return_close_20d"].iloc[-20:].isna().all()
