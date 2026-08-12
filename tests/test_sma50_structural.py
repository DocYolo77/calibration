import numpy as np
import pandas as pd

from yolo_calibration.config import load_features_config
from yolo_calibration.features.technical import add_moving_averages, add_sma50_persistence, add_sma50_slope


def _build(df: pd.DataFrame) -> pd.DataFrame:
    out = add_moving_averages(df)
    out = add_sma50_slope(out)
    out = add_sma50_persistence(out)
    return out


def test_sma50_slope_pct_formula():
    lookback = load_features_config()["sma50_slope"]["lookback_days"]
    dates = pd.bdate_range("2023-01-02", periods=70)
    # Linear uptrend -> SMA50 itself trends up smoothly, easy to hand-check.
    close = [100 + 0.5 * i for i in range(70)]
    df = pd.DataFrame({"date": dates, "ticker": "AAA", "close": close, "high": close, "low": close})
    out = _build(df)

    valid = out.dropna(subset=["sma50_slope_pct"])
    assert not valid.empty
    row = valid.iloc[-1]
    idx = out.index[out["date"] == row["date"]][0]
    sma_now = out.loc[idx, "sma50"]
    sma_lag = out.loc[idx - lookback, "sma50"]
    expected = (sma_now - sma_lag) / sma_lag * 100.0
    assert abs(row["sma50_slope_pct"] - expected) < 1e-9


def test_sma50_slope_nan_during_warmup():
    dates = pd.bdate_range("2023-01-02", periods=10)
    close = [100.0] * 10
    df = pd.DataFrame({"date": dates, "ticker": "AAA", "close": close, "high": close, "low": close})
    out = _build(df)
    assert out["sma50_slope_pct"].isna().all()  # SMA50 itself has no valid values yet


def test_sma50_persistence_streak_and_flip():
    dates = pd.bdate_range("2023-01-02", periods=60)
    # First 50 rows warm up SMA50 at a flat 100. From row 50 (0-indexed),
    # close jumps above SMA50 for 5 days, then drops below for 4 days.
    close = [100.0] * 50 + [105.0] * 5 + [95.0] * 4 + [96.0]
    df = pd.DataFrame({"date": dates, "ticker": "AAA", "close": close,
                        "high": close, "low": close})
    out = _build(df)

    # Row 49 (0-indexed) is the first row with a valid SMA50 (window=50).
    # At that point close==sma50==100 exactly -> NOT above -> counted as a
    # "below" streak of length 1 (persistence uses strict > for "above").
    assert out.loc[49, "sma50_persistence_days"] == -1

    # Rows 50-54: close=105 > sma50 (still influenced by the 100-run, but
    # > holds) -> an ABOVE streak building 1,2,3,4,5.
    above_streak = out.loc[50:54, "sma50_persistence_days"].tolist()
    assert above_streak == [1.0, 2.0, 3.0, 4.0, 5.0]

    # Rows 55-58: close=95 < sma50 -> a BELOW streak (negative), resets to -1
    # then counts down further.
    below_streak = out.loc[55:58, "sma50_persistence_days"].tolist()
    assert below_streak == [-1.0, -2.0, -3.0, -4.0]


def test_sma50_persistence_respects_ticker_boundaries():
    dates = pd.bdate_range("2023-01-02", periods=55)
    close_a = [100.0] * 50 + [110.0] * 5   # ends in a 5-day above-streak
    close_b = [100.0] * 50 + [90.0] * 5    # ends in a 5-day below-streak
    df = pd.concat([
        pd.DataFrame({"date": dates, "ticker": "AAA", "close": close_a, "high": close_a, "low": close_a}),
        pd.DataFrame({"date": dates, "ticker": "BBB", "close": close_b, "high": close_b, "low": close_b}),
    ], ignore_index=True)
    out = _build(df)

    a_last = out[(out["ticker"] == "AAA")].sort_values("date").iloc[-1]
    b_last = out[(out["ticker"] == "BBB")].sort_values("date").iloc[-1]
    # AAA flips to ABOVE at the SMA50 warmup boundary (close==sma50 there
    # counts as "below", so the above-streak is exactly the 5 new days).
    assert a_last["sma50_persistence_days"] == 5.0
    # BBB's warmup-boundary row (close==sma50) is ALSO "below", extending
    # (not resetting) the below-streak across the boundary: 6 rows total.
    assert b_last["sma50_persistence_days"] == -6.0
