"""features/repeat_offender.py: point-in-time previous-extension/"repeat
offender" history. No future leakage anywhere -- every test here either
directly checks a hand-computed value or checks the peak-before-signal-day
invariant structurally."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yolo_calibration.features.repeat_offender import compute_prior_extension_history


def _single_ticker_df(atr_extension, distance_ema10_pct=None, distance_ema20_pct=None, ticker="A"):
    n = len(atr_extension)
    dates = pd.bdate_range("2024-01-02", periods=n)
    if distance_ema10_pct is None:
        distance_ema10_pct = [0.0] * n
    if distance_ema20_pct is None:
        distance_ema20_pct = list(distance_ema10_pct)
    return pd.DataFrame({
        "date": dates, "ticker": ticker,
        "atr_extension": atr_extension,
        "distance_ema10_pct": distance_ema10_pct,
        "distance_ema20_pct": distance_ema20_pct,
    })


def test_rolling_max_excludes_current_day():
    # Row 5 (0-indexed) itself is a huge spike (100) -- it must NEVER be
    # counted in its OWN max_atr_extension_prior_*d columns.
    ext = [1, 1, 1, 1, 1, 100, 1, 1, 1, 1]
    out = compute_prior_extension_history(_single_ticker_df(ext))
    row5 = out.iloc[5]
    assert row5["max_atr_extension_prior_20d"] == 1.0
    assert row5["max_atr_extension_prior_40d"] == 1.0
    assert row5["atr_extension_peak"] == 1.0
    # The NEXT row's window DOES see it (it is now strictly in the past).
    row6 = out.iloc[6]
    assert row6["max_atr_extension_prior_20d"] == 100.0
    assert row6["atr_extension_peak"] == 100.0
    assert row6["days_since_atr_extension_peak"] == 1.0


def test_peak_always_strictly_before_signal_day():
    rng = np.random.default_rng(5)
    ext = rng.normal(3, 4, size=90)
    out = compute_prior_extension_history(_single_ticker_df(ext))
    # days_since_atr_extension_peak counts trading days between the peak
    # row and the current row -- by construction (window is t-w..t-1) this
    # can never be 0 or negative wherever it is populated at all.
    valid = out["days_since_atr_extension_peak"].dropna()
    assert (valid >= 1).all()


def test_days_since_peak_and_drawdown_hand_computed():
    # Manually verified example: peak (9.0) sits at index 2; row 5's
    # trailing-60 window is ext[0:5] = [1,2,9,3,2] -> peak=9, days_since=3.
    ext = [1, 2, 9, 3, 2, 1, 0.5, 0.2, 4, 5]
    out = compute_prior_extension_history(_single_ticker_df(ext))
    row5 = out.iloc[5]
    assert row5["atr_extension_peak"] == 9.0
    assert row5["days_since_atr_extension_peak"] == 3.0
    assert row5["atr_extension_drawdown_from_peak"] == pytest.approx(1.0 - 9.0)

    row9 = out.iloc[9]
    assert row9["atr_extension_peak"] == 9.0
    assert row9["days_since_atr_extension_peak"] == 7.0
    assert row9["atr_extension_drawdown_from_peak"] == pytest.approx(5.0 - 9.0)


def test_min_distance_since_peak_and_touch_flags():
    ext = [1, 2, 9, 3, 2, 1, 0.5, 0.2, 4, 5]
    d10 = [0, 0, 0, 0, 0, -1, -2, -3, 1, 2]
    out = compute_prior_extension_history(_single_ticker_df(ext, distance_ema10_pct=d10))
    row5 = out.iloc[5]  # peak at idx2, since-peak window = d10[3:6] = [0,0,-1]
    assert row5["min_distance_ema10_pct_since_peak"] == -1.0
    assert row5["touched_ema10_since_peak"] == 1.0

    # A ticker that never dips to/through EMA10 since its peak.
    d10_no_touch = [0, 0, 0, 5, 5, 5, 5, 5, 5, 5]
    out2 = compute_prior_extension_history(_single_ticker_df(ext, distance_ema10_pct=d10_no_touch))
    row5b = out2.iloc[5]
    assert row5b["min_distance_ema10_pct_since_peak"] == 5.0
    assert row5b["touched_ema10_since_peak"] == 0.0


def test_partial_history_tracked_transparently():
    ext = [1.0] * 10
    out = compute_prior_extension_history(_single_ticker_df(ext))
    # n_valid_days_prior_peak_window grows with row index (capped at the
    # configured peak window, 60) -- never fabricated beyond what's real.
    counts = out["n_valid_days_prior_peak_window"].tolist()
    assert counts == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # First row has no history at all -- everything derived must be NaN,
    # not a fabricated/leaked value.
    row0 = out.iloc[0]
    assert pd.isna(row0["atr_extension_peak"])
    assert pd.isna(row0["days_since_atr_extension_peak"])
    assert pd.isna(row0["min_distance_ema10_pct_since_peak"])


def test_no_cross_ticker_contamination():
    # Ticker A has a huge peak right before ticker B's rows begin (in row
    # order) -- ticker B's OWN peak computation must never see it.
    df_a = _single_ticker_df([1, 1, 1, 100, 1], ticker="A")
    df_b = _single_ticker_df([2, 2, 2, 2, 2], ticker="B")
    combined = pd.concat([df_a, df_b], ignore_index=True)
    out = compute_prior_extension_history(combined)

    b_rows = out[out["ticker"] == "B"]
    assert (b_rows["atr_extension_peak"].dropna() <= 2.0).all()
    assert not (b_rows["atr_extension_peak"] == 100.0).any()


def test_missing_peak_window_in_windows_days_raises(monkeypatch):
    import yolo_calibration.features.repeat_offender as mod

    def bad_config():
        return {"prior_extension_history": {"windows_days": [20, 40], "peak_window_days": 60}}

    monkeypatch.setattr(mod, "load_features_config", bad_config)
    with pytest.raises(ValueError):
        compute_prior_extension_history(_single_ticker_df([1.0] * 5))
