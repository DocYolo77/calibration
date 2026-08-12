"""Generic anti-leakage check: mutating data on/after a future date must
never change a feature value computed for an earlier date."""

import pandas as pd

from yolo_calibration.features.technical import (
    add_moving_averages,
    add_returns,
    add_thrust,
    add_true_range_atr,
)


def _build(df: pd.DataFrame) -> pd.DataFrame:
    out = add_returns(df)
    out = add_true_range_atr(out)
    out = add_moving_averages(out)
    out = add_thrust(out)
    return out


def test_mutating_future_rows_does_not_change_past_feature_values():
    dates = pd.bdate_range("2023-01-02", periods=60)
    base_close = [100 + 0.3 * i for i in range(60)]
    df = pd.DataFrame({
        "date": dates, "ticker": "AAA",
        "close": base_close,
        "high": [c * 1.02 for c in base_close],
        "low": [c * 0.98 for c in base_close],
    })

    mutated = df.copy()
    # Blow up the last 10 rows only.
    mutated.loc[50:, "close"] *= 5
    mutated.loc[50:, "high"] *= 5
    mutated.loc[50:, "low"] *= 5

    out_base = _build(df)
    out_mut = _build(mutated)

    check_cols = ["return_1d", "return_5d", "return_21d", "atr14", "atr_pct",
                  "ema10", "ema20", "sma50", "thrust_1d", "thrust_1w", "thrust_1m"]
    for col in check_cols:
        pd.testing.assert_series_equal(
            out_base[col].iloc[:49].reset_index(drop=True),
            out_mut[col].iloc[:49].reset_index(drop=True),
            check_names=False,
        )


def test_outcome_columns_are_the_only_place_future_data_is_used():
    """Sanity: the feature-building pipeline (technical.py) never looks at
    row i+1 or later — verified by checking row i's outputs only depend on
    rows [0..i] via a truncation-equivalence check."""
    dates = pd.bdate_range("2023-01-02", periods=40)
    close = [100 + i for i in range(40)]
    df = pd.DataFrame({
        "date": dates, "ticker": "AAA", "close": close,
        "high": [c * 1.01 for c in close], "low": [c * 0.99 for c in close],
    })
    full = _build(df)
    truncated = _build(df.iloc[:25].copy())

    check_cols = ["atr14", "ema10", "ema20", "thrust_1d"]
    for col in check_cols:
        pd.testing.assert_series_equal(
            full[col].iloc[:25].reset_index(drop=True),
            truncated[col].iloc[:25].reset_index(drop=True),
            check_names=False,
        )
