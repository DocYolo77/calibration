import numpy as np
import pandas as pd

from yolo_calibration.features.technical import add_relative_strength_percentiles


def test_rs_percentile_only_ranks_within_eligible_universe():
    date = pd.Timestamp("2023-06-01")
    df = pd.DataFrame({
        "date": [date] * 4,
        "ticker": ["A", "B", "C", "D"],
        "return_1d": [10.0, 5.0, 1.0, -50.0],
        "return_5d": [10.0, 5.0, 1.0, -50.0],
        "return_21d": [10.0, 5.0, 1.0, -50.0],
        "return_3m": [10.0, 5.0, 1.0, -50.0],
        "return_6m": [10.0, 5.0, 1.0, -50.0],
        "return_12m": [10.0, 5.0, 1.0, -50.0],
        # D has the worst return but is NOT eligible -> must be excluded
        # from the ranking pool entirely (not just given a low percentile).
        "eligible": [True, True, True, False],
    })
    out = add_relative_strength_percentiles(df)

    # D (ineligible) gets NaN, not a computed percentile.
    d_row = out[out["ticker"] == "D"].iloc[0]
    assert pd.isna(d_row["rs_percentile_1d"])

    # Among {A, B, C} (the eligible set), A has the best 1d return -> should
    # be the top percentile. If D had been included in the ranking pool,
    # A's percentile would be lower (diluted by a 4th competitor).
    a_pct = out[out["ticker"] == "A"].iloc[0]["rs_percentile_1d"]
    assert a_pct == 100.0  # rank(pct=True) of the max value among 3 items


def test_eligible_flag_is_precondition_not_derived_from_rs():
    """The eligible flag must be determined independently of / before RS
    ranking (spec: "eligible Universe wird VOR RS-Ranking bestimmt").
    Changing a stock's return should never change whether ANOTHER stock is
    eligible."""
    date = pd.Timestamp("2023-06-01")
    base = pd.DataFrame({
        "date": [date] * 3,
        "ticker": ["A", "B", "C"],
        "return_1d": [10.0, 5.0, 1.0],
        "return_5d": [10.0, 5.0, 1.0],
        "return_21d": [10.0, 5.0, 1.0],
        "return_3m": [10.0, 5.0, 1.0],
        "return_6m": [10.0, 5.0, 1.0],
        "return_12m": [10.0, 5.0, 1.0],
        "eligible": [True, True, False],
    })
    mutated = base.copy()
    mutated.loc[mutated["ticker"] == "A", "return_1d"] = 999.0

    out1 = add_relative_strength_percentiles(base)
    out2 = add_relative_strength_percentiles(mutated)

    # C's eligibility (False) and resulting NaN percentile are unaffected by A's return.
    c1 = out1[out1["ticker"] == "C"].iloc[0]["rs_percentile_1d"]
    c2 = out2[out2["ticker"] == "C"].iloc[0]["rs_percentile_1d"]
    assert pd.isna(c1) and pd.isna(c2)


def test_rs_percentile_computed_per_date_independently():
    dates = [pd.Timestamp("2023-06-01")] * 2 + [pd.Timestamp("2023-06-02")] * 2
    df = pd.DataFrame({
        "date": dates,
        "ticker": ["A", "B", "A", "B"],
        "return_1d": [1.0, 2.0, 2.0, 1.0],
        "return_5d": [1.0, 2.0, 2.0, 1.0],
        "return_21d": [1.0, 2.0, 2.0, 1.0],
        "return_3m": [1.0, 2.0, 2.0, 1.0],
        "return_6m": [1.0, 2.0, 2.0, 1.0],
        "return_12m": [1.0, 2.0, 2.0, 1.0],
        "eligible": [True, True, True, True],
    })
    out = add_relative_strength_percentiles(df)
    day1 = out[out["date"] == dates[0]]
    day2 = out[out["date"] == dates[2]]
    # Leader flips between days -> percentiles must flip too, proving no
    # cross-date leakage into the same-day ranking.
    assert day1[day1["ticker"] == "B"].iloc[0]["rs_percentile_1d"] == 100.0
    assert day2[day2["ticker"] == "A"].iloc[0]["rs_percentile_1d"] == 100.0
