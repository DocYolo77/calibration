import pandas as pd

from yolo_calibration.config import load_features_config
from yolo_calibration.features.technical import add_thrust, add_thrust_percentiles


def test_thrust_is_ema_short_minus_ema_long_of_daily_return():
    cfg = load_features_config()["thrust"]
    dates = pd.bdate_range("2023-01-02", periods=40)
    closes = [100 * (1.01 ** i) for i in range(40)]
    df = pd.DataFrame({"date": dates, "ticker": "AAA", "close": closes})

    out = add_thrust(df)

    daily_return_pct = pd.Series(closes).pct_change() * 100.0
    for name, hc in cfg["horizons"].items():
        short = daily_return_pct.ewm(span=hc["ema_short_days"], adjust=cfg["ema_adjust"]).mean()
        long = daily_return_pct.ewm(span=hc["ema_long_days"], adjust=cfg["ema_adjust"]).mean()
        expected = (short - long).to_numpy()
        got = out[name].to_numpy()
        assert abs(expected[-1] - got[-1]) < 1e-9, name


def test_thrust_percentiles_restricted_to_eligible():
    date = pd.Timestamp("2023-06-01")
    df = pd.DataFrame({
        "date": [date] * 3,
        "ticker": ["A", "B", "C"],
        "thrust_1d": [5.0, 1.0, 100.0],
        "thrust_1w": [5.0, 1.0, 100.0],
        "thrust_1m": [5.0, 1.0, 100.0],
        "eligible": [True, True, False],
    })
    out = add_thrust_percentiles(df)
    c_row = out[out["ticker"] == "C"].iloc[0]
    assert pd.isna(c_row["thrust_percentile_1d"])
    a_row = out[out["ticker"] == "A"].iloc[0]
    assert a_row["thrust_percentile_1d"] == 100.0  # best among eligible {A,B}, ignoring C's huge value
