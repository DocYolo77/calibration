import pandas as pd

from yolo_calibration.config import load_features_config
from yolo_calibration.features.technical import add_moving_averages


def test_ema10_ema20_use_adjust_false():
    cfg = load_features_config()["moving_averages"]
    assert cfg["ema_adjust"] is False

    dates = pd.bdate_range("2023-01-02", periods=30)
    closes = [100 + i * 0.5 for i in range(30)]
    df = pd.DataFrame({"date": dates, "ticker": "AAA", "close": closes,
                        "high": closes, "low": closes})
    out = add_moving_averages(df)

    expected_ema10 = pd.Series(closes).ewm(span=cfg["ema_fast_days"], adjust=False).mean()
    expected_ema20 = pd.Series(closes).ewm(span=cfg["ema_slow_days"], adjust=False).mean()
    pd.testing.assert_series_equal(
        out["ema10"].reset_index(drop=True), expected_ema10.rename("ema10"), check_exact=False
    )
    pd.testing.assert_series_equal(
        out["ema20"].reset_index(drop=True), expected_ema20.rename("ema20"), check_exact=False
    )


def test_distance_from_ema_formula():
    dates = pd.bdate_range("2023-01-02", periods=15)
    closes = [100.0] * 15
    df = pd.DataFrame({"date": dates, "ticker": "AAA", "close": closes, "high": closes, "low": closes})
    out = add_moving_averages(df)
    row = out.iloc[-1]
    assert abs(row["distance_ema10_pct"] - ((row["close"] - row["ema10"]) / row["ema10"] * 100.0)) < 1e-9
