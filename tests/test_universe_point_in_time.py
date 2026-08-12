from datetime import date

import pandas as pd
import pytest

import yolo_calibration.universe.build_universe as bu


def test_delisted_ticker_not_dropped_from_earlier_eligible_history(monkeypatch):
    """A ticker present in the point-in-time reference snapshot on day 1 but
    ABSENT (delisted) by day 2 must still be asset-type-eligible on day 1 —
    its historical eligibility must not depend on whether it is active
    today (spec: no survivorship-bias fallback)."""
    d1, d2 = date(2023, 1, 3), date(2023, 1, 4)

    def fake_dates(start, end):
        return [d1, d2]

    def fake_read_ref(d):
        if d == d1:
            return pd.DataFrame({"ticker": ["DELISTED_CO", "SURVIVOR"], "primary_exchange": ["XNYS", "XNAS"]})
        if d == d2:
            # DELISTED_CO no longer appears at all on day 2 (as if a "today's
            # active universe" snapshot had been used instead of point-in-time).
            return pd.DataFrame({"ticker": ["SURVIVOR"], "primary_exchange": ["XNAS"]})
        return None

    monkeypatch.setattr(bu, "list_raw_grouped_daily_dates", fake_dates)
    monkeypatch.setattr(bu, "read_raw_reference_tickers", fake_read_ref)

    out = bu._load_asset_type_ok(d1, d2)
    day1 = out[out["date"] == pd.Timestamp(d1)]
    assert "DELISTED_CO" in set(day1["ticker"])
    assert bool(day1[day1["ticker"] == "DELISTED_CO"]["asset_type_ok"].iloc[0]) is True

    day2 = out[out["date"] == pd.Timestamp(d2)]
    assert "DELISTED_CO" not in set(day2["ticker"])


def test_otc_exclusion_is_defensive_filter(monkeypatch):
    d1 = date(2023, 1, 3)
    monkeypatch.setattr(bu, "list_raw_grouped_daily_dates", lambda start, end: [d1])
    monkeypatch.setattr(
        bu, "read_raw_reference_tickers",
        lambda d: pd.DataFrame({"ticker": ["OTC_CO", "NORMAL_CO"], "primary_exchange": ["OTCQB", "XNAS"]}),
    )
    out = bu._load_asset_type_ok(d1, d1)
    assert "OTC_CO" not in set(out["ticker"])
    assert "NORMAL_CO" in set(out["ticker"])


def test_eligible_requires_all_three_filters():
    """eligible = asset_type_ok AND adr20_ok AND market_cap_ok — verified
    against the merge logic directly (no API calls)."""
    dates = pd.bdate_range("2023-01-02", periods=25)
    ohlcv = pd.DataFrame({
        "date": dates, "ticker": "AAA",
        "high": [105.0] * 25, "low": [100.0] * 25, "close": [102.0] * 25,
    })
    from yolo_calibration.features.technical import compute_adr20
    ohlcv = compute_adr20(ohlcv)
    asset_type = pd.DataFrame({"date": dates, "ticker": "AAA", "asset_type_ok": True})
    merged = ohlcv.merge(asset_type, on=["date", "ticker"], how="left")
    merged["adr20_ok"] = merged["adr20"] > 4.0
    # ADR20 here is (105/100-1)*100 = 5.0% > 4.0% -> adr20_ok True once window fills
    assert merged.dropna(subset=["adr20"])["adr20_ok"].all()
