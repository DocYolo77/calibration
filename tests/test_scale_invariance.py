"""Regression tests for the split-adjustment finding (2026-08-12,
config/market_cap_methodology.yaml `why_unadjusted_close_specifically`):

Massive's adjusted=true grouped-daily series back-adjusts a historical
close using ALL splits known as of the BUILD date, not just splits known
as of that historical date. Empirically this showed up as absurd absolute
prices for serially-reverse-split penny stocks (e.g. MULN at ~$14B/share
on a date it really traded near $1-3).

Two things must hold:
  1. market_cap MUST use the unadjusted close — it is NOT scale-invariant
     (price x externally-sourced share count), so an over-adjusted price
     silently inflates it and can flip eligibility.
  2. Every ratio-based technical feature/outcome MUST be invariant to a
     constant per-ticker price scale factor — they are computed entirely
     from one ticker's own (uniformly scaled) series, so the erroneous
     scale cancels out algebraically, and does NOT need the unadjusted
     series at all.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

import yolo_calibration.data.cache as cache
import yolo_calibration.data.storage as storage
from yolo_calibration.data.massive_client import MassiveClient
from yolo_calibration.features.build_features import build_stock_features_daily
from yolo_calibration.outcomes.build_outcomes import build_stock_outcomes_daily
from yolo_calibration.universe.build_universe import build_market_universe_daily

from test_pipeline_integration import _write_synthetic_raw  # noqa: E402


def _write_scaled_pair_raw(start: date, n_days: int, *, scaled_multiplier: float):
    """Writes two tickers, BASE and SCALED, whose UNDERLYING (unadjusted)
    price path is byte-for-byte identical — only SCALED's ADJUSTED series
    is multiplied by `scaled_multiplier`, isolating exactly the scale
    discrepancy under test (unlike the generic `_write_synthetic_raw`
    helper, which draws independent noise per ticker)."""
    dates = pd.bdate_range(start=start, periods=n_days)
    rng = np.random.default_rng(11)
    closes = [100.0]
    for _ in range(1, n_days):
        closes.append(closes[-1] * (1 + rng.normal(0, 0.015)))

    for i, d in enumerate(dates):
        close = closes[i]
        unadj_records, adj_records = [], []
        for ticker, mult in [("BASE", 1.0), ("SCALED", scaled_multiplier)]:
            unadj_records.append({
                "T": ticker, "o": close * 0.999, "h": close * 1.06, "l": close * 0.94,
                "c": close, "v": 5_000_000, "vw": close, "n": 1000,
                "t": int(pd.Timestamp(d).timestamp() * 1000),
            })
            adj_close = close * mult
            adj_records.append({
                "T": ticker, "o": adj_close * 0.999, "h": adj_close * 1.06, "l": adj_close * 0.94,
                "c": adj_close, "v": 5_000_000, "vw": adj_close, "n": 1000,
                "t": int(pd.Timestamp(d).timestamp() * 1000),
            })
        storage.write_raw_grouped_daily(d.date(), adj_records)
        storage.write_raw_grouped_daily_unadjusted(d.date(), unadj_records)
        ref_records = [{"ticker": t, "type": "CS", "market": "stocks", "primary_exchange": "XNAS"}
                       for t in ["BASE", "SCALED"]]
        storage.write_raw_reference_tickers(d.date(), ref_records)
    return dates


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(storage, "RAW_DIR", raw_dir)
    monkeypatch.setattr(storage, "PROCESSED_DIR", processed_dir)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cache, "_cache_dir", lambda: cache_dir)
    cache._memo.clear()
    return raw_dir, processed_dir


@pytest.fixture
def stub_client(monkeypatch):
    monkeypatch.setattr(MassiveClient, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(
        MassiveClient, "get_ticker_overview",
        lambda self, ticker, as_of_date=None: {"weighted_shares_outstanding": 20_000_000_000},
    )
    return MassiveClient()


def test_market_cap_uses_unadjusted_close_not_inflated_adjusted_close(isolated_storage, stub_client):
    """A ticker whose ADJUSTED series is inflated 1000x (simulating an
    over-adjustment for future splits) must still get a market_cap based
    on its true (unadjusted) price, not the inflated one."""
    tickers = ["NORMAL", "OVERADJUSTED"]
    dates = _write_synthetic_raw(
        date(2023, 1, 2), 40, tickers,
        adjustment_scale={"OVERADJUSTED": 1000.0},  # NORMAL stays at scale 1.0 (no discrepancy)
    )
    start, end = dates[0].date(), dates[-1].date()

    universe = build_market_universe_daily(stub_client, start, end)
    last_date = universe["date"].max()
    row_normal = universe[(universe["ticker"] == "NORMAL") & (universe["date"] == last_date)].iloc[0]
    row_over = universe[(universe["ticker"] == "OVERADJUSTED") & (universe["date"] == last_date)].iloc[0]

    # Same underlying (unadjusted) price level and same shares outstanding
    # (stubbed identically for both) -> market caps must be roughly equal,
    # NOT 1000x apart, despite the adjusted `close` column differing 1000x.
    if pd.notna(row_normal["market_cap"]) and pd.notna(row_over["market_cap"]):
        ratio = row_over["market_cap"] / row_normal["market_cap"]
        assert 0.5 < ratio < 2.0, (
            f"market_cap ratio {ratio} suggests the inflated ADJUSTED close leaked into "
            f"market_cap instead of the unadjusted close being used"
        )


def test_technical_features_are_invariant_to_per_ticker_price_scale(isolated_storage, stub_client):
    """The exact same underlying dynamics, scaled 1000x for one ticker's
    ADJUSTED series only, must produce IDENTICAL ratio-based features —
    this is what makes the adjusted=true artifact harmless for everything
    except market_cap."""
    dates = _write_scaled_pair_raw(date(2023, 1, 2), 60, scaled_multiplier=1000.0)
    start, end = dates[0].date(), dates[-1].date()

    universe = build_market_universe_daily(stub_client, start, end)
    features = build_stock_features_daily(universe, start, end)
    outcomes = build_stock_outcomes_daily(features)

    base = features[features["ticker"] == "BASE"].sort_values("date").reset_index(drop=True)
    scaled = features[features["ticker"] == "SCALED"].sort_values("date").reset_index(drop=True)

    ratio_cols = ["return_1d", "return_5d", "return_21d", "atr_pct", "atr_extension",
                  "distance_ema10_pct", "distance_ema20_pct", "adr20"]
    for col in ratio_cols:
        pd.testing.assert_series_equal(
            base[col].reset_index(drop=True), scaled[col].reset_index(drop=True),
            check_names=False, atol=1e-6, rtol=1e-6,
        )

    base_out = outcomes[outcomes["ticker"] == "BASE"].sort_values("date").reset_index(drop=True)
    scaled_out = outcomes[outcomes["ticker"] == "SCALED"].sort_values("date").reset_index(drop=True)
    for col in ["forward_return_close_5d", "mfe_pct_5d", "mae_pct_5d", "mfe_atr_multiple_5d"]:
        pd.testing.assert_series_equal(
            base_out[col].reset_index(drop=True), scaled_out[col].reset_index(drop=True),
            check_names=False, atol=1e-6, rtol=1e-6,
        )

    # And the raw `close` column itself SHOULD differ ~1000x, confirming
    # the scale discrepancy was actually present (i.e. this test isn't
    # vacuously passing because nothing was scaled).
    close_ratio = (scaled["close"].to_numpy() / base["close"].to_numpy())
    assert np.allclose(close_ratio, 1000.0, rtol=0.01)
