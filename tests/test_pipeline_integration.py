"""End-to-end wiring check: raw checkpoints -> universe -> features ->
outcomes, using synthetic data and a stubbed Massive client (no network).
Catches column-mismatch / wiring bugs that isolated unit tests can miss.
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


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(storage, "RAW_DIR", raw_dir)
    monkeypatch.setattr(storage, "PROCESSED_DIR", processed_dir)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cache, "_cache_dir", lambda: cache_dir)
    cache._memo.clear()  # module-level memo must not leak between tests
    return raw_dir, processed_dir


@pytest.fixture
def stub_client(monkeypatch):
    monkeypatch.setattr(MassiveClient, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(
        MassiveClient, "get_ticker_overview",
        lambda self, ticker, as_of_date=None: {"weighted_shares_outstanding": 20_000_000_000},
    )
    return MassiveClient()


def _write_synthetic_raw(start: date, n_days: int, tickers: list[str]):
    dates = pd.bdate_range(start=start, periods=n_days)
    rng = np.random.default_rng(3)
    for i, d in enumerate(dates):
        records = []
        for t in tickers:
            base = 100.0 + hash(t) % 50
            close = base * (1 + 0.01 * i + rng.normal(0, 0.005))
            records.append({
                "T": t, "o": close * 0.999, "h": close * 1.06, "l": close * 0.94,
                "c": close, "v": 5_000_000, "vw": close, "n": 1000,
                "t": int(pd.Timestamp(d).timestamp() * 1000),
            })
        storage.write_raw_grouped_daily(d.date(), records)
        ref_records = [{"ticker": t, "type": "CS", "market": "stocks", "primary_exchange": "XNAS"}
                       for t in tickers]
        storage.write_raw_reference_tickers(d.date(), ref_records)
    return dates


def test_full_pipeline_wiring(isolated_storage, stub_client):
    tickers = ["AAA", "BBB", "CCC"]
    dates = _write_synthetic_raw(date(2023, 1, 2), 60, tickers)
    start, end = dates[0].date(), dates[-1].date()

    universe = build_market_universe_daily(stub_client, start, end)
    assert not universe.empty
    assert set(universe["ticker"].unique()) == set(tickers)

    features = build_stock_features_daily(universe, start, end)
    assert not features.empty
    for col in ["rs_percentile_1d", "thrust_1d", "atr14", "atr_extension"]:
        assert col in features.columns

    outcomes = build_stock_outcomes_daily(features)
    assert not outcomes.empty
    assert "mfe_pct_5d" in outcomes.columns
    assert "forward_return_close_20d" in outcomes.columns

    # RS percentiles should only be populated for eligible rows.
    non_eligible = features[~features["eligible"]]
    if not non_eligible.empty:
        assert non_eligible["rs_percentile_1m"].isna().all()


def test_market_cap_enrichment_batches_by_ticker_month_not_ticker_day(isolated_storage, monkeypatch):
    """Regression test: a multi-year backfill has millions of candidate
    ticker-days but only a few tens of thousands of unique (ticker, month)
    combinations. Market-cap enrichment must look up shares-outstanding
    once per unique ticker-month, not once per ticker-day, or a full
    backfill's API-call count (and wall-clock time) blows up by orders of
    magnitude. This test fails if that batching regresses."""
    call_log = []

    def counting_get_ticker_overview(self, ticker, as_of_date=None):
        call_log.append((ticker, as_of_date))
        return {"weighted_shares_outstanding": 20_000_000_000}

    monkeypatch.setattr(MassiveClient, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(MassiveClient, "get_ticker_overview", counting_get_ticker_overview)
    client = MassiveClient()

    tickers = ["AAA", "BBB", "CCC"]
    # 60 business days spans parts of 3 calendar months for 3 tickers ->
    # at most 9 unique (ticker, month) pairs, vs. up to 180 ticker-days.
    dates = _write_synthetic_raw(date(2023, 1, 2), 60, tickers)
    start, end = dates[0].date(), dates[-1].date()

    universe = build_market_universe_daily(client, start, end)
    assert not universe.empty

    unique_ticker_months = len({(t, d.year, d.month) for t in tickers for d in dates})
    assert len(call_log) <= unique_ticker_months
    assert len(call_log) < len(universe)  # must be far fewer calls than ticker-day rows
