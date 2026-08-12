"""End-to-end wiring check: raw checkpoints -> universe -> features ->
outcomes, using synthetic data and a stubbed Massive client (no network).
Catches column-mismatch / wiring bugs that isolated unit tests can miss.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

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
