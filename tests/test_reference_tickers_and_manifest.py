"""Materialization of the `reference_tickers` and `raw_manifest` processed
tables from raw per-day checkpoints (reproducibility, spec sections 14/17)."""

from datetime import date

import pandas as pd
import pytest

import yolo_calibration.data.storage as storage
from yolo_calibration.data.loaders import load_reference_tickers_range
from yolo_calibration.data.raw_manifest import build_raw_manifest


@pytest.fixture
def isolated_raw(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(storage, "RAW_DIR", raw_dir)
    return raw_dir


def test_load_reference_tickers_range_concatenates_with_pit_date(isolated_raw):
    d1, d2 = date(2023, 1, 3), date(2023, 1, 4)
    storage.write_raw_reference_tickers(d1, [
        {"ticker": "AAA", "type": "CS", "market": "stocks", "primary_exchange": "XNAS", "active": True},
        {"ticker": "BBB", "type": "CS", "market": "stocks", "primary_exchange": "XNYS", "active": True},
    ])
    storage.write_raw_reference_tickers(d2, [
        # BBB delisted by d2 (no longer appears) -- must NOT retroactively
        # remove it from d1's row (point-in-time, no survivorship bias).
        {"ticker": "AAA", "type": "CS", "market": "stocks", "primary_exchange": "XNAS", "active": True},
    ])

    out = load_reference_tickers_range(d1, d2)
    assert len(out) == 3
    assert set(out.columns) >= {"date", "ticker", "type", "market", "primary_exchange", "active"}

    day1 = out[out["date"] == pd.Timestamp(d1)]
    assert set(day1["ticker"]) == {"AAA", "BBB"}
    day2 = out[out["date"] == pd.Timestamp(d2)]
    assert set(day2["ticker"]) == {"AAA"}


def test_load_reference_tickers_range_empty_when_no_checkpoints(isolated_raw):
    out = load_reference_tickers_range(date(2023, 1, 3), date(2023, 1, 4))
    assert out.empty


def test_build_raw_manifest_covers_all_sources_with_row_counts(isolated_raw):
    d1 = date(2023, 1, 3)
    storage.write_raw_grouped_daily(d1, [{"T": "AAA", "c": 10.0}, {"T": "BBB", "c": 20.0}])
    storage.write_raw_grouped_daily_unadjusted(d1, [{"T": "AAA", "c": 10.0}])
    storage.write_raw_qqq_constituents(d1, [{"ticker": "AAA", "weight": 0.05}])
    storage.write_raw_reference_tickers(d1, [{"ticker": "AAA", "type": "CS"}])

    manifest = build_raw_manifest(d1, d1)
    assert set(manifest["source"]) == {
        "grouped_daily", "grouped_daily_unadjusted", "qqq_constituents", "reference_tickers",
    }
    by_source = manifest.set_index("source")["row_count"]
    assert by_source["grouped_daily"] == 2
    assert by_source["grouped_daily_unadjusted"] == 1
    assert by_source["qqq_constituents"] == 1
    assert by_source["reference_tickers"] == 1


def test_build_raw_manifest_distinguishes_missing_from_empty_checkpoint(isolated_raw):
    d1, d2 = date(2023, 1, 3), date(2023, 1, 4)
    # d1: a legitimately empty checkpoint (e.g. market holiday) -- present, 0 rows.
    storage.write_raw_grouped_daily(d1, [])
    # d2: no checkpoint written at all -- must not appear as a row.
    manifest = build_raw_manifest(d1, d2)
    gd = manifest[manifest["source"] == "grouped_daily"]
    assert len(gd) == 1
    assert gd.iloc[0]["date"] == pd.Timestamp(d1)
    assert gd.iloc[0]["row_count"] == 0


def test_build_raw_manifest_empty_when_nothing_fetched(isolated_raw):
    manifest = build_raw_manifest(date(2023, 1, 3), date(2023, 1, 4))
    assert manifest.empty
