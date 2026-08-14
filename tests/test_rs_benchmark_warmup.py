"""Integration tests for the 2023-as-warmup-only strategy behind the 2024 RS
benchmark report:

  - Raw OHLCV is fetched/loaded for an EXTENDED range (2023-01-01..2024-12-31)
    so return_3m/6m/12m (and therefore rs_percentile_3m/6m/12m) have enough
    trailing history to be populated from early in 2024, instead of NaN for
    most/all of the year (see outcomes/build_market_breadth.py-style warmup
    reasoning, applied here to features/build_features.py's rolling shifts).
  - market_universe_daily (eligibility/adr20/market_cap) is ALSO built over
    the full [2023, 2024] range, not just 2024 -- ADR20 itself needs a
    20-trading-day rolling warmup, so building the universe for 2024-only
    would leave the first ~19 trading days of January ineligible purely from
    ADR20's own NaN, unrelated to the RS-horizon warmup this exercise is
    about. 2023 rows end up with REAL eligibility (not forced False) as a
    side effect -- harmless, since RS ranking is DATE-LOCAL (see
    build_features.py::add_relative_strength_percentiles: a 2023 date's
    eligible set never influences a 2024 date's percentile rank) and the
    report never reads the 2023 partition regardless (see below).
  - The RS benchmark report (reports/rs_benchmark.py via
    cli.py::cmd_build_rs_benchmark_report) reads ONLY the year=2024 parquet
    partition (`read_processed(table, years=[2024])`) — even though
    stock_features_daily/stock_outcomes_daily physically also carry a
    year=2023 partition (all rows there are eligible=False and never read),
    2023 can never appear in the report's output tables.
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd
import pytest

import yolo_calibration.cli as cli
import yolo_calibration.data.cache as cache
import yolo_calibration.data.storage as storage
from yolo_calibration.data.massive_client import MassiveClient
from yolo_calibration.features.build_features import build_stock_features_daily
from yolo_calibration.outcomes.build_outcomes import build_stock_outcomes_daily
from yolo_calibration.universe.build_universe import build_market_universe_daily

WARMUP_YEAR = 2023
REPORT_YEAR = 2024
FUTURE_YEAR = 2025  # must never be read by anything in this test


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
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    return raw_dir, processed_dir


@pytest.fixture
def stub_client(monkeypatch):
    monkeypatch.setattr(MassiveClient, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(
        MassiveClient, "get_ticker_overview",
        lambda self, ticker, as_of_date=None: {"weighted_shares_outstanding": 20_000_000_000},
    )
    return MassiveClient()


def _write_synthetic_raw(dates: pd.DatetimeIndex, tickers: list[str], *, seed: int = 3, price_poison_year: int | None = None):
    """Writes raw grouped-daily (adjusted + unadjusted) + reference-ticker
    checkpoints for every date in `dates`. If `price_poison_year` is given,
    rows for that year get an absurd price offset (10_000_000x) — used to
    prove those dates are never read/aggregated by the report."""
    rng = np.random.default_rng(seed)
    for i, d in enumerate(dates):
        records, unadj_records = [], []
        poison = price_poison_year is not None and d.year == price_poison_year
        for t in tickers:
            base = 100.0 + hash(t) % 50
            close = base * (1 + 0.0005 * i + rng.normal(0, 0.01))
            if poison:
                close *= 10_000_000.0
            row = {
                "T": t, "o": close * 0.999, "h": close * 1.03, "l": close * 0.97,
                "c": close, "v": 5_000_000, "vw": close, "n": 1000,
                "t": int(pd.Timestamp(d).timestamp() * 1000),
            }
            records.append(row)
            unadj_records.append(dict(row))
        storage.write_raw_grouped_daily(d.date(), records)
        storage.write_raw_grouped_daily_unadjusted(d.date(), unadj_records)
        ref_records = [{"ticker": t, "type": "CS", "market": "stocks", "primary_exchange": "XNAS"} for t in tickers]
        storage.write_raw_reference_tickers(d.date(), ref_records)


@pytest.fixture
def warmup_pipeline(isolated_storage, stub_client):
    """Builds the full pipeline the way the real 2023-warmup / 2024-report
    strategy does: raw fetched for [2023-01-02, 2024-Q1]; market_universe_daily
    AND stock_features_daily/stock_outcomes_daily are ALL built over the FULL
    [2023, 2024] range -- not just market_universe_daily's eligibility gate
    (asset_type/adr20/market_cap) but ALSO ADR20 itself needs a 20-trading-day
    warmup, discovered while writing this test: building market_universe_daily
    for 2024-only left the first ~19 trading days of Jan 2024 ineligible
    purely from ADR20's own rolling-window NaN, unrelated to RS. Extending
    market_universe_daily's build range the same way fixes that too. All
    three processed tables get written via write_processed_by_year (which
    auto-splits into year=2023 / year=2024 partitions) -- the REPORT layer
    (not this fixture) is what restricts consumption to year=2024 only."""
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    warmup_dates = pd.bdate_range(f"{WARMUP_YEAR}-01-02", periods=260)  # stays within 2023
    assert (warmup_dates.year == WARMUP_YEAR).all()
    report_dates = pd.bdate_range(f"{REPORT_YEAR}-01-02", periods=40)
    assert (report_dates.year == REPORT_YEAR).all()
    all_dates = warmup_dates.append(report_dates)

    _write_synthetic_raw(all_dates, tickers, price_poison_year=WARMUP_YEAR)

    report_start, report_end = report_dates[0].date(), report_dates[-1].date()
    full_start, full_end = warmup_dates[0].date(), report_dates[-1].date()

    universe_full_range = build_market_universe_daily(stub_client, full_start, full_end)
    assert not universe_full_range.empty
    assert set(pd.to_datetime(universe_full_range["date"]).dt.year.unique()) == {WARMUP_YEAR, REPORT_YEAR}

    features = build_stock_features_daily(universe_full_range, full_start, full_end)
    outcomes = build_stock_outcomes_daily(features)

    storage.write_processed_by_year("market_universe_daily", universe_full_range)
    storage.write_processed_by_year("stock_features_daily", features)
    storage.write_processed_by_year("stock_outcomes_daily", outcomes)

    return {
        "tickers": tickers, "warmup_dates": warmup_dates, "report_dates": report_dates,
        "features": features, "outcomes": outcomes, "universe": universe_full_range,
        "report_start": report_start, "report_end": report_end,
    }


def test_partitions_exist_for_both_warmup_and_report_year(warmup_pipeline):
    """Sanity check on the fixture itself: both year partitions must exist
    on disk (proving 2023 really is present as raw material), so that the
    later "report never reads 2023" assertions are meaningful rather than
    vacuously true."""
    features_dir = storage.PROCESSED_DIR / "stock_features_daily"
    assert (features_dir / f"year={WARMUP_YEAR}").exists()
    assert (features_dir / f"year={REPORT_YEAR}").exists()


def test_rs3m_warmup_across_year_boundary(warmup_pipeline):
    feats = warmup_pipeline["features"]
    first_report_day = pd.Timestamp(warmup_pipeline["report_dates"][0])
    row0 = feats[(feats["date"] == first_report_day) & (feats["ticker"] == "AAA")].iloc[0]
    # Without 2023 warmup, return_3m (63-day shift) on the very first 2024
    # row would be NaN (no 63 prior rows within 2024 alone).
    assert pd.notna(row0["return_3m"])
    assert pd.notna(row0["rs_percentile_3m"])


def test_rs6m_warmup_across_year_boundary(warmup_pipeline):
    feats = warmup_pipeline["features"]
    first_report_day = pd.Timestamp(warmup_pipeline["report_dates"][0])
    row0 = feats[(feats["date"] == first_report_day) & (feats["ticker"] == "AAA")].iloc[0]
    assert pd.notna(row0["return_6m"])
    assert pd.notna(row0["rs_percentile_6m"])


def test_rs12m_warmup_across_year_boundary(warmup_pipeline):
    feats = warmup_pipeline["features"]
    first_report_day = pd.Timestamp(warmup_pipeline["report_dates"][0])
    row0 = feats[(feats["date"] == first_report_day) & (feats["ticker"] == "AAA")].iloc[0]
    # 260 warmup days >= the 252-day shift -> even the FIRST 2024 row has a
    # complete RS12M lookback (unlike the real 2023 calendar, which has only
    # ~250 trading days and may fall a couple of days short right at the
    # 2024 open -- see the operational build's own coverage report for that
    # boundary case; this test proves the MECHANISM, not the exact
    # real-calendar day count).
    assert pd.notna(row0["return_12m"])
    assert pd.notna(row0["rs_percentile_12m"])

    # Coverage across the whole synthetic 2024 window should be complete.
    report_mask = feats["date"].dt.year == REPORT_YEAR
    assert feats.loc[report_mask, "rs_percentile_12m"].notna().all()


def test_2023_gets_real_eligibility_but_is_never_read_by_the_report(warmup_pipeline):
    """2023 rows go through the SAME universe/eligibility machinery as 2024
    (needed for ADR20's own warmup -- see fixture docstring), so some 2023
    rows are genuinely eligible with real RS values -- this is NOT the
    safeguard. The safeguard is that the report layer only ever reads the
    year=2024 partition (proven by the report-content tests below); this
    test just documents that 2023 is a real, non-degenerate period
    internally, so "the report never uses it" is a partition-read guarantee,
    not an accident of 2023 rows being trivially empty/ineligible."""
    feats = warmup_pipeline["features"]
    warmup_rows = feats[feats["date"].dt.year == WARMUP_YEAR]
    assert not warmup_rows.empty
    assert warmup_rows["eligible"].any()
    assert warmup_rows.loc[warmup_rows["eligible"], "rs_percentile_1d"].notna().any()


def test_report_contains_only_2024_signal_rows_despite_2023_partition_on_disk(warmup_pipeline):
    args = argparse.Namespace(
        start=warmup_pipeline["report_start"], end=warmup_pipeline["report_end"],
    )
    rc = cli.cmd_build_rs_benchmark_report(args)
    assert rc == 0

    out_dir = storage.PROCESSED_DIR.parent / "reports" / f"rs_benchmark_{REPORT_YEAR}"
    table = pd.read_csv(out_dir / "rs_percentile_1d__coarse__5d.csv")

    expected_eligible_2024_rows = int(
        (warmup_pipeline["features"]["date"].dt.year == REPORT_YEAR).sum()
    )  # every synthetic row is eligible (adr20/asset_type/market_cap all pass)
    assert table["n"].sum() == expected_eligible_2024_rows
    # ...and NOT the 2023+2024 combined total, which would be far larger.
    total_rows_both_years = len(warmup_pipeline["features"])
    assert table["n"].sum() < total_rows_both_years


def test_report_never_touches_a_future_year_partition_even_if_present(warmup_pipeline, isolated_storage):
    """A year=2025 partition existing on disk (e.g. a later, unrelated build
    in the same environment) must never leak into a 2024 report."""
    poison_dates = pd.bdate_range(f"{FUTURE_YEAR}-01-02", periods=5)
    poison_features = warmup_pipeline["features"][
        warmup_pipeline["features"]["date"].dt.year == REPORT_YEAR
    ].head(len(poison_dates) * len(warmup_pipeline["tickers"])).copy()
    poison_features["date"] = np.tile(poison_dates.values, len(warmup_pipeline["tickers"]))[: len(poison_features)]
    poison_features["rs_percentile_1d"] = 12345.0  # obviously-wrong sentinel value
    storage.write_processed_by_year("stock_features_daily", poison_features)
    assert (storage.PROCESSED_DIR / "stock_features_daily" / f"year={FUTURE_YEAR}").exists()

    args = argparse.Namespace(start=warmup_pipeline["report_start"], end=warmup_pipeline["report_end"])
    rc = cli.cmd_build_rs_benchmark_report(args)
    assert rc == 0

    out_dir = storage.PROCESSED_DIR.parent / "reports" / f"rs_benchmark_{REPORT_YEAR}"
    table = pd.read_csv(out_dir / "rs_percentile_1d__coarse__90-100.csv") \
        if (out_dir / "rs_percentile_1d__coarse__90-100.csv").exists() else None
    # The sentinel value (12345.0) would only be reachable via the 90-100
    # bucket (or would break bucketing entirely) -- simplest robust check:
    # total row count across ALL buckets must equal the known 2024-only
    # count, unchanged by the poisoned 2025 partition's presence.
    full_table = pd.read_csv(out_dir / "rs_percentile_1d__coarse__5d.csv")
    expected = int((warmup_pipeline["features"]["date"].dt.year == REPORT_YEAR).sum())
    assert full_table["n"].sum() == expected


def test_output_range_still_rejects_spanning_into_a_second_year(warmup_pipeline):
    args = argparse.Namespace(
        start=date(REPORT_YEAR, 6, 1), end=date(FUTURE_YEAR, 1, 5),
    )
    assert cli.cmd_build_rs_benchmark_report(args) == 1


def test_race_outcomes_present_and_aggregated_end_to_end(warmup_pipeline):
    args = argparse.Namespace(start=warmup_pipeline["report_start"], end=warmup_pipeline["report_end"])
    rc = cli.cmd_build_rs_benchmark_report(args)
    assert rc == 0

    out_dir = storage.PROCESSED_DIR.parent / "reports" / f"rs_benchmark_{REPORT_YEAR}"
    table = pd.read_csv(out_dir / "rs_percentile_1d__coarse__10d.csv")
    assert "reached_plus_5_before_minus_5_share" in table.columns
    assert "reached_plus_10_before_minus_5_share" in table.columns
    populated = table[table["n"] > 0]
    assert populated["reached_plus_5_before_minus_5_share"].between(0, 1).all()
    assert populated["reached_plus_10_before_minus_5_share"].between(0, 1).all()


def test_no_automatic_best_horizon_or_threshold_logic_end_to_end(warmup_pipeline):
    """Guardrail at the full-pipeline level: no output CSV, and no column
    within any of them, encodes a "best"/"optimal"/"selected" decision."""
    args = argparse.Namespace(start=warmup_pipeline["report_start"], end=warmup_pipeline["report_end"])
    assert cli.cmd_build_rs_benchmark_report(args) == 0

    out_dir = storage.PROCESSED_DIR.parent / "reports" / f"rs_benchmark_{REPORT_YEAR}"
    csvs = list(out_dir.glob("*.csv"))
    assert len(csvs) == 6 * 2 * 3
    forbidden_tokens = ("best", "optimal", "recommend", "selected", "winner", "leader")
    for path in csvs:
        assert not any(tok in path.stem.lower() for tok in forbidden_tokens), path.name
        cols = pd.read_csv(path, nrows=0).columns
        for col in cols:
            assert not any(tok in col.lower() for tok in forbidden_tokens), f"{path.name}::{col}"
