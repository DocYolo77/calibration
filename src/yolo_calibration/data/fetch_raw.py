"""Checkpointed historical fetch orchestration.

Each function iterates candidate trading days, skips any day that already
has a raw checkpoint file on disk (unless force=True), fetches it from
Massive, and writes the checkpoint immediately — so an interrupted run can
resume without re-downloading completed days (spec section 17).
"""

from __future__ import annotations

from datetime import date

from yolo_calibration.data.massive_client import MassiveClient
from yolo_calibration.data.storage import (
    raw_grouped_daily_exists,
    raw_grouped_daily_unadjusted_exists,
    raw_qqq_constituents_exists,
    raw_reference_tickers_exists,
    write_raw_grouped_daily,
    write_raw_grouped_daily_unadjusted,
    write_raw_qqq_constituents,
    write_raw_reference_tickers,
)
from yolo_calibration.utils.dates import iter_candidate_weekdays
from yolo_calibration.utils.logging import get_logger

logger = get_logger(__name__)


def fetch_grouped_daily_range(client: MassiveClient, start: date, end: date, *, force: bool = False) -> dict:
    """Fetch bulk grouped-daily OHLCV for every weekday in [start, end].
    A day with zero results (market holiday) is still checkpointed as an
    empty file so it is never re-attempted."""
    fetched, skipped, holidays = 0, 0, 0
    for d in iter_candidate_weekdays(start, end):
        if not force and raw_grouped_daily_exists(d):
            skipped += 1
            continue
        records = client.get_grouped_daily(d)
        write_raw_grouped_daily(d, records)
        if not records:
            holidays += 1
            logger.info("No grouped-daily data for %s (likely market holiday)", d)
        else:
            fetched += 1
            logger.info("Fetched grouped-daily for %s: %d tickers", d, len(records))
    return {"fetched": fetched, "skipped_existing": skipped, "empty_holiday_days": holidays}


def fetch_grouped_daily_unadjusted_range(client: MassiveClient, start: date, end: date, *,
                                          force: bool = False) -> dict:
    """Fetch bulk grouped-daily OHLCV with adjusted=false (raw, as-traded
    prices). Used ONLY for point-in-time-correct market cap — see
    config/market_cap_methodology.yaml. adjusted=true (fetch_grouped_daily_range)
    remains the basis for all technical/outcome features, which are
    scale-invariant to the (build-time, not point-in-time) split adjustment
    and benefit from split-continuity within rolling windows."""
    fetched, skipped, holidays = 0, 0, 0
    for d in iter_candidate_weekdays(start, end):
        if not force and raw_grouped_daily_unadjusted_exists(d):
            skipped += 1
            continue
        records = client.get_grouped_daily(d, adjusted=False)
        write_raw_grouped_daily_unadjusted(d, records)
        if not records:
            holidays += 1
        else:
            fetched += 1
            logger.info("Fetched unadjusted grouped-daily for %s: %d tickers", d, len(records))
    return {"fetched": fetched, "skipped_existing": skipped, "empty_holiday_days": holidays}


def fetch_qqq_constituents_range(client: MassiveClient, start: date, end: date, *, force: bool = False) -> dict:
    """Fetch point-in-time QQQ ETF constituents for every weekday in
    [start, end]. Raises MassiveAPIError (propagated) if the endpoint is
    unavailable — callers must not silently fall back to a static list
    (spec section 10)."""
    fetched, skipped, empty = 0, 0, 0
    for d in iter_candidate_weekdays(start, end):
        if not force and raw_qqq_constituents_exists(d):
            skipped += 1
            continue
        records = client.get_etf_constituents(composite_ticker="QQQ", effective_date=d)
        write_raw_qqq_constituents(d, records)
        if not records:
            empty += 1
            logger.warning("No QQQ constituent records for effective_date=%s", d)
        else:
            fetched += 1
            logger.info("Fetched QQQ constituents for %s: %d holdings", d, len(records))
    return {"fetched": fetched, "skipped_existing": skipped, "empty_days": empty}


def fetch_reference_tickers_range(client: MassiveClient, start: date, end: date, *,
                                   ticker_type: str = "CS", force: bool = False) -> dict:
    """Fetch the point-in-time (as of that trading day) common-stock ticker
    list for every weekday in [start, end]. This is the basis of the
    asset-type filter and prevents survivorship bias: a ticker delisted
    today still appears on historical dates before its delisting."""
    fetched, skipped = 0, 0
    for d in iter_candidate_weekdays(start, end):
        if not force and raw_reference_tickers_exists(d):
            skipped += 1
            continue
        records = list(client.get_reference_tickers(as_of_date=d, market="stocks", ticker_type=ticker_type))
        write_raw_reference_tickers(d, records)
        fetched += 1
        logger.info("Fetched reference tickers for %s: %d tickers", d, len(records))
    return {"fetched": fetched, "skipped_existing": skipped}
