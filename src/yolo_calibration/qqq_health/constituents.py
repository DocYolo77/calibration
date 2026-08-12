"""Point-in-time QQQ constituent membership (spec section 10).

Source priority (config/qqq_health.yaml `constituents.source_priority`):
  A) Massive ETF-constituents endpoint (composite_ticker=QQQ, effective_date=D)
  B) a verified historical NDX-100 constituent source — NOT implemented in
     Phase 1; no such vendor source was identified/verified, so this path
     raises NotImplementedError rather than silently guessing at one.
  C) hard stop — this module NEVER falls back to today's static holdings
     for a historical date. If (A) has no data for a date, that date's
     QQQ-health track is marked unavailable, not silently backfilled.

This is deliberate: a silent fallback to current constituents would
introduce survivorship bias into the health breadth calculations, which
the project spec explicitly forbids ("Kein stiller Survivorship-Bias-
Fallback").
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from yolo_calibration.config import load_qqq_health_config
from yolo_calibration.data.storage import list_raw_grouped_daily_dates, read_raw_qqq_constituents
from yolo_calibration.utils.logging import get_logger

logger = get_logger(__name__)


class QQQConstituentsUnavailable(RuntimeError):
    """Raised when point-in-time QQQ constituents cannot be established for
    a requested date via any approved source. Callers must treat this as a
    hard stop for the QQQ-health track on that date, per spec section 10."""


def build_qqq_constituents_daily(start: date, end: date) -> pd.DataFrame:
    """Assemble per-day point-in-time QQQ holdings from the raw checkpoint
    files written by data.fetch_raw.fetch_qqq_constituents_range.

    Returns columns: date, constituent_ticker, weight. Also validates the
    minimum-component-count floor from config and raises
    QQQConstituentsUnavailable for any day below it (source data too thin
    to be a reliable breadth universe) rather than silently proceeding.
    """
    cfg = load_qqq_health_config()["constituents"]
    min_components = cfg["min_required_components_for_valid_day"]

    frames = []
    missing_dates = []
    thin_dates = []
    for d in list_raw_grouped_daily_dates(start, end):
        snap = read_raw_qqq_constituents(d)
        if snap is None:
            missing_dates.append(d)
            continue
        if snap.empty or len(snap) < min_components:
            thin_dates.append(d)
            continue
        df = snap.copy()
        df["date"] = pd.Timestamp(d)
        frames.append(df[["date", "constituent_ticker", "weight"]])

    if missing_dates:
        logger.error(
            "QQQ constituents raw checkpoint missing for %d trading day(s) in [%s, %s]. "
            "Run fetch-qqq-constituents first. First few missing: %s",
            len(missing_dates), start, end, missing_dates[:5],
        )
        raise QQQConstituentsUnavailable(
            f"{len(missing_dates)} trading day(s) have no point-in-time QQQ constituent data "
            f"and no verified fallback source is configured (spec section 10, option C: hard stop). "
            f"Example missing dates: {missing_dates[:5]}"
        )
    if thin_dates:
        logger.warning(
            "QQQ constituents below min_required_components_for_valid_day=%d on %d day(s): %s",
            min_components, len(thin_dates), thin_dates[:5],
        )

    if not frames:
        raise QQQConstituentsUnavailable(
            f"No valid QQQ constituent snapshots found in [{start}, {end}]."
        )

    return pd.concat(frames, ignore_index=True).sort_values(["date", "constituent_ticker"]).reset_index(drop=True)


def component_counts(constituents_daily: pd.DataFrame) -> pd.DataFrame:
    """date -> number of constituents used that day (spec section 10:
    "Die tatsächlich pro Tag verwendete Komponentenanzahl speichern.")."""
    return (
        constituents_daily.groupby("date")["constituent_ticker"]
        .nunique()
        .rename("component_count")
        .reset_index()
    )
