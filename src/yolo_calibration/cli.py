"""CLI entrypoint: `python -m yolo_calibration <command> ...` (spec section 16).

Phase 1 intentionally has NO `optimize` / `choose-threshold` command.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

from yolo_calibration.config import make_build_metadata
from yolo_calibration.data.fetch_raw import (
    fetch_grouped_daily_range,
    fetch_qqq_constituents_range,
    fetch_reference_tickers_range,
)
from yolo_calibration.data.massive_client import MassiveClient
from yolo_calibration.data.storage import (
    read_processed,
    write_manifest,
    write_processed_by_year,
)
from yolo_calibration.features.build_features import build_stock_features_daily
from yolo_calibration.outcomes.build_outcomes import build_stock_outcomes_daily
from yolo_calibration.qqq_health.build_qqq_health import build_qqq_health_daily
from yolo_calibration.qqq_health.build_qqq_outcomes import build_qqq_health_outcomes_daily
from yolo_calibration.qqq_health.constituents import QQQConstituentsUnavailable, build_qqq_constituents_daily, component_counts
from yolo_calibration.reports.data_quality import build_data_quality_report, write_data_quality_report
from yolo_calibration.reports.descriptive import generate_qqq_health_sanity_reports, generate_stock_sanity_reports
from yolo_calibration.universe.build_universe import build_market_universe_daily
from yolo_calibration.utils.logging import get_logger

logger = get_logger(__name__)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _add_date_range_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start", required=True, type=_parse_date, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, type=_parse_date, help="YYYY-MM-DD")


def cmd_fetch_raw(args: argparse.Namespace) -> int:
    client = MassiveClient()
    result = fetch_grouped_daily_range(client, args.start, args.end, force=args.force)
    logger.info("Grouped daily fetch: %s", result)
    result2 = fetch_reference_tickers_range(client, args.start, args.end, force=args.force)
    logger.info("Reference tickers fetch: %s", result2)
    return 0


def cmd_fetch_qqq_constituents(args: argparse.Namespace) -> int:
    client = MassiveClient()
    result = fetch_qqq_constituents_range(client, args.start, args.end, force=args.force)
    logger.info("QQQ constituents fetch: %s", result)
    return 0


def cmd_build_universe(args: argparse.Namespace) -> int:
    client = MassiveClient()
    if not args.no_fetch:
        fetch_grouped_daily_range(client, args.start, args.end)
        fetch_reference_tickers_range(client, args.start, args.end)
    df = build_market_universe_daily(client, args.start, args.end)
    write_processed_by_year("market_universe_daily", df)
    write_manifest("market_universe_daily", make_build_metadata(args.start, args.end, source="massive").to_dict())
    logger.info("Wrote market_universe_daily: %d rows", len(df))
    return 0


def cmd_build_stock_features(args: argparse.Namespace) -> int:
    universe = read_processed("market_universe_daily")
    if universe.empty:
        logger.error("market_universe_daily is empty — run build-universe first.")
        return 1
    df = build_stock_features_daily(universe, args.start, args.end)
    write_processed_by_year("stock_features_daily", df)
    write_manifest("stock_features_daily", make_build_metadata(args.start, args.end, source="massive").to_dict())
    logger.info("Wrote stock_features_daily: %d rows", len(df))
    return 0


def cmd_build_stock_outcomes(args: argparse.Namespace) -> int:
    features = read_processed("stock_features_daily")
    if features.empty:
        logger.error("stock_features_daily is empty — run build-stock-features first.")
        return 1
    df = build_stock_outcomes_daily(features)
    write_processed_by_year("stock_outcomes_daily", df)
    write_manifest("stock_outcomes_daily", make_build_metadata(args.start, args.end, source="derived").to_dict())
    logger.info("Wrote stock_outcomes_daily: %d rows", len(df))
    return 0


def cmd_build_qqq_health(args: argparse.Namespace) -> int:
    client = MassiveClient()
    if not args.no_fetch:
        fetch_grouped_daily_range(client, args.start, args.end)
        fetch_qqq_constituents_range(client, args.start, args.end)
    try:
        df = build_qqq_health_daily(args.start, args.end)
    except QQQConstituentsUnavailable as exc:
        logger.error("QQQ health track UNAVAILABLE: %s", exc)
        write_manifest("qqq_health_daily", {"status": "unavailable", "error": str(exc)})
        return 2
    write_processed_by_year("qqq_health_daily", df)
    write_manifest("qqq_health_daily", make_build_metadata(args.start, args.end, source="massive").to_dict())
    logger.info("Wrote qqq_health_daily: %d rows", len(df))
    return 0


def cmd_build_qqq_outcomes(args: argparse.Namespace) -> int:
    qqq_health = read_processed("qqq_health_daily")
    features = read_processed("stock_features_daily")
    outcomes = read_processed("stock_outcomes_daily")
    if qqq_health.empty or features.empty or outcomes.empty:
        logger.error("Missing prerequisite table(s) — run build-qqq-health / build-stock-features / "
                     "build-stock-outcomes first.")
        return 1
    df = build_qqq_health_outcomes_daily(qqq_health, features, outcomes)
    write_processed_by_year("qqq_health_outcomes_daily", df)
    write_manifest("qqq_health_outcomes_daily",
                    make_build_metadata(args.start, args.end, source="derived").to_dict())
    logger.info("Wrote qqq_health_outcomes_daily: %d rows", len(df))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    universe = read_processed("market_universe_daily")
    features = read_processed("stock_features_daily")
    outcomes = read_processed("stock_outcomes_daily")
    qqq_health = read_processed("qqq_health_daily")
    qqq_outcomes = read_processed("qqq_health_outcomes_daily")

    qqq_health_error = None
    component_counts_df = None
    if qqq_health.empty:
        manifest = None
        try:
            from yolo_calibration.data.storage import read_manifest
            manifest = read_manifest("qqq_health_daily")
        except Exception:
            pass
        qqq_health_error = (manifest or {}).get("error", "qqq_health_daily table is empty")
    else:
        try:
            constituents = build_qqq_constituents_daily(args.start, args.end)
            component_counts_df = component_counts(constituents)
        except QQQConstituentsUnavailable as exc:
            qqq_health_error = str(exc)

    report = build_data_quality_report(
        market_universe_daily=universe,
        stock_features_daily=features,
        stock_outcomes_daily=outcomes,
        qqq_health_daily=qqq_health if not qqq_health.empty else None,
        qqq_health_outcomes_daily=qqq_outcomes if not qqq_outcomes.empty else None,
        qqq_constituent_component_counts=component_counts_df,
        qqq_health_error=qqq_health_error,
    )
    json_path, md_path = write_data_quality_report(report)
    logger.info("Wrote data quality report: %s / %s", json_path, md_path)

    if not features.empty and not outcomes.empty:
        generate_stock_sanity_reports(features, outcomes)
    if not qqq_health.empty and not qqq_outcomes.empty:
        generate_qqq_health_sanity_reports(qqq_health, qqq_outcomes)

    if report["warnings"]:
        for w in report["warnings"]:
            logger.warning(w)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m yolo_calibration")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-raw", help="Fetch bulk grouped-daily OHLCV + point-in-time reference tickers.")
    _add_date_range_args(p)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch_raw)

    p = sub.add_parser("fetch-qqq-constituents", help="Fetch point-in-time QQQ ETF constituents.")
    _add_date_range_args(p)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch_qqq_constituents)

    p = sub.add_parser("build-universe", help="Build market_universe_daily.")
    _add_date_range_args(p)
    p.add_argument("--no-fetch", action="store_true", help="Skip fetching raw data; use existing checkpoints only.")
    p.set_defaults(func=cmd_build_universe)

    p = sub.add_parser("build-stock-features", help="Build stock_features_daily.")
    _add_date_range_args(p)
    p.set_defaults(func=cmd_build_stock_features)

    p = sub.add_parser("build-stock-outcomes", help="Build stock_outcomes_daily.")
    _add_date_range_args(p)
    p.set_defaults(func=cmd_build_stock_outcomes)

    p = sub.add_parser("build-qqq-health", help="Build qqq_health_daily.")
    _add_date_range_args(p)
    p.add_argument("--no-fetch", action="store_true")
    p.set_defaults(func=cmd_build_qqq_health)

    p = sub.add_parser("build-qqq-outcomes", help="Build qqq_health_outcomes_daily.")
    _add_date_range_args(p)
    p.set_defaults(func=cmd_build_qqq_outcomes)

    p = sub.add_parser("verify", help="Data quality + Phase 1 descriptive sanity-check reports.")
    _add_date_range_args(p)
    p.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
