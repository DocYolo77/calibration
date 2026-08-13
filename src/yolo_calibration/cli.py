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
    fetch_grouped_daily_unadjusted_range,
    fetch_qqq_constituents_range,
    fetch_reference_tickers_range,
)
from yolo_calibration.data.loaders import load_reference_tickers_range
from yolo_calibration.data.massive_client import MassiveClient
from yolo_calibration.data.raw_manifest import build_raw_manifest
from yolo_calibration.data.storage import (
    read_processed,
    write_manifest,
    write_processed_by_year,
)
from yolo_calibration.features.build_features import build_stock_features_daily
from yolo_calibration.outcomes.build_market_breadth import build_market_breadth_daily
from yolo_calibration.outcomes.build_outcomes import build_stock_outcomes_daily
from yolo_calibration.qqq_health.build_qqq_health import build_qqq_health_daily
from yolo_calibration.qqq_health.build_qqq_outcomes import build_qqq_health_outcomes_daily
from yolo_calibration.qqq_health.constituents import QQQConstituentsUnavailable, build_qqq_constituents_daily, component_counts
from yolo_calibration.reports.data_quality import build_data_quality_report, write_data_quality_report
from yolo_calibration.reports.descriptive import (
    generate_market_breadth_sanity_reports,
    generate_qqq_health_sanity_reports,
    generate_stock_sanity_reports,
)
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
    logger.info("Grouped daily fetch (adjusted): %s", result)
    result_unadj = fetch_grouped_daily_unadjusted_range(client, args.start, args.end, force=args.force)
    logger.info("Grouped daily fetch (unadjusted, for point-in-time market cap): %s", result_unadj)
    result2 = fetch_reference_tickers_range(client, args.start, args.end, force=args.force)
    logger.info("Reference tickers fetch: %s", result2)
    return 0


def cmd_fetch_qqq_constituents(args: argparse.Namespace) -> int:
    client = MassiveClient()
    result = fetch_qqq_constituents_range(client, args.start, args.end, force=args.force)
    logger.info("QQQ constituents fetch: %s", result)
    return 0


def cmd_build_reference_tickers(args: argparse.Namespace) -> int:
    """Materializes the `reference_tickers` processed table from the
    already-fetched raw per-day checkpoints (does not fetch — run
    fetch-raw first)."""
    df = load_reference_tickers_range(args.start, args.end)
    if df.empty:
        logger.error("No raw reference-ticker checkpoints found for the requested range — run fetch-raw first.")
        return 1
    write_processed_by_year("reference_tickers", df)
    write_manifest("reference_tickers", make_build_metadata(args.start, args.end, source="massive").to_dict())
    logger.info("Wrote reference_tickers: %d rows", len(df))
    return 0


def cmd_build_raw_manifest(args: argparse.Namespace) -> int:
    """Materializes the `raw_manifest` processed table: per (source, date),
    whether a raw checkpoint exists and its row count — reproducibility
    coverage metadata, not a content-correctness check."""
    df = build_raw_manifest(args.start, args.end)
    if df.empty:
        logger.error("No raw checkpoints of any kind found for the requested range — run fetch-raw first.")
        return 1
    write_processed_by_year("raw_manifest", df)
    write_manifest("raw_manifest", make_build_metadata(args.start, args.end, source="derived").to_dict())
    logger.info("Wrote raw_manifest: %d rows", len(df))
    return 0


def cmd_build_universe(args: argparse.Namespace) -> int:
    client = MassiveClient()
    if not args.no_fetch:
        fetch_grouped_daily_range(client, args.start, args.end)
        fetch_grouped_daily_unadjusted_range(client, args.start, args.end)
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
    # build_qqq_health_daily() only reads already-fetched raw checkpoints —
    # it never makes a live API call — so a MassiveClient (and therefore
    # MASSIVE_API_KEY) is only needed here when we're actually fetching.
    if not args.no_fetch:
        client = MassiveClient()
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
    if qqq_health.empty:
        logger.error("qqq_health_daily is empty — run build-qqq-health first. Note: this table is empty "
                     "whenever the QQQ health track is unavailable (see README 'Bekannte Limitierungen').")
        return 1
    df = build_qqq_health_outcomes_daily(qqq_health)
    write_processed_by_year("qqq_health_outcomes_daily", df)
    write_manifest("qqq_health_outcomes_daily",
                    make_build_metadata(args.start, args.end, source="derived").to_dict())
    logger.info("Wrote qqq_health_outcomes_daily: %d rows", len(df))
    return 0


def cmd_build_market_breadth(args: argparse.Namespace) -> int:
    features = read_processed("stock_features_daily")
    outcomes = read_processed("stock_outcomes_daily")
    if features.empty or outcomes.empty:
        logger.error("Missing prerequisite table(s) — run build-stock-features / build-stock-outcomes first.")
        return 1
    df = build_market_breadth_daily(features, outcomes)
    write_processed_by_year("market_breadth_daily", df)
    write_manifest("market_breadth_daily", make_build_metadata(args.start, args.end, source="derived").to_dict())
    logger.info("Wrote market_breadth_daily: %d rows", len(df))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    universe = read_processed("market_universe_daily")
    features = read_processed("stock_features_daily")
    outcomes = read_processed("stock_outcomes_daily")
    qqq_health = read_processed("qqq_health_daily")
    qqq_outcomes = read_processed("qqq_health_outcomes_daily")
    market_breadth = read_processed("market_breadth_daily")

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
        market_breadth_daily=market_breadth if not market_breadth.empty else None,
    )
    json_path, md_path = write_data_quality_report(report)
    logger.info("Wrote data quality report: %s / %s", json_path, md_path)

    if not features.empty and not outcomes.empty:
        generate_stock_sanity_reports(features, outcomes)
    if not qqq_health.empty and not qqq_outcomes.empty:
        generate_qqq_health_sanity_reports(qqq_health, qqq_outcomes)
    if not market_breadth.empty:
        generate_market_breadth_sanity_reports(market_breadth)

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

    p = sub.add_parser("build-reference-tickers", help="Materialize reference_tickers from raw checkpoints.")
    _add_date_range_args(p)
    p.set_defaults(func=cmd_build_reference_tickers)

    p = sub.add_parser("build-raw-manifest", help="Materialize raw_manifest (checkpoint coverage) table.")
    _add_date_range_args(p)
    p.set_defaults(func=cmd_build_raw_manifest)

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

    p = sub.add_parser("build-qqq-outcomes", help="Build qqq_health_outcomes_daily (QQQ index outcomes).")
    _add_date_range_args(p)
    p.set_defaults(func=cmd_build_qqq_outcomes)

    p = sub.add_parser("build-market-breadth", help="Build market_breadth_daily (always buildable, "
                                                      "independent of QQQ health availability).")
    _add_date_range_args(p)
    p.set_defaults(func=cmd_build_market_breadth)

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
