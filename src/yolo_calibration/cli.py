"""CLI entrypoint: `python -m yolo_calibration <command> ...` (spec section 16).

Phase 1 intentionally has NO `optimize` / `choose-threshold` command.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime

import pandas as pd

from yolo_calibration.config import REPO_ROOT, make_build_metadata
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
from yolo_calibration.reports.ema_pullback_range_study import build_ema_pullback_range_study
from yolo_calibration.reports.opportunity_state_candidate_rules import build_candidate_rules_report
from yolo_calibration.reports.opportunity_state_study import build_opportunity_state_study
from yolo_calibration.reports.rs_atr_extension_study import build_rs_atr_extension_study
from yolo_calibration.reports.rs_benchmark import build_rs_benchmark_report, build_rs_benchmark_report_common_sample
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


def cmd_build_rs_benchmark_report(args: argparse.Namespace) -> int:
    """Purely descriptive research report (reports/rs_benchmark.py): every
    RS horizon (1D/1W/1M/3M/6M/12M) bucketed against the 5D/10D/20D stock
    outcomes, for ONE calendar year at a time. Hard-guarded to a single
    calendar year so this command cannot silently be pointed at a range
    spanning years the caller did not intend to analyze yet — the guard is
    year-agnostic by design (not literally hardcoded to 2024), but the
    workflow/CLI invocation controls which year is actually requested."""
    if args.start.year != args.end.year:
        logger.error("RS benchmark report must be built for a single calendar year; got %s..%s.",
                     args.start, args.end)
        return 1
    year = args.start.year

    features = read_processed("stock_features_daily", years=[year])
    outcomes = read_processed("stock_outcomes_daily", years=[year])
    if features.empty or outcomes.empty:
        logger.error("Missing prerequisite table(s) for year %d — run build-stock-features / "
                     "build-stock-outcomes first.", year)
        return 1

    # Defense in depth: restrict to the exact requested range even though
    # the year-partitioned read above already scopes to `year`.
    features = features[(features["date"] >= pd.Timestamp(args.start)) & (features["date"] <= pd.Timestamp(args.end))]
    outcomes = outcomes[(outcomes["date"] >= pd.Timestamp(args.start)) & (outcomes["date"] <= pd.Timestamp(args.end))]

    tables = build_rs_benchmark_report(features, outcomes)

    out_dir = REPO_ROOT / "reports" / f"rs_benchmark_{year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, table in tables.items():
        table.to_csv(out_dir / f"{key}.csv", index=False)
    logger.info("Wrote %d RS benchmark tables to %s", len(tables), out_dir)
    return 0


def cmd_build_rs_benchmark_common_sample_report(args: argparse.Namespace) -> int:
    """Robustness check ONLY (reports/rs_benchmark.py::build_rs_benchmark_report_common_sample)
    — writes to a SEPARATE directory (rs_benchmark_{year}_common_sample/)
    and never touches/overwrites the full-sample rs_benchmark_{year}/
    tables from cmd_build_rs_benchmark_report. Same single-calendar-year
    guard and year-partition-only read as the full-sample command."""
    if args.start.year != args.end.year:
        logger.error("RS benchmark common-sample report must be built for a single calendar year; got %s..%s.",
                     args.start, args.end)
        return 1
    year = args.start.year

    features = read_processed("stock_features_daily", years=[year])
    outcomes = read_processed("stock_outcomes_daily", years=[year])
    if features.empty or outcomes.empty:
        logger.error("Missing prerequisite table(s) for year %d — run build-stock-features / "
                     "build-stock-outcomes first.", year)
        return 1

    features = features[(features["date"] >= pd.Timestamp(args.start)) & (features["date"] <= pd.Timestamp(args.end))]
    outcomes = outcomes[(outcomes["date"] >= pd.Timestamp(args.start)) & (outcomes["date"] <= pd.Timestamp(args.end))]

    tables, summary = build_rs_benchmark_report_common_sample(features, outcomes)

    out_dir = REPO_ROOT / "reports" / f"rs_benchmark_{year}_common_sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, table in tables.items():
        table.to_csv(out_dir / f"{key}.csv", index=False)
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %d RS benchmark common-sample tables + summary to %s (n_common_sample=%d, "
                "retained_pct=%.2f%%)", len(tables), out_dir, summary["n_common_sample"], summary["retained_pct"])
    return 0


def cmd_build_rs_atr_extension_study(args: argparse.Namespace) -> int:
    """Descriptive Phase-2 RS x ATR-Extension cross-tabulation
    (reports/rs_atr_extension_study.py), for ONE calendar year at a time.
    Same single-calendar-year guard, same year-partition-only read, and the
    same point-in-time discipline as cmd_build_rs_benchmark_report — writes
    to a NEW, separate rs_atr_extension_{year}/ directory and never touches
    the existing rs_benchmark_{year}/ or rs_benchmark_{year}_common_sample/
    report directories."""
    if args.start.year != args.end.year:
        logger.error("RS x ATR Extension study must be built for a single calendar year; got %s..%s.",
                     args.start, args.end)
        return 1
    year = args.start.year

    features = read_processed("stock_features_daily", years=[year])
    outcomes = read_processed("stock_outcomes_daily", years=[year])
    if features.empty or outcomes.empty:
        logger.error("Missing prerequisite table(s) for year %d — run build-stock-features / "
                     "build-stock-outcomes first.", year)
        return 1

    # Defense in depth: restrict to the exact requested range even though
    # the year-partitioned read above already scopes to `year`.
    features = features[(features["date"] >= pd.Timestamp(args.start)) & (features["date"] <= pd.Timestamp(args.end))]
    outcomes = outcomes[(outcomes["date"] >= pd.Timestamp(args.start)) & (outcomes["date"] <= pd.Timestamp(args.end))]

    tables, summary = build_rs_atr_extension_study(features, outcomes)

    out_dir = REPO_ROOT / "reports" / f"rs_atr_extension_{year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, table in tables.items():
        table.to_csv(out_dir / f"{key}.csv", index=False)
        table.to_parquet(out_dir / f"{key}.parquet", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "METHODOLOGY.md").write_text(
        f"# RS x ATR Extension Study — {year}\n\n"
        "Purely descriptive Phase-2 cross-tabulation. NO threshold selection, "
        "NO \"best\" cell/RS-horizon marking, NO trading rule — see "
        "src/yolo_calibration/reports/rs_atr_extension_study.py module docstring "
        "for the full bucket/metric/baseline definitions.\n\n"
        f"- Signal period: {args.start} .. {args.end} (exclusively; pre-{year} data "
        "used only as point-in-time lookback for feature computation).\n"
        f"- n eligible {year} DATE x TICKER rows: {summary['n_eligible_2024_total']}\n"
        f"- RS horizon coverage: {json.dumps(summary['rs_horizon_coverage'])}\n"
        f"- Small-sample threshold (transparency only, no merging): {summary['small_sample_threshold']} "
        f"({summary['small_sample_cell_count']} cells below it)\n"
    )
    logger.info("Wrote RS x ATR Extension study (%d cross-matrix rows, %d small-sample cells) to %s",
                len(tables["cross_matrix"]), summary["small_sample_cell_count"], out_dir)
    return 0


def cmd_build_opportunity_state_study(args: argparse.Namespace) -> int:
    """Descriptive Phase-2 Opportunity-State study
    (reports/opportunity_state_study.py): RS x EMA10/EMA20-distance and
    previous-ATR-extension x current-EMA-distance ("repeat offender")
    cross-tabulations, for ONE target calendar year. Unlike the other
    Phase-2 report commands, this one REQUIRES `start` to be in an earlier
    calendar year than `end` — the previous-extension-history windows
    (config/features.yaml `prior_extension_history`, 60 trading days) need
    real trailing history the target year alone cannot supply, and
    `end.year` is treated as the single target/output year (mirrors the
    existing rs_benchmark_year workflow input: `start` may span extra
    years purely as point-in-time lookback, only `end.year`'s signal rows
    ever appear in the report). Writes to a NEW, separate
    opportunity_state_{year}/ directory and never touches the existing
    rs_benchmark_*/ or rs_atr_extension_*/ report directories."""
    if args.start.year >= args.end.year:
        logger.error("Opportunity-State study needs `start` in an earlier calendar year than `end` "
                     "(end.year is the single target year; start..end supplies the trailing lookback "
                     "history) — got %s..%s.", args.start, args.end)
        return 1
    year = args.end.year
    lookback_years = list(range(args.start.year, args.end.year + 1))

    features = read_processed("stock_features_daily", years=lookback_years)
    outcomes = read_processed("stock_outcomes_daily", years=[year])
    if features.empty or outcomes.empty:
        logger.error("Missing prerequisite table(s) for years %s — run build-stock-features / "
                     "build-stock-outcomes first.", lookback_years)
        return 1

    # Defense in depth: restrict to the exact requested ranges even though
    # the year-partitioned reads above already scope the tables.
    features = features[(features["date"] >= pd.Timestamp(args.start)) & (features["date"] <= pd.Timestamp(args.end))]
    year_start, year_end = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
    outcomes = outcomes[(outcomes["date"] >= year_start) & (outcomes["date"] <= year_end)]

    tables, summary = build_opportunity_state_study(features, outcomes, year)

    out_dir = REPO_ROOT / "reports" / f"opportunity_state_{year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, table in tables.items():
        table.to_csv(out_dir / f"{key}.csv", index=False)
        table.to_parquet(out_dir / f"{key}.parquet", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "METHODOLOGY.md").write_text(
        f"# Opportunity-State Study — {year}\n\n"
        "Purely descriptive Phase-2 cross-tabulation. NO threshold selection, "
        "NO Normal/Extended/Resetting classification, NO trading rule — see "
        "src/yolo_calibration/reports/opportunity_state_study.py module docstring "
        "for the full bucket/metric/baseline definitions.\n\n"
        f"- Target/signal year: {year} (exclusively); {args.start}..{year-1}-12-31 used only as "
        "point-in-time lookback for RS3M/6M and the previous-extension-history windows.\n"
        f"- n eligible {year} rows: {summary['n_eligible_total']}\n"
        f"- RS horizon coverage: {json.dumps(summary['rs_horizon_coverage'])}\n"
        f"- High-RS (>= {summary['high_rs_threshold']}) coverage: {json.dumps(summary['high_rs_coverage'])}\n"
        f"- Previous-extension-peak coverage: {json.dumps(summary['prior_extension_coverage'])}\n"
        f"- Small-sample threshold (transparency only, no merging): {summary['small_sample_threshold']} "
        f"({summary['small_sample_cell_count_total']} cells below it across all tables)\n"
    )
    logger.info("Wrote Opportunity-State study (%d tables, %d small-sample cells) to %s",
                len(tables), summary["small_sample_cell_count_total"], out_dir)
    return 0


def cmd_build_opportunity_state_candidate_rules_report(args: argparse.Namespace) -> int:
    """Applies the FROZEN Opportunity-State Candidate Rules v1
    (reports/opportunity_state_candidate_rules.py) to ONE target calendar
    year. Same lookback semantics as cmd_build_opportunity_state_study
    (`start` must be an earlier calendar year than `end`; `end`.year is the
    target year) -- this is what makes the SAME command, run once with
    end.year=2024 and once with end.year=2023, produce the 2024 report the
    rules were derived from and the frozen 2023 robustness check off the
    identical code path. Writes to its own
    opportunity_state_candidate_rules_{year}/ directory, never touching the
    other Phase-2 report directories."""
    if args.start.year >= args.end.year:
        logger.error("Opportunity-State Candidate Rules report needs `start` in an earlier calendar year "
                     "than `end` (end.year is the single target year) — got %s..%s.", args.start, args.end)
        return 1
    year = args.end.year
    lookback_years = list(range(args.start.year, args.end.year + 1))

    features = read_processed("stock_features_daily", years=lookback_years)
    outcomes = read_processed("stock_outcomes_daily", years=[year])
    if features.empty or outcomes.empty:
        logger.error("Missing prerequisite table(s) for years %s — run build-stock-features / "
                     "build-stock-outcomes first.", lookback_years)
        return 1

    features = features[(features["date"] >= pd.Timestamp(args.start)) & (features["date"] <= pd.Timestamp(args.end))]
    year_start, year_end = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
    outcomes = outcomes[(outcomes["date"] >= year_start) & (outcomes["date"] <= year_end)]

    tables, summary = build_candidate_rules_report(features, outcomes, year)

    out_dir = REPO_ROOT / "reports" / f"opportunity_state_candidate_rules_{year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, table in tables.items():
        table.to_csv(out_dir / f"{key}.csv", index=False)
        table.to_parquet(out_dir / f"{key}.parquet", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "METHODOLOGY.md").write_text(
        f"# Opportunity-State Candidate Rules v1 — {year}\n\n"
        "FROZEN candidate classification (Normal/Extended/Resetting), NOT a production rule. "
        "See src/yolo_calibration/reports/opportunity_state_candidate_rules.py module docstring "
        "for the exact thresholds and their justification from the 2024 descriptive studies.\n\n"
        f"- Target/signal year: {year} (exclusively); {args.start}..{year-1}-12-31 used only as "
        "point-in-time lookback.\n"
        f"- n eligible {year} rows: {summary['n_eligible_total']}\n"
        f"- Candidate rule thresholds: {json.dumps(summary['candidate_rule_thresholds'])}\n"
        f"- State population: {json.dumps(summary['state_population'])}\n"
        f"- Small-sample threshold (transparency only, no merging): {summary['small_sample_threshold']} "
        f"({summary['small_sample_cell_count_total']} cells below it)\n"
    )
    logger.info("Wrote Opportunity-State Candidate Rules report for %d (%d small-sample cells) to %s",
                year, summary["small_sample_cell_count_total"], out_dir)
    return 0


def cmd_build_ema_pullback_range_study(args: argparse.Namespace) -> int:
    """EMA Pullback Range Study (reports/ema_pullback_range_study.py):
    calibrates ONLY the future "EMA10 Pullback" / "EMA20 Pullback"
    dashboard labels for ONE target calendar year. Same lookback
    semantics as the other Phase-2 report commands (`start` must be an
    earlier calendar year than `end`; `end`.year is the target year).
    Writes to its own ema_pullback_range_{year}/ directory, never
    touching the other Phase-2 report directories. No candidate range is
    selected or frozen by this command -- only descriptive tables."""
    if args.start.year >= args.end.year:
        logger.error("EMA Pullback Range Study needs `start` in an earlier calendar year than `end` "
                     "(end.year is the single target year) — got %s..%s.", args.start, args.end)
        return 1
    year = args.end.year
    lookback_years = list(range(args.start.year, args.end.year + 1))

    features = read_processed("stock_features_daily", years=lookback_years)
    outcomes = read_processed("stock_outcomes_daily", years=[year])
    if features.empty or outcomes.empty:
        logger.error("Missing prerequisite table(s) for years %s — run build-stock-features / "
                     "build-stock-outcomes first.", lookback_years)
        return 1

    features = features[(features["date"] >= pd.Timestamp(args.start)) & (features["date"] <= pd.Timestamp(args.end))]
    year_start, year_end = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
    outcomes = outcomes[(outcomes["date"] >= year_start) & (outcomes["date"] <= year_end)]

    tables, summary = build_ema_pullback_range_study(features, outcomes, year)

    out_dir = REPO_ROOT / "reports" / f"ema_pullback_range_{year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, table in tables.items():
        table.to_csv(out_dir / f"{key}.csv", index=False)
        table.to_parquet(out_dir / f"{key}.parquet", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "METHODOLOGY.md").write_text(
        f"# EMA Pullback Range Study — {year}\n\n"
        "Calibrates ONLY the EMA10 Pullback / EMA20 Pullback dashboard labels. Purely descriptive — "
        "no candidate range is selected or frozen here. See "
        "src/yolo_calibration/reports/ema_pullback_range_study.py module docstring for the exact "
        "definitions (successful push, Resetting exclusion, Extended retention).\n\n"
        f"- Target/signal year: {year} (exclusively); {args.start}..{year-1}-12-31 used only as "
        "point-in-time lookback.\n"
        f"- n eligible {year} rows: {summary['n_eligible_total']}\n"
        f"- Strong-RS (>= {summary['high_rs_threshold']}) coverage: {json.dumps(summary['strong_rs_coverage'])}\n"
        f"- Resetting exclusion: {json.dumps(summary['resetting_exclusion'])}\n"
        f"- Small-sample threshold (transparency only, no merging): {summary['small_sample_threshold']} "
        f"({summary['small_sample_cell_count_total']} cells below it)\n"
    )
    logger.info("Wrote EMA Pullback Range Study for %d (%d small-sample cells) to %s",
                year, summary["small_sample_cell_count_total"], out_dir)
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

    p = sub.add_parser("build-rs-benchmark-report",
                        help="Descriptive RS-horizon-vs-outcome benchmark report for one calendar year "
                             "(no threshold/best-horizon selection).")
    _add_date_range_args(p)
    p.set_defaults(func=cmd_build_rs_benchmark_report)

    p = sub.add_parser("build-rs-benchmark-common-sample-report",
                        help="Robustness check: same RS benchmark, restricted to rows where all 6 RS "
                             "horizons are simultaneously populated. Writes to a separate directory, "
                             "never replaces the full-sample report.")
    _add_date_range_args(p)
    p.set_defaults(func=cmd_build_rs_benchmark_common_sample_report)

    p = sub.add_parser("build-rs-atr-extension-study",
                        help="Descriptive RS x ATR-Extension cross-tabulation for one calendar year "
                             "(no threshold/best-cell selection). Writes to a new, separate directory.")
    _add_date_range_args(p)
    p.set_defaults(func=cmd_build_rs_atr_extension_study)

    p = sub.add_parser("build-opportunity-state-study",
                        help="Descriptive RS x EMA10/EMA20-distance and previous-extension "
                             "('repeat offender') cross-tabulations for one target calendar year "
                             "(no state classification, no threshold selection). `end`.year is the "
                             "target year; `start` must be in an earlier year to supply lookback.")
    _add_date_range_args(p)
    p.set_defaults(func=cmd_build_opportunity_state_study)

    p = sub.add_parser("build-opportunity-state-candidate-rules-report",
                        help="Applies the FROZEN Candidate Rules v1 (Normal/Extended/Resetting) to one "
                             "target calendar year. `end`.year is the target year; `start` must be in an "
                             "earlier year to supply lookback. Same command/thresholds for the 2024 report "
                             "and the frozen 2023 robustness check.")
    _add_date_range_args(p)
    p.set_defaults(func=cmd_build_opportunity_state_candidate_rules_report)

    p = sub.add_parser("build-ema-pullback-range-study",
                        help="Calibrates ONLY the EMA10/EMA20 Pullback dashboard labels for one target "
                             "calendar year (no candidate range selected/frozen). `end`.year is the "
                             "target year; `start` must be in an earlier year to supply lookback.")
    _add_date_range_args(p)
    p.set_defaults(func=cmd_build_ema_pullback_range_study)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
