"""Data quality report generator (spec section 19). Produces both a
machine-readable JSON and a small human-readable Markdown summary after
every historical build. Contains NO threshold optimization — descriptive
counts and warnings only.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from yolo_calibration.config import REPO_ROOT


def build_data_quality_report(
    *,
    market_universe_daily: pd.DataFrame,
    stock_features_daily: pd.DataFrame,
    stock_outcomes_daily: pd.DataFrame,
    qqq_health_daily: pd.DataFrame | None,
    qqq_health_outcomes_daily: pd.DataFrame | None,
    qqq_constituent_component_counts: pd.DataFrame | None,
    qqq_health_error: str | None = None,
) -> dict:
    report: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "warnings": [],
    }

    if not market_universe_daily.empty:
        mu = market_universe_daily.copy()
        mu["year"] = pd.to_datetime(mu["date"]).dt.year
        report["trading_days"] = int(mu["date"].nunique())
        report["tickers_per_year"] = mu.groupby("year")["ticker"].nunique().to_dict()
        eligible_per_day = mu[mu["eligible"]].groupby("date")["ticker"].nunique()
        report["median_eligible_universe_size"] = (
            float(eligible_per_day.median()) if not eligible_per_day.empty else None
        )
        report["missing_market_cap_pct"] = float(mu["market_cap"].isna().mean() * 100.0)
        report["missing_ohlc_pct"] = float(mu[["close", "high", "low"]].isna().any(axis=1).mean() * 100.0)
        if eligible_per_day.empty or eligible_per_day.median() < 50:
            report["warnings"].append(
                "Median eligible universe size is unexpectedly small (<50) — check filters/data coverage."
            )

        # Diagnostic only. Found via manual data exploration of the 2023
        # test backfill: a small number of tickers (mostly micro-caps with
        # several reverse splits) show ADJUSTED close prices in the hundreds
        # of billions of USD (e.g. MULN ~$14B/share on 2023-06-15 vs. a real
        # traded price of ~$1-3). Root cause (confirmed, see
        # config/market_cap_methodology.yaml `why_unadjusted_close_specifically`):
        # adjusted=true back-adjusts using ALL splits known as of the BUILD
        # date, not just the historical date. `market_cap` has already been
        # fixed to use the separately-fetched UNADJUSTED close instead (see
        # universe/build_universe.py) and is unaffected by this. Every other
        # feature/outcome is a ratio computed within one ticker's own
        # (uniformly scaled) series and is provably scale-invariant — see
        # tests/test_scale_invariance.py. This diagnostic exists so an
        # anomalously-scaled `close`/`high`/`low`/`atr14` column (still on
        # the adjusted-to-build-time basis, by design, for ratio features)
        # is never mistaken for a literal historical trade price.
        # $1,000,000 is chosen as a threshold safely above the highest
        # legitimate US common stock price on record (BRK.A, ~$700k).
        implausible = mu[mu["close"] > 1_000_000]
        report["implausible_price_row_count"] = int(len(implausible))
        report["implausible_price_ticker_count"] = int(implausible["ticker"].nunique())
        if not implausible.empty:
            report["warnings"].append(
                f"{len(implausible)} row(s) across {implausible['ticker'].nunique()} ticker(s) have an "
                f"ADJUSTED close price above $1,000,000 — expected artifact of adjusted=true using "
                f"build-time split knowledge (see config/market_cap_methodology.yaml). market_cap already "
                f"uses the unadjusted close and is unaffected; all other features are scale-invariant "
                f"(tests/test_scale_invariance.py). Do not read `close`/`high`/`low`/`atr14` as literal "
                f"historical prices for these tickers. Example tickers: "
                f"{sorted(implausible['ticker'].unique())[:10]}"
            )
    else:
        report["warnings"].append("market_universe_daily is empty.")

    report["stock_feature_rows"] = int(len(stock_features_daily))
    report["stock_outcome_rows"] = int(len(stock_outcomes_daily))

    if qqq_health_error:
        report["qqq_health_status"] = "unavailable"
        report["qqq_health_error"] = qqq_health_error
        report["warnings"].append(f"QQQ health track unavailable: {qqq_health_error}")
    else:
        report["qqq_health_status"] = "available"
        report["qqq_health_rows"] = int(len(qqq_health_daily)) if qqq_health_daily is not None else 0
        report["qqq_health_outcome_rows"] = (
            int(len(qqq_health_outcomes_daily)) if qqq_health_outcomes_daily is not None else 0
        )
        if qqq_constituent_component_counts is not None and not qqq_constituent_component_counts.empty:
            cc = qqq_constituent_component_counts
            report["qqq_component_count_median"] = float(cc["component_count"].median())
            report["qqq_component_count_min"] = int(cc["component_count"].min())
            report["qqq_component_count_max"] = int(cc["component_count"].max())
            if cc["component_count"].min() < 90:
                report["warnings"].append(
                    "At least one day had fewer than 90 QQQ constituents — check for data gaps."
                )

    return report


def write_data_quality_report(report: dict, out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or (REPO_ROOT / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_quality_report.json"
    md_path = out_dir / "data_quality_report.md"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = ["# Data Quality Report", "", f"Generated: {report.get('generated_at_utc')}", ""]
    lines.append(f"- Trading days: {report.get('trading_days', 'n/a')}")
    lines.append(f"- Tickers per year: {report.get('tickers_per_year', 'n/a')}")
    lines.append(f"- Median eligible universe size: {report.get('median_eligible_universe_size', 'n/a')}")
    lines.append(f"- Missing market cap %: {report.get('missing_market_cap_pct', 'n/a')}")
    lines.append(f"- Missing OHLC %: {report.get('missing_ohlc_pct', 'n/a')}")
    lines.append(f"- Implausible price rows (close > $1,000,000): "
                  f"{report.get('implausible_price_row_count', 'n/a')} "
                  f"across {report.get('implausible_price_ticker_count', 'n/a')} ticker(s)")
    lines.append(f"- Stock feature rows: {report.get('stock_feature_rows', 'n/a')}")
    lines.append(f"- Stock outcome rows: {report.get('stock_outcome_rows', 'n/a')}")
    lines.append(f"- QQQ health status: {report.get('qqq_health_status', 'n/a')}")
    if report.get("qqq_health_status") == "available":
        lines.append(f"- QQQ health rows: {report.get('qqq_health_rows', 'n/a')}")
        lines.append(f"- QQQ health outcome rows: {report.get('qqq_health_outcome_rows', 'n/a')}")
        lines.append(f"- QQQ component count (median/min/max): "
                      f"{report.get('qqq_component_count_median', 'n/a')} / "
                      f"{report.get('qqq_component_count_min', 'n/a')} / "
                      f"{report.get('qqq_component_count_max', 'n/a')}")
    else:
        lines.append(f"- QQQ health error: {report.get('qqq_health_error', 'n/a')}")

    if report.get("warnings"):
        lines.append("")
        lines.append("## Warnings")
        for w in report["warnings"]:
            lines.append(f"- {w}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
