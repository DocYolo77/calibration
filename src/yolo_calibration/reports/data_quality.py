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

from yolo_calibration.config import REPO_ROOT, load_universe_config


def _market_cap_eligibility_sensitivity(mu: pd.DataFrame) -> dict:
    """Standing diagnostic (config/market_cap_methodology.yaml
    `market_cap_eligibility_sensitivity`): flags ticker-days whose
    eligibility is close enough to the $1B cutoff that a plausible vendor
    data error could flip it, and specifically calls out cases where that
    proximity coincides with an implausible month-over-month market_cap
    swing (the MULN-style shares-outstanding-instability pattern found
    2026-08-13 in the 2023 backfill). Descriptive only — never changes
    `eligible`/`market_cap_ok`."""
    cfg = load_universe_config()["market_cap"]
    min_usd = cfg["min_usd"]
    sens_cfg = cfg.get("sensitivity", {})
    band_pct = sens_cfg.get("near_threshold_band_pct", 0.15)
    vol_ratio_threshold = sens_cfg.get("volatility_ratio_threshold", 3.0)

    df = mu.dropna(subset=["market_cap"]).copy()
    if df.empty:
        return {
            "near_threshold_band_pct": band_pct,
            "volatility_ratio_threshold": vol_ratio_threshold,
            "near_threshold_row_count": 0,
            "near_threshold_ticker_count": 0,
            "near_threshold_volatile_ticker_count": 0,
            "near_threshold_volatile_tickers_example": [],
        }

    lower, upper = min_usd * (1 - band_pct), min_usd * (1 + band_pct)
    near = df[(df["market_cap"] >= lower) & (df["market_cap"] <= upper)]
    near_tickers = set(near["ticker"].unique())

    # Month-over-month market_cap ratio, per ticker, restricted to tickers
    # that have at least one near-threshold row (a volatile swing far from
    # the threshold is not an eligibility-sensitivity concern).
    volatile_tickers: set[str] = set()
    if near_tickers:
        near_hist = df[df["ticker"].isin(near_tickers)].copy()
        near_hist["month"] = pd.to_datetime(near_hist["date"]).values.astype("datetime64[M]")
        monthly = near_hist.groupby(["ticker", "month"])["market_cap"].mean().reset_index()
        monthly = monthly.sort_values(["ticker", "month"])
        monthly["ratio"] = monthly.groupby("ticker")["market_cap"].pct_change() + 1.0
        spikes = monthly[
            (monthly["ratio"] > vol_ratio_threshold) | (monthly["ratio"] < 1.0 / vol_ratio_threshold)
        ]
        volatile_tickers = set(spikes["ticker"].unique())

    return {
        "near_threshold_band_pct": band_pct,
        "volatility_ratio_threshold": vol_ratio_threshold,
        "near_threshold_row_count": int(len(near)),
        "near_threshold_ticker_count": int(len(near_tickers)),
        "near_threshold_volatile_ticker_count": int(len(volatile_tickers)),
        "near_threshold_volatile_tickers_example": sorted(volatile_tickers)[:10],
    }


def _tie_break_diagnostics(stock_outcomes_daily: pd.DataFrame) -> dict:
    """Aggregates the `reached_plus_X_before_minus_X_tie_{H}d` columns
    (outcomes/build_outcomes.py) into per-column counts: how many rows hit
    both the +X% and -X% threshold on the SAME forward trading day, making
    the true intraday order non-determinable from daily OHLC alone. The
    conservative tie-break convention itself is unchanged — this is a
    count/audit of how often it had to be invoked, not a correction."""
    tie_cols = [c for c in stock_outcomes_daily.columns if c.endswith("_tie_5d")
                or c.endswith("_tie_10d") or c.endswith("_tie_20d")]
    per_column = {}
    for c in sorted(tie_cols):
        base_col = c.replace("_tie_", "_")
        n_determinable = int(stock_outcomes_daily[base_col].notna().sum()) if base_col in stock_outcomes_daily else 0
        n_ties = int(stock_outcomes_daily[c].fillna(False).astype(bool).sum())
        per_column[c] = {
            "tie_count": n_ties,
            "window_complete_count": n_determinable,
            "tie_pct": (100.0 * n_ties / n_determinable) if n_determinable else 0.0,
        }
    return {
        "columns": per_column,
        "total_ties": sum(v["tie_count"] for v in per_column.values()),
    }


def build_data_quality_report(
    *,
    market_universe_daily: pd.DataFrame,
    stock_features_daily: pd.DataFrame,
    stock_outcomes_daily: pd.DataFrame,
    qqq_health_daily: pd.DataFrame | None,
    qqq_health_outcomes_daily: pd.DataFrame | None,
    qqq_constituent_component_counts: pd.DataFrame | None,
    qqq_health_error: str | None = None,
    market_breadth_daily: pd.DataFrame | None = None,
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

        sensitivity = _market_cap_eligibility_sensitivity(mu)
        report["market_cap_eligibility_sensitivity"] = sensitivity
        if sensitivity["near_threshold_ticker_count"] > 0:
            report["warnings"].append(
                f"{sensitivity['near_threshold_row_count']} row(s) across "
                f"{sensitivity['near_threshold_ticker_count']} ticker(s) have market_cap within "
                f"{sensitivity['near_threshold_band_pct'] * 100:.0f}% of the $1B eligibility cutoff — "
                f"eligibility here could plausibly flip on vendor shares-outstanding data alone. Of those, "
                f"{sensitivity['near_threshold_volatile_ticker_count']} ticker(s) also show a "
                f">{sensitivity['volatility_ratio_threshold']:.0f}x month-over-month market_cap swing "
                f"(see config/market_cap_methodology.yaml known_limitations — shares-outstanding "
                f"instability, e.g. frequent-reverse-split names). market_cap is NOT adjusted by this "
                f"diagnostic. Example tickers: {sensitivity['near_threshold_volatile_tickers_example']}"
            )
    else:
        report["warnings"].append("market_universe_daily is empty.")

    report["stock_feature_rows"] = int(len(stock_features_daily))
    report["stock_outcome_rows"] = int(len(stock_outcomes_daily))

    if not stock_outcomes_daily.empty:
        tie_diag = _tie_break_diagnostics(stock_outcomes_daily)
        report["reached_before_tie_break_diagnostics"] = tie_diag
        if tie_diag["total_ties"] > 0:
            report["warnings"].append(
                f"{tie_diag['total_ties']} row(s) (summed across all reached_plus_X_before_minus_X "
                f"horizon/threshold combinations) hit both thresholds on the SAME forward trading day — "
                f"the true intraday order is NOT determinable from daily OHLC. The documented conservative "
                f"tie-break (down-first) was applied; see reached_before_tie_break_diagnostics for the "
                f"per-column breakdown and outcomes/build_outcomes.py for the convention."
            )

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

    # market_breadth_daily is independent of QQQ health availability (see
    # outcomes/build_market_breadth.py) — reported unconditionally.
    if market_breadth_daily is not None and not market_breadth_daily.empty:
        report["market_breadth_rows"] = int(len(market_breadth_daily))
        for bucket in (80, 90, 95):
            col = f"future_rs{bucket}plus_share_20d"
            if col in market_breadth_daily.columns:
                report[f"market_breadth_{col}_median"] = float(market_breadth_daily[col].median())
    else:
        report["market_breadth_rows"] = 0

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
    sensitivity = report.get("market_cap_eligibility_sensitivity")
    if sensitivity:
        lines.append(
            f"- Market cap near $1B threshold (±{sensitivity['near_threshold_band_pct'] * 100:.0f}%): "
            f"{sensitivity['near_threshold_row_count']} row(s) across "
            f"{sensitivity['near_threshold_ticker_count']} ticker(s); of those, "
            f"{sensitivity['near_threshold_volatile_ticker_count']} ticker(s) also show a "
            f">{sensitivity['volatility_ratio_threshold']:.0f}x month-over-month market_cap swing "
            f"(shares-outstanding instability, see config/market_cap_methodology.yaml)"
        )
    lines.append(f"- Stock feature rows: {report.get('stock_feature_rows', 'n/a')}")
    lines.append(f"- Stock outcome rows: {report.get('stock_outcome_rows', 'n/a')}")
    tie_diag = report.get("reached_before_tie_break_diagnostics")
    if tie_diag:
        lines.append(f"- reached_plus_X_before_minus_X same-day ties (order not determinable from daily OHLC): "
                      f"{tie_diag['total_ties']} row(s) total")
        for col, stats in sorted(tie_diag["columns"].items()):
            lines.append(f"  - {col}: {stats['tie_count']} / {stats['window_complete_count']} "
                          f"({stats['tie_pct']:.3f}%)")
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

    lines.append(f"- Market breadth rows (market_breadth_daily, independent of QQQ health): "
                  f"{report.get('market_breadth_rows', 'n/a')}")
    for bucket in (80, 90, 95):
        key = f"market_breadth_future_rs{bucket}plus_share_20d_median"
        if key in report:
            lines.append(f"  - median future_rs{bucket}plus_share_20d: {report[key]:.4f}")

    if report.get("warnings"):
        lines.append("")
        lines.append("## Warnings")
        for w in report["warnings"]:
            lines.append(f"- {w}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
