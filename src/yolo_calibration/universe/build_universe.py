"""Daily YOLO Research Universe construction.

Filter order is fixed (cost reasons, config/universe.yaml `filter_order`):
  1. asset type  (cheap: from already-checkpointed point-in-time reference
     ticker snapshots)
  2. ADR20       (cheap: computed locally from already-fetched OHLCV)
  3. market cap  (expensive: one API call per surviving ticker-day, cached)

Produces the `market_universe_daily` table: date, ticker, asset_type_ok,
adr20, market_cap, eligible.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from yolo_calibration.config import load_universe_config
from yolo_calibration.data.cache import cached_call
from yolo_calibration.data.loaders import load_grouped_daily_range
from yolo_calibration.data.massive_client import MassiveClient
from yolo_calibration.data.storage import list_raw_grouped_daily_dates, read_raw_reference_tickers
from yolo_calibration.features.technical import compute_adr20
from yolo_calibration.utils.logging import get_logger

logger = get_logger(__name__)


def _load_asset_type_ok(start: date, end: date) -> pd.DataFrame:
    """Point-in-time asset-type-eligible (ticker, date) pairs, from the
    per-day reference-ticker checkpoints (already server-side filtered to
    type=CS; exclude_otc applied defensively here on primary_exchange)."""
    cfg = load_universe_config()
    exclude_otc = cfg["asset_type"]["exclude_otc"]
    rows = []
    for d in list_raw_grouped_daily_dates(start, end):
        snap = read_raw_reference_tickers(d)
        if snap is None or snap.empty:
            continue
        s = snap.copy()
        if exclude_otc and "primary_exchange" in s.columns:
            s = s[~s["primary_exchange"].astype(str).str.upper().str.startswith("OTC")]
        if "active" in s.columns:
            pass  # active flag intentionally NOT used to filter — point-in-time date param already handles this
        for ticker in s["ticker"].unique():
            rows.append({"date": pd.Timestamp(d), "ticker": ticker, "asset_type_ok": True})
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "asset_type_ok"])
    return pd.DataFrame(rows)


def fetch_point_in_time_shares_outstanding(client: MassiveClient, ticker: str, bucket_date: date) -> float | None:
    """Point-in-time weighted_shares_outstanding for (ticker, bucket_date).
    `bucket_date` must already be the first of a month — this IS the
    monthly caching granularity described in config/market_cap_methodology.yaml
    (shares outstanding only changes at SEC filing boundaries, so querying
    with a first-of-month date is always a conservative, never-future,
    point-in-time value — just not maximally fresh within the month)."""
    key = f"{ticker}:{bucket_date.isoformat()}"

    def _fetch():
        overview = client.get_ticker_overview(ticker, as_of_date=bucket_date)
        if not overview:
            return None
        return overview.get("weighted_shares_outstanding")

    return cached_call("ticker_overview_shares", key, ttl_days=36500, fetch_fn=_fetch)


def compute_point_in_time_market_cap(client: MassiveClient, ticker: str, as_of_date: date,
                                      close_price: float) -> float | None:
    """market_cap = historical close * point-in-time weighted_shares_outstanding.
    See config/market_cap_methodology.yaml. Convenience single-row wrapper;
    build_market_universe_daily uses the batched
    fetch_point_in_time_shares_outstanding path directly for performance
    (one lookup per unique (ticker, month) instead of per ticker-day)."""
    shares = fetch_point_in_time_shares_outstanding(client, ticker, as_of_date.replace(day=1))
    if shares is None or close_price is None or pd.isna(close_price):
        return None
    return float(close_price) * float(shares)


def build_market_universe_daily(client: MassiveClient, start: date, end: date) -> pd.DataFrame:
    cfg = load_universe_config()
    adr_min = cfg["adr20"]["min_pct"]
    mcap_min = cfg["market_cap"]["min_usd"]

    logger.info("Loading grouped daily OHLCV for ADR20 computation...")
    ohlcv = load_grouped_daily_range(start, end)
    if ohlcv.empty:
        raise RuntimeError(
            "No raw grouped-daily checkpoints found for the requested range. "
            "Run `fetch-raw` / the historical build workflow first."
        )
    ohlcv = compute_adr20(ohlcv)

    logger.info("Loading point-in-time asset-type eligibility...")
    asset_type = _load_asset_type_ok(start, end)

    merged = ohlcv.merge(asset_type, on=["date", "ticker"], how="left")
    merged["asset_type_ok"] = merged["asset_type_ok"].fillna(False)

    merged["adr20_ok"] = merged["adr20"] > adr_min
    candidates = merged[merged["asset_type_ok"] & merged["adr20_ok"]].copy()

    # Shares outstanding is looked up per (ticker, month) — not per
    # ticker-day — since that's the actual caching/API granularity (see
    # fetch_point_in_time_shares_outstanding). Deduplicating BEFORE the
    # lookup loop turns what would be millions of ticker-day iterations
    # (mostly redundant repeats of the same handful of tickers across
    # consecutive days) into one lookup per unique ticker-month, then a
    # single vectorized merge + multiply back onto every candidate row.
    candidates["bucket_month"] = candidates["date"].values.astype("datetime64[M]")
    unique_ticker_months = candidates[["ticker", "bucket_month"]].drop_duplicates().reset_index(drop=True)
    logger.info(
        "Market-cap enrichment: %d unique ticker-months to look up (from %d candidate ticker-days, "
        "post asset-type + ADR20 filter)...",
        len(unique_ticker_months), len(candidates),
    )
    shares_values = [
        fetch_point_in_time_shares_outstanding(client, row.ticker, pd.Timestamp(row.bucket_month).date())
        for row in unique_ticker_months.itertuples(index=False)
    ]
    unique_ticker_months["shares_outstanding"] = shares_values

    candidates = candidates.merge(unique_ticker_months, on=["ticker", "bucket_month"], how="left")
    candidates["market_cap"] = candidates["close"] * candidates["shares_outstanding"]
    candidates["market_cap_ok"] = candidates["market_cap"].fillna(0) >= mcap_min

    out = merged.merge(
        candidates[["date", "ticker", "market_cap", "market_cap_ok"]],
        on=["date", "ticker"],
        how="left",
    )
    out["market_cap_ok"] = out["market_cap_ok"].fillna(False)
    out["eligible"] = out["asset_type_ok"] & out["adr20_ok"] & out["market_cap_ok"]

    return out[[
        "date", "ticker", "close", "high", "low",
        "asset_type_ok", "adr20", "adr20_ok",
        "market_cap", "market_cap_ok", "eligible",
    ]].sort_values(["date", "ticker"]).reset_index(drop=True)
