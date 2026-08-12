"""Thin REST client for the Massive (formerly Polygon.io) market data API.

Design constraints (see project spec section 2):
  - MASSIVE_API_KEY is read only from the environment, never from a file.
  - The key is never written to logs, exceptions, or cached files.
  - Bulk/grouped-daily endpoints are preferred over per-ticker requests.
  - All requests are retried with backoff on transient failures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from yolo_calibration.config import get_massive_api_key, load_massive_api_config
from yolo_calibration.utils.logging import get_logger

logger = get_logger(__name__)


class MassiveAPIError(RuntimeError):
    """Raised for non-retryable API failures. Never includes the API key."""


@dataclass
class _RateLimiter:
    requests_per_minute: int
    _window_start: float = 0.0
    _count: int = 0

    def wait(self) -> None:
        now = time.monotonic()
        if now - self._window_start >= 60:
            self._window_start = now
            self._count = 0
        if self._count >= self.requests_per_minute:
            sleep_for = 60 - (now - self._window_start)
            if sleep_for > 0:
                logger.info("Rate limit reached, sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)
            self._window_start = time.monotonic()
            self._count = 0
        self._count += 1


class MassiveClient:
    """Session-backed client. Construct once and reuse across a build run."""

    def __init__(self, api_key: str | None = None, config: dict[str, Any] | None = None):
        self._cfg = config or load_massive_api_config()
        self._api_key = api_key or get_massive_api_key()
        self._base_url = self._cfg["base_url"].rstrip("/")
        http_cfg = self._cfg["http"]
        self._timeout = http_cfg["timeout_seconds"]

        retry = Retry(
            total=http_cfg["max_retries"],
            backoff_factor=http_cfg["backoff_factor_seconds"],
            status_forcelist=http_cfg["retry_on_status"],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session = requests.Session()
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._session.headers.update({"Authorization": f"Bearer {self._api_key}"})

        self._rate_limiter = _RateLimiter(http_cfg["requests_per_minute_soft_limit"])

    # -- low level -----------------------------------------------------

    def _get(self, path_or_url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = path_or_url if path_or_url.startswith("http") else f"{self._base_url}{path_or_url}"
        self._rate_limiter.wait()
        resp = self._session.get(url, params=params, timeout=self._timeout)
        if resp.status_code == 401:
            raise MassiveAPIError(
                "Massive API returned 401 Unauthorized. Check that MASSIVE_API_KEY "
                "is set and valid. (Key value is intentionally not shown.)"
            )
        if resp.status_code == 404:
            raise MassiveAPIError(f"Massive API 404 for {path_or_url} (params={params})")
        if not resp.ok:
            # Never include headers (which carry the Authorization token) in the error.
            raise MassiveAPIError(
                f"Massive API request failed: {resp.status_code} for {path_or_url} "
                f"body={resp.text[:500]!r}"
            )
        return resp.json()

    def _paginate(self, path: str, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        payload = self._get(path, params)
        while True:
            for item in payload.get("results", []) or []:
                yield item
            next_url = payload.get("next_url")
            if not next_url:
                break
            payload = self._get(next_url)

    # -- endpoints -------------------------------------------------------

    def get_grouped_daily(self, trading_date: date, *, adjusted: bool = True,
                           include_otc: bool = False) -> list[dict[str, Any]]:
        """Bulk OHLCV for ALL US stocks on one date (single request)."""
        ep = self._cfg["endpoints"]["grouped_daily"]
        path = ep["path"].format(date=trading_date.isoformat())
        payload = self._get(path, {"adjusted": str(adjusted).lower(),
                                    "include_otc": str(include_otc).lower()})
        return payload.get("results", []) or []

    def get_reference_tickers(self, *, as_of_date: date | None = None,
                               active: bool | None = None,
                               market: str = "stocks",
                               ticker_type: str | None = None,
                               limit: int = 1000) -> Iterator[dict[str, Any]]:
        """Point-in-time ticker reference list. Paginates transparently."""
        ep = self._cfg["endpoints"]["reference_tickers"]
        params: dict[str, Any] = {"market": market, "limit": limit}
        if as_of_date is not None:
            params["date"] = as_of_date.isoformat()
        if active is not None:
            params["active"] = str(active).lower()
        if ticker_type is not None:
            params["type"] = ticker_type
        yield from self._paginate(ep["path"], params)

    def get_ticker_types(self, *, asset_class: str = "stocks", locale: str = "us") -> list[dict[str, Any]]:
        ep = self._cfg["endpoints"]["ticker_types"]
        payload = self._get(ep["path"], {"asset_class": asset_class, "locale": locale})
        return payload.get("results", []) or []

    def get_ticker_overview(self, ticker: str, *, as_of_date: date | None = None) -> dict[str, Any] | None:
        """Point-in-time shares-outstanding / reference data for one ticker."""
        ep = self._cfg["endpoints"]["ticker_overview"]
        path = ep["path"].format(ticker=ticker)
        params = {}
        if as_of_date is not None:
            params["date"] = as_of_date.isoformat()
        try:
            payload = self._get(path, params)
        except MassiveAPIError as exc:
            if "404" in str(exc):
                return None
            raise
        return payload.get("results")

    def get_etf_constituents(self, *, composite_ticker: str,
                              effective_date: date | None = None,
                              limit: int = 5000) -> list[dict[str, Any]]:
        """Point-in-time ETF holdings (e.g. QQQ) as of effective_date."""
        ep = self._cfg["endpoints"]["etf_constituents"]
        params: dict[str, Any] = {"composite_ticker": composite_ticker, "limit": limit}
        if effective_date is not None:
            params["effective_date"] = effective_date.isoformat()
        return list(self._paginate(ep["path"], params))
