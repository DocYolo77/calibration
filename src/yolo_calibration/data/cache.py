"""Simple disk cache for reference/fundamental data (spec section 3: "Reference-
/Fundamental-Daten sinnvoll cachen"). Not used for OHLCV bars, which are
written directly to the partitioned parquet tables in storage.py.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from yolo_calibration.config import REPO_ROOT, load_massive_api_config
from yolo_calibration.utils.logging import get_logger

logger = get_logger(__name__)


def _cache_dir() -> Path:
    cfg = load_massive_api_config()["caching"]
    d = REPO_ROOT / cfg["reference_data_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key_to_filename(namespace: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return _cache_dir() / f"{namespace}__{digest}.json"


def cached_call(namespace: str, key: str, ttl_days: float, fetch_fn: Callable[[], Any]) -> Any:
    """Return cached value for (namespace, key) if fresh, else call fetch_fn,
    persist the result, and return it. `key` should encode all params that
    affect the result (e.g. f"{ticker}:{date}")."""
    path = _key_to_filename(namespace, key)
    now = time.time()
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            age_days = (now - payload["_cached_at"]) / 86400.0
            if age_days <= ttl_days:
                return payload["value"]
        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupt cache entry %s, refetching", path)

    value = fetch_fn()
    path.write_text(
        json.dumps({"_cached_at": now, "_key": key, "value": value}, default=str),
        encoding="utf-8",
    )
    return value
