"""Central config loader.

Every research-relevant number (dates, filters, feature/indicator windows,
outcome horizons, API endpoints) is defined in YAML under `config/` and
loaded through this module. No other module should hardcode a duplicate of
these values — see config/README_PHASE1_SCOPE.md.
"""

from __future__ import annotations

import functools
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        raise ValueError(f"Config file {path} is empty or invalid.")
    return data


@functools.lru_cache(maxsize=None)
def load_time_splits() -> dict[str, Any]:
    return _load_yaml("time_splits.yaml")


@functools.lru_cache(maxsize=None)
def load_universe_config() -> dict[str, Any]:
    return _load_yaml("universe.yaml")


@functools.lru_cache(maxsize=None)
def load_market_cap_methodology() -> dict[str, Any]:
    return _load_yaml("market_cap_methodology.yaml")


@functools.lru_cache(maxsize=None)
def load_features_config() -> dict[str, Any]:
    return _load_yaml("features.yaml")


@functools.lru_cache(maxsize=None)
def load_qqq_health_config() -> dict[str, Any]:
    return _load_yaml("qqq_health.yaml")


@functools.lru_cache(maxsize=None)
def load_massive_api_config() -> dict[str, Any]:
    return _load_yaml("massive_api.yaml")


@dataclass(frozen=True)
class TimeSplit:
    name: str
    start: date
    end: date | None  # None means "resolve dynamically to latest available date"


def get_time_split(name: str, *, resolve_end: date | None = None) -> TimeSplit:
    """Return a TimeSplit. For splits with an open end (out_of_sample), pass
    resolve_end (typically the latest fully-available trading day) to pin it.
    """
    splits = load_time_splits()["splits"]
    if name not in splits:
        raise KeyError(f"Unknown time split '{name}'. Known: {list(splits)}")
    raw = splits[name]
    start = datetime.strptime(raw["start"], "%Y-%m-%d").date()
    end_raw = raw.get("end")
    if end_raw is None:
        end = resolve_end
    else:
        end = datetime.strptime(end_raw, "%Y-%m-%d").date()
    return TimeSplit(name=name, start=start, end=end)


def get_history_buffer_start() -> date:
    raw = load_time_splits()["data_fetch"]["history_buffer_start"]
    return datetime.strptime(raw, "%Y-%m-%d").date()


def get_massive_api_key() -> str:
    import os

    key = os.environ.get(load_massive_api_config()["auth"]["env_var"], "")
    if not key:
        raise RuntimeError(
            "MASSIVE_API_KEY is not set. Set it as an environment variable "
            "or GitHub Actions secret — never store it in a file."
        )
    return key


def get_code_commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class BuildMetadata:
    """Reproducibility metadata attached to every generated dataset
    (project spec section 15)."""

    build_timestamp_utc: str
    data_start: str
    data_end: str
    config_version: int
    code_commit_hash: str
    source: str
    feature_definition_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "build_timestamp_utc": self.build_timestamp_utc,
            "data_start": self.data_start,
            "data_end": self.data_end,
            "config_version": self.config_version,
            "code_commit_hash": self.code_commit_hash,
            "source": self.source,
            "feature_definition_version": self.feature_definition_version,
        }


def make_build_metadata(
    data_start: date,
    data_end: date,
    source: str,
    feature_definition_version: str = "phase1-v1",
) -> BuildMetadata:
    features_cfg = load_features_config()
    return BuildMetadata(
        build_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        data_start=data_start.isoformat(),
        data_end=data_end.isoformat(),
        config_version=features_cfg.get("schema_version", 0),
        code_commit_hash=get_code_commit_hash(),
        source=source,
        feature_definition_version=feature_definition_version,
    )
