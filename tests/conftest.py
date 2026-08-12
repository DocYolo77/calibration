from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def make_ohlcv(ticker: str, start: str, n_days: int, *, base_price: float = 100.0,
               daily_drift_pct: float = 0.0, daily_range_pct: float = 2.0,
               seed: int = 0) -> pd.DataFrame:
    """Deterministic synthetic daily OHLCV series for one ticker, business
    days only, for use in unit tests (no network / API access)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    closes = [base_price]
    for _ in range(1, n_days):
        pct = daily_drift_pct + rng.normal(0, 0.3)
        closes.append(closes[-1] * (1 + pct / 100.0))
    closes = np.array(closes)
    highs = closes * (1 + daily_range_pct / 200.0)
    lows = closes * (1 - daily_range_pct / 200.0)
    opens = closes * (1 + rng.normal(0, 0.1, size=n_days) / 100.0)
    return pd.DataFrame({
        "date": dates,
        "ticker": ticker,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": rng.integers(1_000_000, 5_000_000, size=n_days),
        "vwap": closes,
    })


@pytest.fixture
def two_ticker_ohlcv() -> pd.DataFrame:
    a = make_ohlcv("AAA", "2023-01-02", 80, base_price=50.0, seed=1)
    b = make_ohlcv("BBB", "2023-01-02", 80, base_price=200.0, daily_drift_pct=0.1, seed=2)
    return pd.concat([a, b], ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
