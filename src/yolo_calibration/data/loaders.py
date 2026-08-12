"""Helpers to assemble raw per-day checkpoint files into tidy DATE x TICKER
DataFrames used by universe/feature/outcome builders."""

from __future__ import annotations

from datetime import date

import pandas as pd

from yolo_calibration.data.storage import (
    list_raw_grouped_daily_dates,
    list_raw_grouped_daily_unadjusted_dates,
    read_raw_grouped_daily,
    read_raw_grouped_daily_unadjusted,
)

_RENAME = {
    "T": "ticker",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vw": "vwap",
    "n": "transactions",
    "t": "epoch_ms",
}


def load_grouped_daily_range(start: date, end: date) -> pd.DataFrame:
    """Concatenate raw grouped-daily checkpoint files in [start, end] into a
    single tidy DataFrame: date, ticker, open, high, low, close, volume, vwap.
    Only dates that already have a raw checkpoint are included — callers are
    responsible for having fetched the range first (see cli.py build-universe
    / fetch step)."""
    frames = []
    for d in list_raw_grouped_daily_dates(start, end):
        raw = read_raw_grouped_daily(d)
        if raw is None or raw.empty:
            continue
        df = raw.rename(columns=_RENAME).copy()
        df["date"] = pd.Timestamp(d)
        keep = ["date", "ticker", "open", "high", "low", "close", "volume", "vwap"]
        for col in keep:
            if col not in df.columns:
                df[col] = pd.NA
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume", "vwap"])
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def load_grouped_daily_unadjusted_range(start: date, end: date) -> pd.DataFrame:
    """Concatenate raw UNADJUSTED grouped-daily checkpoint files into a
    tidy DataFrame: date, ticker, close_unadjusted. Used only for
    point-in-time-correct market cap (see universe/build_universe.py)."""
    frames = []
    for d in list_raw_grouped_daily_unadjusted_dates(start, end):
        raw = read_raw_grouped_daily_unadjusted(d)
        if raw is None or raw.empty:
            continue
        df = raw.rename(columns=_RENAME).copy()
        df["date"] = pd.Timestamp(d)
        if "close" not in df.columns:
            df["close"] = pd.NA
        frames.append(df[["date", "ticker", "close"]].rename(columns={"close": "close_unadjusted"}))
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "close_unadjusted"])
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out
