"""Point-in-time "previous extension / repeat-offender" history features
(Phase 2, reports/opportunity_state_study.py) — config/features.yaml
`prior_extension_history` block. These are feature DEFINITIONS, not a
"resetting"/"extended" threshold: no state classification happens here.

Leakage guarantee: every value at row t is computed using ONLY rows
strictly BEFORE t for the rolling-max and peak columns (a trailing window
[t-w, t), t itself excluded) — the peak day is therefore always strictly
earlier than the signal day by construction, not by a runtime check. The
two "since peak" columns look from the day AFTER the peak through and
including day t itself; using day t's own EMA distance there is not future
leakage — it is simply "how close has price come to the EMA since the
peak, as of today," the same kind of same-day snapshot every other feature
in this codebase already uses (e.g. atr_extension itself).

Operates on a tidy per-ticker-sorted DataFrame with columns: date, ticker,
atr_extension, distance_ema10_pct, distance_ema20_pct. Requires the input
to already span enough trailing history for the configured windows (see
CLI/report layer for how the lookback range is assembled) — a ticker with
less history than `peak_window_days` simply gets a smaller, honestly
partial window (tracked via n_valid_days_prior_peak_window), never a
future-looking or fabricated one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from yolo_calibration.config import load_features_config


def compute_prior_extension_history(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_features_config()["prior_extension_history"]
    windows = list(cfg["windows_days"])
    peak_window = cfg["peak_window_days"]
    if peak_window not in windows:
        raise ValueError("prior_extension_history.peak_window_days must be one of windows_days")
    other_windows = [w for w in windows if w != peak_window]

    out = df.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    n = len(out)

    ext = out["atr_extension"].to_numpy(dtype=float)
    d10 = out["distance_ema10_pct"].to_numpy(dtype=float)
    d20 = out["distance_ema20_pct"].to_numpy(dtype=float)
    tickers = out["ticker"].to_numpy()

    max_cols = {w: np.full(n, np.nan) for w in other_windows}
    peak_val = np.full(n, np.nan)
    n_valid_peak_window = np.zeros(n, dtype=int)
    days_since_peak = np.full(n, np.nan)
    drawdown_from_peak = np.full(n, np.nan)
    min_d10_since_peak = np.full(n, np.nan)
    min_d20_since_peak = np.full(n, np.nan)

    ticker_change = np.r_[True, tickers[1:] != tickers[:-1]] if n else np.array([], dtype=bool)
    ticker_start_idx = np.flatnonzero(ticker_change)
    ticker_end_idx = np.r_[ticker_start_idx[1:], n]

    for t_start, t_end in zip(ticker_start_idx, ticker_end_idx):
        t_ext = ext[t_start:t_end]
        t_d10 = d10[t_start:t_end]
        t_d20 = d20[t_start:t_end]
        m = t_end - t_start

        for i in range(m):
            row = t_start + i

            for w in other_windows:
                lo = max(0, i - w)
                window = t_ext[lo:i]
                if window.size and np.any(~np.isnan(window)):
                    max_cols[w][row] = np.nanmax(window)

            lo_peak = max(0, i - peak_window)
            peak_window_vals = t_ext[lo_peak:i]
            if peak_window_vals.size == 0 or np.all(np.isnan(peak_window_vals)):
                continue

            valid_mask = ~np.isnan(peak_window_vals)
            n_valid_peak_window[row] = int(valid_mask.sum())
            rel_idx = int(np.nanargmax(peak_window_vals))
            peak_idx = lo_peak + rel_idx  # strictly < i, i.e. strictly before the signal day
            pv = float(t_ext[peak_idx])
            peak_val[row] = pv
            days_since_peak[row] = i - peak_idx

            if not np.isnan(t_ext[i]):
                drawdown_from_peak[row] = t_ext[i] - pv

            since_peak_d10 = t_d10[peak_idx + 1 : i + 1]
            since_peak_d20 = t_d20[peak_idx + 1 : i + 1]
            if since_peak_d10.size and np.any(~np.isnan(since_peak_d10)):
                min_d10_since_peak[row] = np.nanmin(since_peak_d10)
            if since_peak_d20.size and np.any(~np.isnan(since_peak_d20)):
                min_d20_since_peak[row] = np.nanmin(since_peak_d20)

    for w in other_windows:
        out[f"max_atr_extension_prior_{w}d"] = max_cols[w]
    out[f"max_atr_extension_prior_{peak_window}d"] = peak_val
    out["atr_extension_peak"] = peak_val  # alias: the canonical "previous peak" height
    out["n_valid_days_prior_peak_window"] = n_valid_peak_window
    out["days_since_atr_extension_peak"] = days_since_peak
    out["atr_extension_drawdown_from_peak"] = drawdown_from_peak
    out["min_distance_ema10_pct_since_peak"] = min_d10_since_peak
    out["min_distance_ema20_pct_since_peak"] = min_d20_since_peak
    out["touched_ema10_since_peak"] = np.where(
        np.isnan(min_d10_since_peak), np.nan, (min_d10_since_peak <= 0).astype(float)
    )
    out["touched_ema20_since_peak"] = np.where(
        np.isnan(min_d20_since_peak), np.nan, (min_d20_since_peak <= 0).astype(float)
    )
    return out
