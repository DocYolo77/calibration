"""Spec section 18 explicitly requires: "einen Test einbauen, der fehlschlägt,
wenn für einen historischen QQQ-Health-Run versehentlich eine statische
heutige Komponentenliste verwendet wird." This file is that test, plus the
"hard stop, no silent fallback" requirement from section 10.
"""

from datetime import date

import pandas as pd
import pytest

import yolo_calibration.qqq_health.constituents as qc


def test_membership_reflects_the_day_not_a_static_current_list(monkeypatch):
    """Ticker OLD_CO was a QQQ constituent only on the early date; NEW_CO
    joined only on the later date (as would happen after an index
    rebalance / addition-removal event). A correct point-in-time build
    must show OLD_CO present on d1/absent on d2, and the reverse for
    NEW_CO. A build that (bug) applied "today's" static constituent list
    to all historical dates would instead show the SAME ticker set on
    both days — which is exactly what this test catches."""
    d1, d2 = date(2023, 3, 1), date(2023, 3, 2)

    def fake_dates(start, end):
        return [d1, d2]

    def fake_read(d):
        if d == d1:
            return pd.DataFrame({"constituent_ticker": ["OLD_CO", "STABLE_CO"], "weight": [0.01, 0.02]})
        if d == d2:
            return pd.DataFrame({"constituent_ticker": ["NEW_CO", "STABLE_CO"], "weight": [0.015, 0.02]})
        return None

    monkeypatch.setattr(qc, "list_raw_grouped_daily_dates", fake_dates)
    monkeypatch.setattr(qc, "read_raw_qqq_constituents", fake_read)
    # Lower the min-component floor for this tiny synthetic fixture.
    import yolo_calibration.config as cfgmod
    real_cfg = cfgmod.load_qqq_health_config()
    patched_cfg = {**real_cfg, "constituents": {**real_cfg["constituents"],
                                                 "min_required_components_for_valid_day": 1}}
    monkeypatch.setattr(qc, "load_qqq_health_config", lambda: patched_cfg)

    out = qc.build_qqq_constituents_daily(d1, d2)

    day1_members = set(out[out["date"] == pd.Timestamp(d1)]["constituent_ticker"])
    day2_members = set(out[out["date"] == pd.Timestamp(d2)]["constituent_ticker"])

    assert day1_members == {"OLD_CO", "STABLE_CO"}
    assert day2_members == {"NEW_CO", "STABLE_CO"}

    # This is the assertion that fails under a "static current list" bug:
    # a static-list implementation would make day1_members == day2_members.
    assert day1_members != day2_members
    assert "OLD_CO" not in day2_members
    assert "NEW_CO" not in day1_members


def test_hard_stop_when_pit_data_missing_no_silent_fallback(monkeypatch):
    """If a trading day has no point-in-time constituent snapshot, the
    build must raise — never silently substitute another day's (e.g.
    today's) holdings."""
    d1, d2 = date(2023, 3, 1), date(2023, 3, 2)
    monkeypatch.setattr(qc, "list_raw_grouped_daily_dates", lambda start, end: [d1, d2])

    def fake_read(d):
        if d == d1:
            return pd.DataFrame({"constituent_ticker": ["A"], "weight": [0.5]})
        return None  # d2: no data available at all

    monkeypatch.setattr(qc, "read_raw_qqq_constituents", fake_read)
    import yolo_calibration.config as cfgmod
    real_cfg = cfgmod.load_qqq_health_config()
    patched_cfg = {**real_cfg, "constituents": {**real_cfg["constituents"],
                                                 "min_required_components_for_valid_day": 1}}
    monkeypatch.setattr(qc, "load_qqq_health_config", lambda: patched_cfg)

    with pytest.raises(qc.QQQConstituentsUnavailable):
        qc.build_qqq_constituents_daily(d1, d2)


def test_component_counts_are_stored_per_day():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2023-03-01")] * 2 + [pd.Timestamp("2023-03-02")] * 3,
        "constituent_ticker": ["A", "B", "A", "B", "C"],
    })
    counts = qc.component_counts(df)
    assert counts[counts["date"] == pd.Timestamp("2023-03-01")]["component_count"].iloc[0] == 2
    assert counts[counts["date"] == pd.Timestamp("2023-03-02")]["component_count"].iloc[0] == 3
