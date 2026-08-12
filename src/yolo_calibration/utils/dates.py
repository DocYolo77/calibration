"""Date helpers. The authoritative trading calendar is derived empirically
from actual grouped-daily API responses (a date with zero or negligible
results is treated as a non-trading day), not from a hardcoded holiday list.
"""

from __future__ import annotations

from datetime import date, timedelta


def iter_candidate_weekdays(start: date, end: date):
    """Mon-Fri dates in [start, end], inclusive. Includes market holidays —
    callers must skip days where the API returns no/empty data, since that
    is the ground truth for "was this a trading day"."""
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)
