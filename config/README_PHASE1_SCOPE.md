# Config scope guard — Phase 1

All research-relevant numbers (dates, filters, windows, ATR/EMA/RS/thrust
definitions, outcome horizons) live in this directory as YAML, loaded through
`yolo_calibration.config`. No module under `src/yolo_calibration` may
hardcode a duplicate of these numbers — always load from config.

**What belongs here:** eligibility filters, feature/indicator definitions,
outcome-window definitions, time-split boundaries, API endpoint config.

**What does NOT belong here (Phase 1 explicitly excludes it):**
- Leader / fresh-leader / constructive-reset / extended definitions
- Thrust percentile cutoffs (e.g. 80/85/90)
- QQQ health state thresholds or state labels
- Market regime score definitions

Those are Phase 2 deliverables, calibrated on 2023-2024, validated on 2025,
and tested once on 2026 — not chosen by the coding agent. See
`config/time_splits.yaml` and the root `README.md` "Stop Condition" section.
