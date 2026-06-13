---
name: tracking-leaf-export
description: Formal leaf-tracking export API lives in openalea.phenomenal.tracking.export
metadata:
  type: project
---

Leaf-tracking results are exported via `openalea.phenomenal.tracking.export` (re-exported from the `tracking` package): `export_leaf_tracking(tracked_plant, fmt=...)` plus `tracking_to_records` (list[dict], long format), `tracking_to_dataframe` (lazy pandas), `tracking_to_json`. Also as methods: `TrackedPlant.to_records/to_dataframe/to_json`.

Rows are one-per-(snapshot, rank) with fields: time, snapshot_index, rank (1-based; None if leaf unranked), leaf_id, mature, length, height, azimuth, check_continuity, missing, error. Robust by design: missing leaves / unaligned sequences / bad indices / incomplete features become None + `missing`/`error` markers instead of raising.

`TrackedSnapshot` now stores `.time` (added so exports carry the time point; previously discarded after `load()`).

**Why:** Added so downstream analysis/notebooks/services consume tracking output without reaching into internal objects.

**How to apply:** Keep the module numpy/pandas-free at import time so it stays usable in the broken local env — see [[env-python-validation]]. Unit tests use stub objects; the real-data integration test reuses the `test/data/tracking/time_series` fixture pattern from `test_tracking.py`.
