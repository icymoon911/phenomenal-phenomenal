"""
Export utilities for leaf tracking results.

After running the leaf tracking algorithm (:class:`~openalea.phenomenal.tracking.trackedPlant.TrackedPlant`,
``mature_leaf_tracking`` then optionally ``growing_leaf_tracking``), the matching
between time points and leaves lives inside the ``TrackedPlant`` / ``TrackedSnapshot``
objects. This module turns that internal state into stable, ready-to-consume
tables so that downstream analysis scripts, notebooks or services do not have to
reach into the tracking objects by hand.

The export is produced in *long format*: one row per (snapshot, rank). For every
snapshot and every aligned rank it reports the time point, the snapshot order, the
leaf id, the rank, the maturity status, and the ``length`` / ``height`` / ``azimuth``
features. Three output flavours are offered, all built from the same records:

- :func:`tracking_to_records` -> ``list[dict]`` (dict / JSON friendly, the core form)
- :func:`tracking_to_dataframe` -> ``pandas.DataFrame`` (pandas imported lazily)
- :func:`tracking_to_json` -> a JSON string (optionally written to a file)

:func:`export_leaf_tracking` is the single entry point dispatching on ``fmt``.

Robustness
----------
The export never raises on partial / inconsistent tracking results. Missing leaves
(a rank with no associated leaf), an alignment that was not run yet, a leaf index
pointing outside the snapshot, or a leaf whose feature dict is incomplete are all
reported as ``None`` values together with a ``missing`` flag and/or an ``error``
marker, so a single bad snapshot or leaf cannot bring down the whole export.

Ranks convention
-----------------
As documented in :mod:`~openalea.phenomenal.tracking.trackedPlant`, ranks start at 0
internally but are reported starting at 1 here. ``rank`` is ``None`` for a detected
leaf that was not assigned to any rank (see ``include_unranked``).
"""

import json

# Column order used for records and DataFrame output.
_FIELDS = (
    "time",
    "snapshot_index",
    "rank",
    "leaf_id",
    "mature",
    "length",
    "height",
    "azimuth",
    "check_continuity",
    "missing",
    "error",
)

# Feature keys expected on every TrackedLeaf.features dict.
_FEATURE_KEYS = ("mature", "length", "height", "azimuth")


def _to_native(value):
    """Convert numpy scalars to plain Python types and ``NaN`` to ``None``.

    Done without importing numpy (numpy scalars are detected by their ``item``
    method and ``dtype`` attribute) so this module stays import-light and usable
    even where numpy is unavailable. The result is always JSON serialisable.
    """

    if value is None:
        return None

    # numpy scalars / 0-d arrays expose both .item() and .dtype
    if hasattr(value, "item") and hasattr(value, "dtype"):
        try:
            value = value.item()
        except Exception:
            return value

    # NaN (float('nan') != itself) -> None, so JSON stays valid and missing reads clearly
    if isinstance(value, float) and value != value:
        return None

    return value


def _base_row(snapshot_index, time, check):
    """Return a row dict with every field present, defaults filled in."""

    row = {field: None for field in _FIELDS}
    row["snapshot_index"] = snapshot_index
    row["time"] = _to_native(time)
    row["check_continuity"] = _to_native(check)
    row["missing"] = False
    return row


def _fill_features(row, leaf):
    """Populate the feature columns of ``row`` from ``leaf.features``.

    Sets an ``error`` marker if the feature dict is absent or incomplete, but
    always copies whatever values are available.
    """

    features = getattr(leaf, "features", None)
    if not isinstance(features, dict):
        row["error"] = "leaf has no features dict"
        return row

    row["mature"] = _to_native(features.get("mature"))
    row["length"] = _to_native(features.get("length"))
    row["height"] = _to_native(features.get("height"))
    row["azimuth"] = _to_native(features.get("azimuth"))

    missing_keys = [key for key in _FEATURE_KEYS if key not in features]
    if missing_keys:
        row["error"] = "missing feature keys: " + ", ".join(missing_keys)
    return row


def _snapshot_rows(snapshot, snapshot_index, include_unranked):
    """Build the list of rows for a single snapshot.

    One row per aligned rank (surfacing missing leaves), followed by one row per
    detected leaf that was not assigned a rank when ``include_unranked`` is True.
    """

    time = getattr(snapshot, "time", None)
    check = getattr(snapshot, "check_continuity", None)
    leaves = list(getattr(snapshot, "leaves", []) or [])
    sequence = list(getattr(snapshot, "sequence", []) or [])

    rows = []

    # ----- one row per aligned rank -------------------------------------------
    for rank_index, leaf_index in enumerate(sequence):
        row = _base_row(snapshot_index, time, check)
        row["rank"] = rank_index + 1  # ranks reported starting at 1

        if leaf_index is None or leaf_index == -1:
            # a rank with no associated leaf == missing leaf at this rank
            row["missing"] = True
            rows.append(row)
            continue

        if not isinstance(leaf_index, int) or not (0 <= leaf_index < len(leaves)):
            # alignment references a leaf that does not exist in this snapshot
            row["leaf_id"] = leaf_index if isinstance(leaf_index, int) else None
            row["missing"] = True
            row["error"] = "leaf index {} out of range (n_leaves={})".format(
                leaf_index, len(leaves)
            )
            rows.append(row)
            continue

        row["leaf_id"] = leaf_index
        _fill_features(row, leaves[leaf_index])
        rows.append(row)

    # ----- detected leaves that were never assigned a rank --------------------
    if include_unranked:
        ranked = {i for i in sequence if isinstance(i, int) and i != -1}
        for leaf_index, leaf in enumerate(leaves):
            if leaf_index in ranked:
                continue
            row = _base_row(snapshot_index, time, check)
            row["rank"] = None  # detected but not tracked to a rank
            row["leaf_id"] = leaf_index
            _fill_features(row, leaf)
            rows.append(row)

    # ----- never silently drop a snapshot -------------------------------------
    if not rows:
        row = _base_row(snapshot_index, time, check)
        if not sequence:
            row["error"] = (
                "snapshot has no aligned sequence "
                "(run mature_leaf_tracking() first, or pass include_unranked=True)"
            )
        else:
            row["error"] = "snapshot produced no leaf rows"
        rows.append(row)

    return rows


def tracking_to_records(tracked_plant, include_unranked=True):
    """Export leaf tracking results as a list of plain dict records (long format).

    Parameters
    ----------
    tracked_plant : TrackedPlant
        A tracked plant whose ``mature_leaf_tracking`` (and optionally
        ``growing_leaf_tracking``) has already been run. Only the public
        attributes ``snapshots`` / ``snapshot.leaves`` / ``snapshot.sequence`` /
        ``snapshot.check_continuity`` / ``snapshot.time`` / ``leaf.features`` are
        read, so any object exposing the same attributes works.
    include_unranked : bool
        If True (default), detected leaves that were not assigned to any rank are
        also exported with ``rank=None``. If False, only aligned ranks are kept.

    Returns
    -------
    list of dict
        One dict per (snapshot, rank), with the keys listed in
        :data:`_FIELDS`. Values are plain Python types (JSON serialisable);
        unavailable values are ``None``. See module docstring for the robustness
        contract (``missing`` / ``error`` markers).
    """

    snapshots = getattr(tracked_plant, "snapshots", None)
    if snapshots is None:
        raise TypeError(
            "tracked_plant has no 'snapshots' attribute; "
            "expected a TrackedPlant-like object"
        )

    records = []
    for snapshot_index, snapshot in enumerate(snapshots):
        try:
            records.extend(
                _snapshot_rows(snapshot, snapshot_index, include_unranked)
            )
        except Exception as error:  # one bad snapshot must not break the export
            row = _base_row(snapshot_index, getattr(snapshot, "time", None), None)
            row["error"] = "failed to export snapshot: {!r}".format(error)
            records.append(row)

    return records


def tracking_to_dataframe(tracked_plant, include_unranked=True):
    """Export leaf tracking results as a :class:`pandas.DataFrame` (long format).

    Same content as :func:`tracking_to_records`, with stable column order. pandas
    is imported lazily so it is only required when this function is actually used.
    """

    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "pandas is required for tracking_to_dataframe(); install pandas, "
            "or use tracking_to_records() / tracking_to_json() instead."
        ) from error

    records = tracking_to_records(tracked_plant, include_unranked=include_unranked)
    return pd.DataFrame(records, columns=list(_FIELDS))


def tracking_to_json(tracked_plant, path=None, include_unranked=True, indent=2):
    """Export leaf tracking results as a JSON string (long format).

    Parameters
    ----------
    tracked_plant : TrackedPlant
    path : str or pathlib.Path, optional
        If given, the JSON is also written to this file (UTF-8).
    include_unranked : bool
    indent : int or None
        Passed through to :func:`json.dumps`.

    Returns
    -------
    str
        The JSON document (a list of record objects).
    """

    records = tracking_to_records(tracked_plant, include_unranked=include_unranked)
    text = json.dumps(records, indent=indent, ensure_ascii=False)

    if path is not None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    return text


def export_leaf_tracking(tracked_plant, fmt="records", **kwargs):
    """Export leaf tracking results in the requested format.

    Single entry point dispatching to :func:`tracking_to_records`,
    :func:`tracking_to_dataframe` or :func:`tracking_to_json`.

    Parameters
    ----------
    tracked_plant : TrackedPlant
    fmt : str
        One of ``"records"`` (default), ``"dataframe"`` or ``"json"``.
    **kwargs
        Forwarded to the selected exporter (e.g. ``include_unranked``, and
        ``path`` / ``indent`` for ``"json"``).

    Returns
    -------
    list of dict, pandas.DataFrame, or str
        Depending on ``fmt``.
    """

    key = str(fmt).lower()
    if key == "records":
        return tracking_to_records(tracked_plant, **kwargs)
    if key in ("dataframe", "df", "pandas"):
        return tracking_to_dataframe(tracked_plant, **kwargs)
    if key == "json":
        return tracking_to_json(tracked_plant, **kwargs)

    raise ValueError(
        "unknown fmt {!r}; expected one of 'records', 'dataframe', 'json'".format(fmt)
    )
