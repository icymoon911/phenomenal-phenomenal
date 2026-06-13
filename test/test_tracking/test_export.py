"""Tests for openalea.phenomenal.tracking.export.

Two layers:

* Pure-Python unit tests that exercise the export logic and its robustness
  contract (missing leaves, unaligned sequences, out-of-range indices,
  incomplete features, value coercion) using lightweight stub objects. They do
  not import numpy / the rest of phenomenal, so they run anywhere.
* An integration test that runs the real tracking pipeline on the bundled
  ``test/data/tracking`` time series and exports it. It is skipped if importing
  numpy crashes the interpreter in the current environment (probed in a
  subprocess so a hard crash cannot take the test session down).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from openalea.phenomenal.tracking import export as ex

test_subdir = Path(__file__).parent if "__file__" in globals() else Path(".").resolve()
data_dir = test_subdir.parent / "data" / "tracking"


# ---------------------------------------------------------------------------
# environment probe: importing numpy hard-crashes some broken installs, which a
# normal try/except cannot catch. Probe it in a child process instead.
# ---------------------------------------------------------------------------
def _numpy_imports_cleanly():
    proc = subprocess.run(
        [sys.executable, "-c", "import numpy"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


_NUMPY_OK = _numpy_imports_cleanly()


# ---------------------------------------------------------------------------
# lightweight stubs duck-typing TrackedLeaf / TrackedSnapshot / TrackedPlant
# ---------------------------------------------------------------------------
class _Leaf:
    def __init__(self, features):
        self.features = features


class _Snapshot:
    def __init__(self, leaves, check, sequence, time=None):
        self.leaves = leaves
        self.check_continuity = check
        self.sequence = sequence
        self.time = time


class _Plant:
    def __init__(self, snapshots):
        self.snapshots = snapshots


class _FakeNumpyScalar:
    """Mimics a numpy scalar (``.item()`` + ``.dtype``) without importing numpy."""

    dtype = "float64"

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def _sample_plant():
    """A plant covering every interesting case for the exporter."""

    full = {"mature": True, "azimuth": 91.5, "height": _FakeNumpyScalar(12.0), "length": 30.0}
    growing = {"mature": False, "azimuth": float("nan"), "height": 5.0, "length": 2.0}
    incomplete = {"mature": True, "azimuth": 10.0}  # height + length missing

    return _Plant(
        [
            # rank1 -> leaf0, rank2 -> missing (-1), rank3 -> leaf1
            _Snapshot([_Leaf(full), _Leaf(growing)], True, [0, -1, 1], time=100.0),
            # rank2 points to a non-existent leaf; leaf1/leaf2 stay unranked
            _Snapshot(
                [_Leaf(full), _Leaf(incomplete), _Leaf(growing)],
                False,
                [0, 5],
                time=200.0,
            ),
            # no alignment yet
            _Snapshot([_Leaf(full)], True, [], time=300.0),
        ]
    )


# ---------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------
def test_records_schema_and_serialisable():
    records = ex.tracking_to_records(_sample_plant())
    assert records
    for row in records:
        assert set(row.keys()) == set(ex._FIELDS)
    # every value is JSON-serialisable (no numpy scalars / NaN leaking through)
    json.dumps(records)


def test_requested_fields_present():
    """time, snapshot order, leaf id, rank, mature, length, height, azimuth."""
    for field in (
        "time",
        "snapshot_index",
        "leaf_id",
        "rank",
        "mature",
        "length",
        "height",
        "azimuth",
    ):
        assert field in ex._FIELDS


def test_ranked_leaf_row():
    row = ex.tracking_to_records(_sample_plant())[0]
    assert row["snapshot_index"] == 0
    assert row["time"] == 100.0
    assert row["rank"] == 1  # ranks reported starting at 1
    assert row["leaf_id"] == 0
    assert row["mature"] is True
    assert row["length"] == 30.0
    assert row["check_continuity"] is True
    assert row["missing"] is False
    assert row["error"] is None


def test_numpy_scalar_is_coerced():
    row = ex.tracking_to_records(_sample_plant())[0]
    assert row["height"] == 12.0
    assert isinstance(row["height"], float)
    assert type(row["height"]).__module__ == "builtins"


def test_nan_becomes_none():
    rows = ex.tracking_to_records(_sample_plant())
    growing = next(r for r in rows if r["snapshot_index"] == 0 and r["rank"] == 3)
    assert growing["azimuth"] is None


def test_missing_leaf_at_rank():
    rows = ex.tracking_to_records(_sample_plant())
    missing = next(r for r in rows if r["snapshot_index"] == 0 and r["rank"] == 2)
    assert missing["missing"] is True
    assert missing["leaf_id"] is None
    assert missing["mature"] is None


def test_out_of_range_index_is_flagged_not_raised():
    rows = ex.tracking_to_records(_sample_plant())
    bad = next(r for r in rows if r["snapshot_index"] == 1 and r["rank"] == 2)
    assert bad["missing"] is True
    assert bad["error"] and "out of range" in bad["error"]


def test_incomplete_features_marked():
    rows = ex.tracking_to_records(_sample_plant())
    row = next(
        r
        for r in rows
        if r["snapshot_index"] == 1 and r["rank"] is None and r["leaf_id"] == 1
    )
    assert row["error"] and "length" in row["error"] and "height" in row["error"]
    assert row["mature"] is True  # available value still exported
    assert row["azimuth"] == 10.0


def test_unranked_leaves_included_and_excludable():
    plant = _sample_plant()
    with_unranked = ex.tracking_to_records(plant, include_unranked=True)
    unranked = [r for r in with_unranked if r["rank"] is None and r["error"] is None]
    assert unranked  # detected-but-not-tracked leaves are present

    without = ex.tracking_to_records(plant, include_unranked=False)
    # no plain rank=None data rows remain (only ranked rows, plus error/placeholder)
    assert all(r["rank"] is not None or r["error"] for r in without)


def test_unaligned_snapshot_not_silently_dropped():
    # snapshot 2 has an empty sequence; with unranked excluded it must still
    # yield a placeholder row carrying an explanatory error, not vanish.
    plant = _sample_plant()
    rows = ex.tracking_to_records(plant, include_unranked=False)
    snap2 = [r for r in rows if r["snapshot_index"] == 2]
    assert snap2
    assert any(r["error"] for r in snap2)


def test_missing_time_attribute_defaults_to_none():
    snap = _Snapshot([_Leaf({"mature": True, "azimuth": 1.0, "height": 1.0, "length": 1.0})], True, [0])
    del snap.time
    rows = ex.tracking_to_records(_Plant([snap]))
    assert rows[0]["time"] is None


def test_snapshot_failure_isolated():
    class _BoomSnapshot:
        time = 7.0
        check_continuity = True
        leaves = []

        @property
        def sequence(self):
            raise RuntimeError("boom")

    rows = ex.tracking_to_records(_Plant([_BoomSnapshot()]))
    assert len(rows) == 1
    assert rows[0]["snapshot_index"] == 0
    assert rows[0]["error"] and "boom" in rows[0]["error"]


def test_non_tracked_plant_raises_typeerror():
    with pytest.raises(TypeError):
        ex.tracking_to_records(object())


def test_json_roundtrip_and_file(tmp_path):
    plant = _sample_plant()
    text = ex.tracking_to_json(plant)
    parsed = json.loads(text)
    assert len(parsed) == len(ex.tracking_to_records(plant))

    path = tmp_path / "tracking.json"
    ex.tracking_to_json(plant, path=path)
    assert json.loads(path.read_text(encoding="utf-8")) == parsed


def test_export_dispatch():
    plant = _sample_plant()
    assert ex.export_leaf_tracking(plant, "records") == ex.tracking_to_records(plant)
    assert ex.export_leaf_tracking(plant, "json") == ex.tracking_to_json(plant)
    with pytest.raises(ValueError):
        ex.export_leaf_tracking(plant, "bogus")


@pytest.mark.skipif(not _NUMPY_OK, reason="numpy import crashes in this environment")
def test_dataframe_format():
    plant = _sample_plant()
    try:
        import pandas  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pandas"):
            ex.tracking_to_dataframe(plant)
        return

    df = ex.tracking_to_dataframe(plant)
    assert list(df.columns) == list(ex._FIELDS)
    assert len(df) == len(ex.tracking_to_records(plant))


# ---------------------------------------------------------------------------
# integration test on the bundled time series (runs in CI, skipped if numpy
# cannot be imported in this environment)
# ---------------------------------------------------------------------------
@pytest.fixture
def tracked_plant():
    if not _NUMPY_OK:
        pytest.skip("numpy import crashes in this environment")

    import openalea.phenomenal.object.voxelSegmentation as phm_seg
    from openalea.phenomenal.tracking.phenomenal_coupling import phm_to_phenotrack_input
    from openalea.phenomenal.tracking.trackedPlant import TrackedPlant

    fd = data_dir / "time_series"
    phm_segs, timestamps = [], []
    for filename in os.listdir(fd):
        timestamps.append(int(filename.split(".gz")[0]))
        phm_segs.append(
            phm_seg.VoxelSegmentation.read_from_json_gz(os.path.join(fd, filename))
        )

    phenotrack_segs, _ = phm_to_phenotrack_input(phm_segs, timestamps)
    plant = TrackedPlant.load(phenotrack_segs)
    plant.mature_leaf_tracking()
    plant.growing_leaf_tracking()
    return plant, phenotrack_segs


def test_integration_records_from_real_time_series(tracked_plant):
    plant, phenotrack_segs = tracked_plant

    records = ex.tracking_to_records(plant)
    assert records
    for row in records:
        assert set(row.keys()) == set(ex._FIELDS)
    json.dumps(records)  # serialisable

    # one snapshot per loaded (stem-validated) time point, all times carried out
    expected_times = {float(seg["time"]) for seg in phenotrack_segs}
    got_times = {row["time"] for row in records}
    assert got_times == expected_times

    # there is at least one real ranked mature leaf in the export
    assert any(
        row["rank"] is not None and row["mature"] and row["error"] is None
        for row in records
    )


def test_integration_object_methods_and_formats(tracked_plant):
    plant, _ = tracked_plant

    records = plant.to_records()
    assert records == ex.tracking_to_records(plant)

    parsed = json.loads(plant.to_json())
    assert len(parsed) == len(records)

    try:
        import pandas  # noqa: F401
    except ImportError:
        return
    assert len(plant.to_dataframe()) == len(records)
