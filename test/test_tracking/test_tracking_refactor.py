import numpy as np

from openalea.phenomenal.tracking.alignment import align_with_trace, multi_alignment
from openalea.phenomenal.tracking.alignment_postprocessing import (
    detect_abnormal_ranks,
    leaf_polylines_distance,
)
from openalea.phenomenal.tracking.config import DEFAULT_CONFIG
from openalea.phenomenal.tracking.scoring import gap_policy
from openalea.phenomenal.tracking.trackedPlant import (
    TrackedLeaf,
    TrackedPlant,
    TrackedSnapshot,
    check_time_intervals,
)


def make_leaf(mature, azimuth, height, length, polyline):
    return TrackedLeaf(
        polyline=np.array(polyline, dtype=float),
        features={
            "mature": mature,
            "azimuth": azimuth,
            "height": height,
            "length": length,
        },
    )


def straight_polyline(base_height, length, azimuth):
    azimuth_rad = np.deg2rad(azimuth)
    tip = [
        length * np.cos(azimuth_rad),
        length * np.sin(azimuth_rad),
        base_height + length,
    ]
    return [
        [0.0, 0.0, base_height],
        [tip[0] / 2.0, tip[1] / 2.0, base_height + length / 2.0],
        tip,
    ]


def synthetic_tracked_plant():
    snapshots = []
    base_ranks = [
        (10.0, 40.0, 0.0),
        (30.0, 50.0, 90.0),
        (55.0, 60.0, 200.0),
        (85.0, 65.0, 300.0),
    ]
    for time_index in range(6):
        leaves = []
        mature_count = min(len(base_ranks), 1 + time_index)
        for rank_index in range(mature_count):
            height, length, azimuth = base_ranks[rank_index]
            shifted_height = height + 0.4 * time_index
            shifted_length = length + 0.2 * time_index
            shifted_azimuth = (azimuth + 1.5 * time_index) % 360
            leaves.append(
                make_leaf(
                    True,
                    shifted_azimuth,
                    shifted_height,
                    shifted_length,
                    straight_polyline(
                        shifted_height,
                        shifted_length,
                        shifted_azimuth,
                    ),
                )
            )

        top_height = base_ranks[min(mature_count, len(base_ranks) - 1)][0] + 0.4 * time_index
        leaves.append(
            make_leaf(
                False,
                0.0,
                top_height,
                15.0,
                straight_polyline(top_height, 15.0 + time_index, (20 * time_index) % 360),
            )
        )
        snapshots.append(TrackedSnapshot(leaves, check=True))

    return TrackedPlant(snapshots=snapshots)


def test_align_with_trace_matches_multi_alignment():
    sequences = [
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        np.array([[0.0, 0.0], [1.2, 1.2], [2.3, 2.3]]),
        np.array([[0.0, 0.0], [2.1, 2.1]]),
    ]

    traced_matrix, steps = align_with_trace(sequences, gap=1.0, align_range=2)
    plain_matrix = multi_alignment(sequences, gap=1.0, align_range=2)

    assert np.array_equal(traced_matrix, plain_matrix)
    assert len(steps) == len(sequences) - 1
    assert steps[0]["step"] == 1
    assert set(steps[0]) == {"step", "ref_indexes", "new_index", "rx", "ry"}


def test_config_defaults_and_gap_policy_are_consistent():
    policy = gap_policy(
        DEFAULT_CONFIG.alignment.gap,
        DEFAULT_CONFIG.alignment.gap_extremity_factor,
    )

    assert policy.gap == DEFAULT_CONFIG.alignment.gap
    assert policy.extremity == (
        DEFAULT_CONFIG.alignment.gap * DEFAULT_CONFIG.alignment.gap_extremity_factor
    )
    assert DEFAULT_CONFIG.post_processing.abnormal_min_relative_count == 0.5
    assert DEFAULT_CONFIG.ref_skeleton.nmax == 15
    assert DEFAULT_CONFIG.continuity.discontinuity == 5.0


def test_postprocessing_threshold_and_polyline_distance_are_customizable():
    alignment_matrix = np.array(
        [
            [0, 1, 2, 3, 4],
            [0, -1, 1, 2, 3],
            [0, -1, 1, 2, 3],
            [0, -1, 1, -1, 2],
        ]
    )

    assert detect_abnormal_ranks(alignment_matrix) == [1]
    assert detect_abnormal_ranks(alignment_matrix, min_relative_count=0.25) == []

    polyline_ref = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 20.0]])
    polyline_candidate = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 10.0], [1.0, 1.0, 22.0]])

    distance_default = leaf_polylines_distance(polyline_ref, polyline_candidate)
    distance_low_sample = leaf_polylines_distance(polyline_ref, polyline_candidate, n=5)

    assert distance_default > 0
    assert distance_low_sample > 0


def test_tracked_plant_exposes_debug_steps_and_reference_skeleton():
    tracked_plant = synthetic_tracked_plant()
    tracked_plant.mature_leaf_tracking()

    assert tracked_plant.debug is not None
    assert len(tracked_plant.debug.steps) == len(tracked_plant.snapshots) - 1

    reference = tracked_plant.get_ref_skeleton()
    assert sorted(reference.keys()) == [1, 2, 3, 4]

    tracked_plant.growing_leaf_tracking()
    ranks, _, checks = tracked_plant.output()

    assert len(ranks) == len(tracked_plant.snapshots)
    assert all(checks)
    assert all(len(snapshot.sequence) == len(tracked_plant.snapshots[0].sequence) for snapshot in tracked_plant.snapshots)


def test_check_time_intervals_keeps_historical_default_behavior():
    result = check_time_intervals([0, 1, 2, 3, 30, 31])
    assert list(map(bool, result)) == [True, True, True, True, False, False]
