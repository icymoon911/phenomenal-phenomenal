# -*- python -*-
#
#       Copyright INRIA - CIRAD - INRA
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
# ==============================================================================
"""Regression tests for the growing-leaf length computation.

Historically, when two leaves touched each other by their tips (or a leaf tip
folded back onto the stem) the growing-leaf blade collapsed to a single node and
the reported length became ``0``, which then propagated to leaf ordering and the
whole-plant statistics. These tests pin the fixed behaviour with a minimal,
deterministic case instead of relying on the random voxel grid used elsewhere.
"""
import pytest

from openalea.phenomenal.object.voxelOrgan import VoxelOrgan
from openalea.phenomenal.object.voxelSegmentation import VoxelSegmentation
from openalea.phenomenal.segmentation.maize_analysis import (
    growing_leaf_real_index_position_base,
    maize_growing_leaf_analysis,
    maize_growing_leaf_analysis_real_length,
)


# ==============================================================================
# Helpers
# ==============================================================================


def _vertical_leaf(n_nodes, voxels_size=1):
    """Build a minimal straight, vertical growing leaf.

    The polyline runs along +z with one node per voxel layer (``voxels_size``
    spacing). Around it we add a "plus" shaped voxel cross-section so that the
    plane interception in ``maize_growing_leaf_analysis`` returns a well-formed,
    non-empty set of closest nodes for every polyline node.
    """
    polyline = [(0, 0, z * voxels_size) for z in range(n_nodes)]

    voxels = set()
    for x, y, z in polyline:
        voxels.add((x, y, z))
        voxels.add((x + voxels_size, y, z))
        voxels.add((x - voxels_size, y, z))
        voxels.add((x, y + voxels_size, z))
        voxels.add((x, y - voxels_size, z))

    vo = VoxelOrgan(label="growing_leaf")
    vo.add_voxel_segment(voxels, polyline)
    vo.info = {}
    return vo, polyline


# ==============================================================================
# Unit tests on the pseudo-stem / blade boundary
# ==============================================================================


def test_growing_leaf_index_ignores_tip_connection():
    """A tip connection must not move the pseudo-stem boundary."""
    polyline = [(0, 0, z) for z in range(12)]
    base_shared = {(0, 0, 0), (0, 0, 1), (0, 0, 2)}
    # Simulate two leaf tips touching: a shared voxel near the tip.
    tip_shared = base_shared | {(0, 0, 11)}

    index, diagnostic = growing_leaf_real_index_position_base(polyline, tip_shared)

    # Only the leading contiguous run (the real pseudo-stem) is trusted.
    assert index == 2
    assert diagnostic["tip_connection_detected"] is True

    # Document the contrast with the previous, buggy rule (highest shared
    # index): it jumped to the tip and left a single-node blade -> length 0.
    old_index = max(i for i, node in enumerate(polyline) if node in tip_shared)
    assert old_index == 11
    assert len(polyline[old_index:]) == 1


def test_growing_leaf_index_not_biased_without_tip_connection():
    """A normal leaf (shared voxels only at the base) is unaffected by the fix."""
    polyline = [(0, 0, z) for z in range(12)]
    base_shared = {(0, 0, 0), (0, 0, 1), (0, 0, 2)}

    index, diagnostic = growing_leaf_real_index_position_base(polyline, base_shared)

    assert index == 2
    assert diagnostic["tip_connection_detected"] is False


def test_growing_leaf_index_fully_bundled():
    """When the whole polyline is shared, the boundary reaches the last node.

    The analysis-level fallback (exercised in the integration tests) then turns
    this degenerate case into a full-polyline measurement rather than a 0 / None
    length.
    """
    polyline = [(0, 0, z) for z in range(5)]
    shared = set(polyline)

    index, diagnostic = growing_leaf_real_index_position_base(polyline, shared)

    assert index == len(polyline) - 1
    assert diagnostic["tip_connection_detected"] is False


# ==============================================================================
# Integration tests on maize_growing_leaf_analysis
# ==============================================================================


def test_maize_growing_leaf_length_survives_tip_connection():
    """End-to-end: a tip-connected growing leaf gets a real, non-zero length."""
    voxels_size = 1
    stem_vector_mean = (0.0, 0.0, 1.0)
    n_nodes = 12

    vo, _ = _vertical_leaf(n_nodes, voxels_size)
    base = {(0, 0, 0), (0, 0, 1), (0, 0, 2)}
    connected = base | {(0, 0, n_nodes - 1)}  # tip touches another organ

    vo = maize_growing_leaf_analysis(vo, voxels_size, stem_vector_mean, connected)

    assert vo is not None
    assert vo.info["pm_length"] > 0
    # Blade = nodes from the end of the base run (index 2) up to the tip.
    assert vo.info["pm_length"] == pytest.approx(float(n_nodes - 3))

    diagnostic = vo.info["pm_length_diagnostic"]
    assert diagnostic["tip_connection_detected"] is True
    assert diagnostic["fallback_used"] is False
    assert diagnostic["real_length"] == pytest.approx(vo.info["pm_length"])
    # Pseudo-stem length stays consistent and non-negative.
    assert vo.info["pm_length_speudo_stem"] >= 0


def test_maize_growing_leaf_length_identical_with_or_without_tip_connection():
    """The fix must not bias the length of a leaf that has no tip connection."""
    voxels_size = 1
    stem_vector_mean = (0.0, 0.0, 1.0)
    n_nodes = 12
    base = {(0, 0, 0), (0, 0, 1), (0, 0, 2)}

    vo_plain, _ = _vertical_leaf(n_nodes, voxels_size)
    vo_connected, _ = _vertical_leaf(n_nodes, voxels_size)

    plain = maize_growing_leaf_analysis(
        vo_plain, voxels_size, stem_vector_mean, base
    )
    connected = maize_growing_leaf_analysis(
        vo_connected, voxels_size, stem_vector_mean, base | {(0, 0, n_nodes - 1)}
    )

    assert plain.info["pm_length"] > 0
    assert plain.info["pm_length"] == pytest.approx(connected.info["pm_length"])


def test_maize_growing_leaf_real_length_handles_no_shared_voxel():
    """An isolated growing leaf must not crash the ordering computation."""
    voxels_size = 1
    vms = VoxelSegmentation(voxels_size)
    vo, _ = _vertical_leaf(6, voxels_size)
    vms.voxel_organs.append(vo)

    # vo is the only organ: nothing is shared with the rest of the plant. The
    # old code did numpy.max over an empty array and raised; now it falls back
    # to the leaf's own highest node.
    z = maize_growing_leaf_analysis_real_length(vms, vo)
    assert z == 5


def test_maize_growing_leaf_fallback_when_fully_bundled():
    """A fully-bundled leaf degrades to a full-polyline length, never 0."""
    voxels_size = 1
    stem_vector_mean = (0.0, 0.0, 1.0)
    n_nodes = 12

    vo, polyline = _vertical_leaf(n_nodes, voxels_size)
    # Every spine node is shared: trimming the pseudo-stem would empty the blade.
    fully_shared = set(polyline)

    vo = maize_growing_leaf_analysis(vo, voxels_size, stem_vector_mean, fully_shared)

    assert vo is not None
    assert vo.info["pm_length"] > 0
    diagnostic = vo.info["pm_length_diagnostic"]
    assert diagnostic["fallback_used"] is True
    assert diagnostic["fallback_reason"] == "blade_too_short_after_trim"
