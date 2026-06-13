# -*- python -*-
#
#       Copyright INRIA - CIRAD - INRA
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
# ==============================================================================
from __future__ import print_function, absolute_import

import math
import numpy
import scipy.integrate

from .plane_interception import (
    intercept_points_along_path_with_planes,
    max_distance_in_points,
)


# ==============================================================================


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / numpy.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::
    0 and between 2pi

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return numpy.arccos(numpy.clip(numpy.dot(v1_u, v2_u), -1.0, 1.0))


def get_max_distance(node, nodes):
    max_distance = 0
    max_node = node

    for n in nodes:
        distance = abs(numpy.linalg.norm(numpy.array(node) - numpy.array(n)))
        if distance >= max_distance:
            max_distance = distance
            max_node = n

    return max_node, max_distance


def compute_width_organ(closest_nodes):
    width = []
    for nodes in closest_nodes:
        width.append(max_distance_in_points(nodes))

    return width


def compute_curvilinear_abscissa(polyline, length):
    v = 0.0
    curvilinear_abscissa = [0]
    for n1, n2 in zip(polyline, polyline[1:]):
        v += numpy.linalg.norm(numpy.array(n1) - numpy.array(n2))
        curvilinear_abscissa.append(v / float(length))

    return curvilinear_abscissa


def compute_length_organ(polyline):
    length = 0
    for n1, n2 in zip(polyline, polyline[1:]):
        length += numpy.linalg.norm(numpy.array(n1) - numpy.array(n2))

    return length


def compute_inclination_angle(polyline, step=1):
    if not len(polyline) > step:
        return None

    length = 0
    for (x0, y0, z0), (x1, y1, z1) in zip(polyline[::step], polyline[step::step]):
        length += numpy.linalg.norm(numpy.array((x1 - x0, y1 - y0, z1 - z0)))

    angles = []
    z_axis = numpy.array([0, 0, 1])

    for (x0, y0, z0), (x1, y1, z1) in zip(polyline[::step], polyline[step::step]):
        vector = numpy.array((x1 - x0, y1 - y0, z1 - z0))
        norm = numpy.linalg.norm(vector)
        angle = angle_between(z_axis, vector)
        angles.append(math.degrees(angle) * (norm / length))

    inclination_angle = sum(angles) / float(len(angles))

    if inclination_angle > 180.0:
        inclination_angle -= 360.0
    return inclination_angle


def compute_fitted_width(width, curvilinear_abscissa):
    x = numpy.array(curvilinear_abscissa)
    XX = numpy.vstack((x**2, x)).T
    p_all = numpy.linalg.lstsq(XX, width[::-1], rcond=None)[0]
    fitted_width = numpy.dot(p_all, XX.T)

    return fitted_width


def compute_vector_mean(polyline):
    x, y, z = polyline[0]

    vectors = []
    for i in range(1, len(polyline)):
        xx, yy, zz = polyline[i]

        v = (xx - x, yy - y, zz - z)
        vectors.append(v)

    vector_mean = numpy.array(vectors).mean(axis=0)

    return vector_mean


def compute_azimuth_angle(polyline):
    vector_mean = compute_vector_mean(polyline)
    x, y, _ = vector_mean
    azimuth_angle = math.degrees(math.atan2(y, x))

    return azimuth_angle, tuple(vector_mean)


def compute_insertion_angle(polyline, stem_vector_mean):
    x, y, z = polyline[0]

    vectors = []
    for i in range(1, len(polyline) // 4 + 1):
        xx, yy, zz = polyline[i]
        vectors.append((xx - x, yy - y, zz - z))

    insertion_vector = numpy.array(vectors).mean(axis=0)
    insertion_angle = angle_between(insertion_vector, stem_vector_mean)
    insertion_angle = math.degrees(insertion_angle)

    if insertion_angle > 180.0:
        insertion_angle -= 360.0

    return insertion_angle, tuple(insertion_vector)


# ==============================================================================
# ==============================================================================
# ==============================================================================


def voxel_base_height(vo, polyline, min_distance=30):
    """
    Search voxels that have a distance < min_distance to the polyline, and return the height of the lowest one.
    This can be used to determine the insertion height of a maize mature leaf (with vo = mature leaf organ,
    polyline = stem polyline), based on voxel data (since it's often more accurate than polyline data)
    """

    # all voxels
    vxs = numpy.array(list(vo.voxels_position()))

    # only voxels with distance to polyline the lowest point < min_distance
    vxs2 = []
    for x, y, z in vxs:
        # TODO : approx ?
        # TODO : param min_distance depending on leaf length ?
        x_stem, y_stem, _ = polyline[numpy.argmin(numpy.abs(polyline[:, 2] - z))]
        if numpy.sqrt((x_stem - x) ** 2 + (y_stem - y) ** 2) < min_distance:
            vxs2.append([x, y, z])
    vxs2 = numpy.array(vxs2)

    if vxs2.size == 0:
        height = vo.info["pm_position_base"]
    else:
        height = vxs2[numpy.argsort(vxs2[:, 2])][0]

    return height


def organ_analysis(organ, polyline, closest_nodes, stem_vector_mean=None):
    if len(polyline) <= 1:
        return None

    organ.info["pm_position_tip"] = tuple(polyline[-1])
    organ.info["pm_position_base"] = tuple(polyline[0])
    organ.info["pm_z_tip"] = polyline[-1][2]
    organ.info["pm_z_base"] = polyline[0][2]

    length = compute_length_organ(polyline)
    organ.info["pm_length"] = length
    normalized_curvilinear_abscissa = compute_curvilinear_abscissa(polyline, length)
    curvilinear_abscissa = compute_curvilinear_abscissa(polyline, 1)
    # Compute width
    width = compute_width_organ(closest_nodes)

    organ.info["pm_width_max"] = max(width)
    organ.info["pm_width_mean"] = sum(width) / float(len(width))
    organ.info["pm_surface"] = scipy.integrate.simpson(y=width, x=curvilinear_abscissa)

    fitted_width = compute_fitted_width(width, normalized_curvilinear_abscissa)
    organ.info["pm_fitted_width_max"] = max(fitted_width)
    organ.info["pm_fitted_width_mean"] = sum(fitted_width) / float(len(fitted_width))
    organ.info["pm_fitted_surface"] = scipy.integrate.simpson(
        y=fitted_width, x=curvilinear_abscissa
    )
    # Compute azimuth
    azimuth_angle, vector_mean = compute_azimuth_angle(polyline)
    organ.info["pm_vector_mean"] = vector_mean
    organ.info["pm_azimuth_angle"] = azimuth_angle

    inclination_angle = compute_inclination_angle(polyline)
    organ.info["pm_inclination_angle"] = inclination_angle

    if stem_vector_mean is not None and len(polyline) >= 4:
        insertion_angle, vector = compute_insertion_angle(polyline, stem_vector_mean)

        organ.info["pm_insertion_angle_vector"] = vector
        organ.info["pm_insertion_angle"] = insertion_angle

    return organ


def maize_stem_analysis(vo, voxels_size, distance_plane=0.75):
    voxels_position = vo.voxels_position()
    polyline = vo.get_longest_segment().polyline

    if len(polyline) <= 1:
        return vo

    # ==========================================================================
    # Compute height of the leaf

    closest_nodes, _ = intercept_points_along_path_with_planes(
        numpy.array(list(voxels_position)),
        polyline,
        distance_from_plane=distance_plane * voxels_size,
        without_connection=True,
        voxels_size=voxels_size,
    )

    # ==========================================================================

    return organ_analysis(vo, polyline, closest_nodes)


def maize_mature_leaf_analysis(vo, voxels_size, stem_vector_mean, distance_plane=0.75):
    voxels_position = vo.voxels_position()
    polyline = vo.get_longest_segment().polyline

    if len(polyline) <= 3:
        return None

    vo.info["pm_full_length"] = compute_length_organ(polyline)

    # ==========================================================================
    # Compute height of the leaf

    closest_nodes, _ = intercept_points_along_path_with_planes(
        numpy.array(list(voxels_position)),
        polyline,
        distance_from_plane=distance_plane * voxels_size,
        voxels_size=voxels_size,
    )

    # ==========================================================================
    # Compute extremity
    index_position_base = vo.get_real_index_position_base()

    # ==========================================================================

    real_polyline = polyline[index_position_base:]
    real_closest_nodes = closest_nodes[index_position_base:]

    vo = organ_analysis(vo, real_polyline, real_closest_nodes, stem_vector_mean)

    return vo


def maize_growing_leaf_analysis_real_length(maize_segmented, vo):
    voxels = maize_segmented.get_voxels_position(except_organs=[vo])
    longest_polyline = vo.get_longest_segment().polyline
    shared = set(voxels).intersection(longest_polyline)

    if not shared:
        # This growing leaf does not share a single voxel with the rest of the
        # plant (an isolated / degenerate topology). ``numpy.max`` on an empty
        # array would raise, so we fall back to the highest node of its own
        # polyline. The returned value is only used to order growing leaves, so
        # this keeps the ordering well defined instead of crashing the whole
        # analysis.
        points = numpy.array(list(longest_polyline))
    else:
        points = numpy.array(list(shared))

    z = numpy.max(points[:, 2])
    return z


# Minimum number of polyline nodes a growing-leaf blade must keep for
# ``organ_analysis`` to produce meaningful metrics (insertion angle needs >= 4).
_MIN_GROWING_LEAF_BLADE_NODES = 4


def growing_leaf_real_index_position_base(polyline, voxels):
    """Locate the boundary between a growing leaf pseudo-stem and its blade.

    The *pseudo-stem* of a growing leaf is the part of its polyline that is
    still bundled with the stem (or with leaves analysed earlier). Those shared
    voxels form a run anchored at the *base* of the polyline (``polyline[0]``);
    the emerged blade is everything past that run.

    The previous implementation simply kept the highest polyline index that
    intersected ``voxels``. When two leaves touch each other by their tips, the
    current leaf shares a voxel near its *tip* with another organ, so that
    highest index jumped close to the end of the polyline, the blade collapsed
    to a single node and the reported leaf length became ``0``. To avoid this we
    only trust the *leading contiguous run* of shared nodes, mirroring
    :meth:`VoxelOrgan.get_real_index_position_base` which applies the same
    contiguity rule from the tip side.

    Parameters
    ----------
    polyline : list
        Ordered polyline nodes. ``polyline[0]`` is the base, ``polyline[-1]``
        the tip.
    voxels : iterable
        Voxel positions shared with the stem and the already-processed organs.

    Returns
    -------
    index_position_base : int
        Index of the last node of the leading contiguous pseudo-stem run.
    diagnostic : dict
        Information about how the boundary was found. See
        :func:`maize_growing_leaf_analysis`.
    """
    shared = set(polyline).intersection(set(voxels))

    # Walk from the base and keep only the contiguous run of shared nodes. Stop
    # at the first node that is not shared: anything shared *after* a gap is a
    # connection artifact (typically two leaf tips touching) and must not be
    # mistaken for pseudo-stem.
    index_position_base = 0
    for i, node in enumerate(polyline):
        if node in shared:
            index_position_base = i
        else:
            break

    # A shared node located after the leading run means the leaf is connected to
    # another organ somewhere above its base (tip-to-tip being the usual case).
    # This is exactly what the old code latched onto to zero the length, so we
    # surface it as a diagnostic instead of silently mis-trimming.
    tip_connection_detected = any(
        node in shared for node in polyline[index_position_base + 1 :]
    )

    diagnostic = {
        "method": "contiguous_base_run",
        "index_position_base": index_position_base,
        "polyline_length_nodes": len(polyline),
        "n_shared_nodes": len(shared),
        "tip_connection_detected": bool(tip_connection_detected),
    }

    return index_position_base, diagnostic


def maize_growing_leaf_analysis(
    vo, voxels_size, stem_vector_mean, voxels, distance_plane=0.75
):
    voxels_position = vo.voxels_position()
    polyline = vo.get_longest_segment().polyline
    vo.info["pm_full_length"] = compute_length_organ(polyline)

    if len(polyline) <= 1:
        return vo

    real_longest_polyline = vo.real_longest_polyline()
    vo.info["pm_length_with_speudo_stem"] = compute_length_organ(real_longest_polyline)

    # ==========================================================================
    # Compute height of the leaf
    closest_nodes, _ = intercept_points_along_path_with_planes(
        numpy.array(list(voxels_position)),
        polyline,
        distance_from_plane=distance_plane * voxels_size,
        voxels_size=voxels_size,
    )

    # ==========================================================================
    # Compute extremity (boundary between pseudo-stem and emerged blade).
    # Only the leading contiguous shared run is treated as pseudo-stem so that a
    # tip connection with another organ can no longer collapse the blade.
    index_position_base, length_diagnostic = growing_leaf_real_index_position_base(
        polyline, voxels
    )

    # Degenerate-topology guard: if trimming the pseudo-stem would leave too few
    # nodes to measure (e.g. the leaf is almost entirely bundled, or its base
    # run reaches the tip), keep the whole polyline rather than reporting a 0 /
    # ``None`` length. The (small) pseudo-stem over-count is recorded in the
    # diagnostic and the pseudo-stem length is reported separately anyway.
    fallback_used = False
    fallback_reason = None
    if (len(polyline) - index_position_base) < _MIN_GROWING_LEAF_BLADE_NODES:
        fallback_used = True
        fallback_reason = "blade_too_short_after_trim"
        index_position_base = 0

    real_polyline = polyline[index_position_base:]
    real_closest_nodes = closest_nodes[index_position_base:]

    vo = organ_analysis(vo, real_polyline, real_closest_nodes, stem_vector_mean)

    length_diagnostic["fallback_used"] = fallback_used
    length_diagnostic["fallback_reason"] = fallback_reason

    if vo is not None:
        length_diagnostic["real_length"] = vo.info.get("pm_length")
        length_diagnostic["full_length"] = vo.info.get("pm_full_length")
        vo.info["pm_length_diagnostic"] = length_diagnostic

        # Clamp to >= 0: in a fallback the blade length can momentarily exceed
        # ``pm_length_with_speudo_stem`` (computed from a slightly shorter
        # polyline), which would otherwise yield a nonsensical negative
        # pseudo-stem length and bias plant statistics.
        vo.info["pm_length_speudo_stem"] = max(
            0.0,
            vo.info["pm_length_with_speudo_stem"] - vo.info["pm_length"],
        )

    return vo


def maize_analysis(maize_segmented):
    """Update info field of the VoxelSegmentation object with the analysis
    result computed. Each organ is a specific algorithm to extract information.

    Parameters
    ----------
    maize_segmented : VoxelSegmentation

    Returns
    -------
    maize_segmented: VoxelSegmentation
    """

    voxels_size = maize_segmented.voxels_size
    for vo in maize_segmented.voxel_organs:
        vo.info = {
            "pm_label": vo.label,
            "pm_sub_label": vo.sub_label,
            "pm_voxels_volume": (
                len(vo.voxels_position()) * maize_segmented.voxels_size**3
            ),
        }

    # ==========================================================================

    vo_stem = maize_segmented.get_stem()
    vo_stem = maize_stem_analysis(vo_stem, voxels_size)

    # ==========================================================================

    mature_leafs = []
    stem_polyline = numpy.array(
        list(maize_segmented.get_stem().get_highest_polyline().polyline)
    )
    for vo_mature_leaf in maize_segmented.get_mature_leafs():
        vo_mature_leaf = maize_mature_leaf_analysis(
            vo_mature_leaf, voxels_size, vo_stem.info["pm_vector_mean"]
        )

        if vo_mature_leaf is None:
            continue

        if "pm_z_base" in vo_mature_leaf.info:
            vo_mature_leaf.info["pm_z_base_voxel"] = voxel_base_height(
                vo_mature_leaf, stem_polyline
            )[2]

        mature_leafs.append((vo_mature_leaf, vo_mature_leaf.info["pm_z_base"]))
    mature_leafs.sort(key=lambda x: x[1])

    # ==========================================================================

    growing_leafs = []
    for vo_growing_leaf in maize_segmented.get_growing_leafs():
        z = maize_growing_leaf_analysis_real_length(maize_segmented, vo_growing_leaf)
        growing_leafs.append((vo_growing_leaf, z))
    growing_leafs.sort(key=lambda x: x[1])

    voxels = vo_stem.voxels_position()
    for vo, _ in growing_leafs:
        # Tip connections (two leaves touching by their tips, a leaf tip folding
        # back onto the stem, ...) used to collapse the blade and report a 0
        # length here. maize_growing_leaf_analysis now only trims the leading
        # contiguous pseudo-stem run and records a diagnostic in
        # ``vo.info["pm_length_diagnostic"]`` (see
        # growing_leaf_real_index_position_base).
        vo = maize_growing_leaf_analysis(
            vo, voxels_size, vo_stem.info["pm_vector_mean"], voxels
        )

        if vo is None:
            continue

        voxels = voxels.union(vo.voxels_position())

    # ==========================================================================

    for leaf_number, (vo, _) in enumerate(mature_leafs + growing_leafs):
        vo.info["pm_leaf_number"] = leaf_number + 1

    maize_segmented.update_plant_info()

    return maize_segmented
