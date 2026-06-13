"""
Post-processing stage, applied *after* the alignment of sequences of ligulated
leaves.

It contains the two steps that run once the raw alignment matrix is available:

- Remove abnormal columns in the alignment matrix (i.e. ranks corresponding to
  artefacts, not to real leaves).
  (:func:`detect_abnormal_ranks`)

- Backwards tracking of each leaf in its growth phase until its emergence, based
  on a polyline distance.
  (:func:`leaf_polylines_distance`)

The thresholds that used to be inline constants are now keyword arguments whose
defaults are sourced from :data:`config.DEFAULT_CONFIG`.
"""

import numpy as np

from openalea.phenomenal.tracking.config import DEFAULT_CONFIG
from openalea.phenomenal.tracking.polyline_utils import (
    polyline_quantile_coordinate,
    polyline_length,
)


def detect_abnormal_ranks(
    alignment_matrix,
    min_relative_count=DEFAULT_CONFIG.post_processing.abnormal_min_relative_count,
):
    """
    Specific to plant alignment.
    Detect abnormal columns in 'alignment_matrix' object resulting from multi
    alignment based on the following criteria:
    - A column is abnormal if it contains fewer aligned vectors than
    ``min_relative_count`` times the average of its two neighbouring columns
    (value != -1 in 'alignment_matrix'). With the default 0.5, that means a
    column holding less than half as many leaves as its neighbours.
    - first and last columns can't be abnormal

    Parameters
    ----------
    alignment_matrix : 2D array
        result of multi_alignment() function
    min_relative_count : float
        fraction of the neighbouring columns' average count below which a column
        is flagged abnormal (default sourced from config).

    Returns
    -------
    list(int)
        indexes of the abnormal columns / ranks.
    """

    alignment_matrix = np.array(alignment_matrix)
    counts = [
        len([k for k in alignment_matrix[:, i] if k != -1])
        for i in range(alignment_matrix.shape[1])
    ]
    abnormal_ranks = []
    for i, value in enumerate(counts):
        if 0 < i < len(counts) - 1 and value < min_relative_count * np.mean(
            [counts[i - 1], counts[i + 1]]
        ):
            abnormal_ranks.append(i)

    return abnormal_ranks


def leaf_polylines_distance(
    polyline_ref,
    polyline_candidate,
    n=DEFAULT_CONFIG.post_processing.polyline_distance_samples,
    eps=DEFAULT_CONFIG.post_processing.polyline_distance_eps,
):
    """
    Computes the distance between two leaf polylines.

    The two polylines are sampled at ``n`` evenly spaced curvilinear positions;
    the distance is the sum of point-to-point euclidean distances, normalised by
    the reference polyline length (floored at ``eps`` to avoid division issues).

    Parameters
    ----------
    polyline_ref : array
    polyline_candidate : array
    n : int
        number of curvilinear samples (default sourced from config).
    eps : float
        numerical floor for the normalising length (default sourced from config).

    Returns
    -------
    float
    """

    # computing distance
    dist = 0
    for q in np.linspace(0, 1, n):
        pos1 = polyline_quantile_coordinate(polyline_ref, q)
        pos2 = polyline_quantile_coordinate(polyline_candidate, q)
        dist += np.sqrt(np.sum((pos1 - pos2) ** 2))

    # scale standardization
    dist_rescaled = dist / np.max((polyline_length(polyline_ref), eps))

    return dist_rescaled
