# -*- python -*-
#
#       Copyright INRIA - CIRAD - INRA
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
# ==============================================================================
"""
Scoring stage of the alignment pipeline.

This module isolates *how dissimilar two things are* from *how the alignment
uses that dissimilarity* (see :mod:`alignment`). Two levels are exposed:

* :func:`vector_distance` -- the leaf-feature distance between two feature
  vectors. This is the swappable scoring strategy; pass a different callable
  with the same ``(vec1, vec2) -> float`` signature to experiment with another
  metric without touching the alignment code.
* :func:`profile_distance` -- the dissimilarity between two *profiles* (a
  profile is a stack of aligned vectors, possibly containing gaps). It averages
  :func:`vector_distance` over every comparable (non-gap) pair.

The gap / extremity penalty strategy lives in :class:`config.GapPenalty`; use
:func:`gap_policy` as a convenience factory from the raw ``gap`` /
``gap_extremity_factor`` numbers used by the alignment functions.
"""

from __future__ import annotations

import numpy as np

from openalea.phenomenal.tracking.config import GapPenalty


def is_gap(vector) -> bool:
    """Return True if ``vector`` is a gap, i.e. filled with NaN."""
    return bool(np.all(np.isnan(vector)))


def vector_distance(vec1, vec2):
    """Leaf-feature distance between two vectors of the same length.

    Default scoring strategy: the euclidean distance. This is the single place
    defining how two leaf feature vectors are compared.

    Parameters
    ----------
    vec1 : 1D array
    vec2 : 1D array, of same length as vec1

    Returns
    -------
    float
    """
    return np.linalg.norm(vec1 - vec2)


def profile_distance(x, y, empty_score, distance=vector_distance):
    """Dissimilarity between two profiles ``x`` and ``y``.

    A profile is a 2D array ``(number of vectors, vector length)`` whose rows may
    be gaps (NaN). The score is the mean :func:`distance` over every pair of
    non-gap rows.

    When no comparable pair exists (every pairing involves a gap) there is no
    meaningful distance to compute; ``empty_score`` is returned instead. This
    makes explicit what used to be an unnamed fallback constant inside the
    Needleman-Wunsch loop: the alignment passes the *extremity gap penalty* so
    that an all-gap column is treated like a terminal gap.

    Parameters
    ----------
    x : 2D array
        size (number of vectors, vector length)
    y : 2D array
        size (number of vectors, vector length)
    empty_score : float
        value returned when ``x`` and ``y`` share no comparable (non-gap) pair.
    distance : callable, optional
        vector-to-vector scoring strategy, defaults to :func:`vector_distance`.

    Returns
    -------
    float
    """
    scores = [
        distance(xvec, yvec)
        for xvec in x
        for yvec in y
        if not is_gap(xvec) and not is_gap(yvec)
    ]

    if scores:
        return np.mean(scores)
    return empty_score


def gap_policy(gap, gap_extremity_factor=1.0):
    """Build the :class:`config.GapPenalty` strategy from raw parameters.

    Parameters
    ----------
    gap : float
        interior gap penalty.
    gap_extremity_factor : float
        terminal-gap penalty is ``gap * gap_extremity_factor``.

    Returns
    -------
    config.GapPenalty
    """
    return GapPenalty(gap=gap, extremity_factor=gap_extremity_factor)
