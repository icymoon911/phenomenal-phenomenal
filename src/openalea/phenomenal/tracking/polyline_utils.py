"""Geometry helpers operating on 3D polylines (n x 3 arrays).

Pure, stateless utilities shared by the scoring and post-processing stages. No
tunable lives here; behaviour depends only on the polyline geometry.
"""

import numpy as np


def polyline_length(pl):
    """Total curvilinear length of a polyline (sum of segment lengths)."""
    return np.sum(
        [
            np.linalg.norm(np.array(pl[k]) - np.array(pl[k + 1]))
            for k in range(len(pl) - 1)
        ]
    )


def polyline_quantile_coordinate(pl, q):
    """Coordinate of the point located at curvilinear quantile ``q`` in [0, 1].

    ``q = 0`` returns the first point, ``q = 1`` the last; intermediate values
    are linearly interpolated along the polyline by arc length.
    """
    pl = np.array(pl)
    d = np.diff(pl, axis=0)
    segdists = np.sqrt((d**2).sum(axis=1))
    s = np.cumsum(segdists) / np.sum(segdists)
    s = np.concatenate((np.array([0]), s))

    try:
        i_q = next(i for i, val in enumerate(s) if val >= q)
    except StopIteration:
        i_q = len(s) - 1

    a, b = pl[i_q - 1], pl[i_q]
    q_pl = a + (b - a) * ((q - s[i_q - 1]) / (s[i_q] - s[i_q - 1]))

    return q_pl


def polyline_until_z(pl, z):
    """Return the polyline section located above height ``z``.

    The cut is done at vertex granularity: the returned section starts at the
    first vertex whose ``z`` coordinate is strictly greater than ``z`` (it does
    not interpolate a vertex exactly at height ``z``). If the whole polyline is
    at or below ``z``, it is returned unchanged.
    """
    if np.max(np.array(pl)[:, 2]) <= z:
        i = 0
    else:
        i = next((i for i, pos in enumerate(pl) if pos[2] > z))
    return pl[i:]


def polyline_simplification(pl, n):
    """Resample a polyline to ``n`` points evenly spaced by arc length.

    Polylines with fewer than ``n`` points are returned unchanged (as an array).
    """
    if len(pl) < n:
        return np.array(pl)
    return np.array([polyline_quantile_coordinate(pl, q) for q in np.linspace(0, 1, n)])
