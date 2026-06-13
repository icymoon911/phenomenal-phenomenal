"""Multiple sequence alignment.

This module owns the *alignment* stage only: turning a dissimilarity (provided
by :mod:`scoring`) and a gap policy (:class:`config.GapPenalty`) into an
alignment of sequences. The scoring details and the magic numbers live
elsewhere now:

* leaf-feature / profile distances -> :mod:`openalea.phenomenal.tracking.scoring`
* gap and extremity penalties      -> :class:`openalea.phenomenal.tracking.config.GapPenalty`

Public API (unchanged signatures and behaviour):
:func:`needleman_wunsch`, :func:`multi_alignment`, :func:`scoring_function`,
:func:`alignment_score`, :func:`insert_gaps`.

:func:`align_with_trace` is an additive helper returning the per-step pairwise
alignments, handy when debugging why a given pair of sequences was matched.
"""

from copy import deepcopy
import numpy as np

from openalea.phenomenal.tracking.scoring import (
    vector_distance,
    profile_distance,
    gap_policy,
)


def needleman_wunsch(X, Y, gap, gap_extremity_factor=1.0, distance=None):
    """
    Performs pairwise alignment of profiles X and Y with Needleman-Wunsch algorithm.
    A profile is defined as an array of one or more sequences of the same length.
    Each sequence includes one or several vectors of the same length, and might
    contain gaps (vectors filled with NaN)
    Source code : https://gist.github.com/slowkow/06c6dba9180d013dfd82bec217d22eb5
    The source code was modified to correct a few errors, and adapted to fit all
    requirements (extremity gap, customized scoring function, etc.)

    Parameters
    ----------
    X : array of shape (profile length, sequence length, vector length)
        profile 1
    Y : array of shape (profile length, sequence length, vector length)
        profile 2
    gap : float
        gap penalty parameter (cost of an interior gap)
    gap_extremity_factor : float
        optional factor to increase/decrease gap penalty on sequence extremities
        (terminal gap penalty = gap * gap_extremity_factor)
    distance : callable, optional
        vector-to-vector scoring strategy used to score matches, defaults to
        :func:`scoring.vector_distance` (euclidean). Allows swapping the
        leaf-feature distance without editing the algorithm.

    Returns
    -------
    (list, list)
    """

    if X.size == 0 and Y.size == 0:
        rx, ry = [], []
        return rx, ry

    if distance is None:
        distance = vector_distance

    # gap policy : answers "interior gap?" vs "extremity gap?" in one place.
    policy = gap_policy(gap, gap_extremity_factor)
    gap_extremity = policy.extremity

    nx = X.shape[1]
    ny = Y.shape[1]

    # Optimal score at each possible pair of characters.
    F = np.zeros((nx + 1, ny + 1))
    F[:, 0] = np.linspace(start=0, stop=-nx * gap_extremity, num=nx + 1)
    F[0, :] = np.linspace(start=0, stop=-ny * gap_extremity, num=ny + 1)

    # Pointers to trace through an optimal alignment.
    P = np.zeros((nx + 1, ny + 1))
    P[:, 0] = 3
    P[0, :] = 4

    # Temporary scores.
    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            # match: dissimilarity of the two profile columns. When the columns
            # share no comparable (non-gap) pair, the extremity penalty is used
            # as the fallback score (see scoring.profile_distance).
            t[0] = F[i, j] - profile_distance(
                X[:, i, :], Y[:, j, :], empty_score=gap_extremity, distance=distance
            )
            # gap in one sequence: extremity penalty on the borders, else interior.
            t[1] = F[i, j + 1] - policy.penalty(at_extremity=(j + 1 == ny))
            t[2] = F[i + 1, j] - policy.penalty(at_extremity=(i + 1 == nx))

            tmax = np.max(t)
            F[i + 1, j + 1] = tmax
            if t[0] == tmax:
                P[i + 1, j + 1] += 2
            if t[1] == tmax:
                P[i + 1, j + 1] += 3
            if t[2] == tmax:
                P[i + 1, j + 1] += 4

    # Trace through an optimal alignment.
    i, j = nx, ny

    rx, ry = [], []
    condition = True
    while condition:
        if P[i, j] in [2, 5, 6, 9]:
            rx.append(i - 1)
            ry.append(j - 1)
            i -= 1
            j -= 1
        elif P[i, j] in [3, 5, 7, 9]:
            rx.append(i - 1)
            ry.append(-1)  # gap
            i -= 1
        elif P[i, j] in [4, 6, 7, 9]:
            rx.append(-1)  # gap
            ry.append(j - 1)
            j -= 1

        condition = i > 0 or j > 0

    rx = rx[::-1]
    ry = ry[::-1]

    return rx, ry


def scoring_function(vec1, vec2):
    """
    Compute a dissimilarity score between two vectors of same length, which is
    equal to their euclidian distance.

    Kept for backward compatibility; delegates to
    :func:`scoring.vector_distance`, which is the canonical scoring strategy.

    Parameters
    ----------
    vec1 : 1D array
    vec2 : 1D array, of same length than vec1

    Returns
    -------
    float
    """

    return vector_distance(vec1, vec2)


def alignment_score(x, y, gap_extremity):
    """
    Compute a dissimilarity score between two arrays of vectors x and y.
    x and y can have different lengths, but all vectors in x and y must have
    the same length.

    Kept for backward compatibility; delegates to
    :func:`scoring.profile_distance` with ``gap_extremity`` as the explicit
    all-gap fallback score.

    Parameters
    ----------
    x : 2D array
        size (number of vectors, vector length)
    y : 2D array
        size (number of vectors, vector length)
    gap_extremity : float

    Returns
    -------
    float

    """

    return profile_distance(x, y, empty_score=gap_extremity)


def insert_gaps(all_sequences, seq_indexes, alignment):
    """
    Add gaps in sequences of 'all_sequences' whose indexes is in 'seq_indexes'.
    A gap is defined as a NAN array element in a given sequence.
    Gaps positions are given by 'alignment'.

    Parameters
    ----------
    all_sequences : list(2D array)
    seq_indexes : list(int)
    alignment : list(int)
        result from needleman_wunsch()

    Returns
    -------
    """

    all_sequences2 = deepcopy(all_sequences)
    gap_indexes = [i for i, e in enumerate(alignment) if e == -1]

    vec_len = max(len(vec) for seq in all_sequences for vec in seq)

    for si in seq_indexes:
        for gi in gap_indexes:
            if all_sequences2[si].size == 0:
                all_sequences2[si] = np.full((1, vec_len), np.nan)
            else:
                all_sequences2[si] = np.insert(all_sequences2[si], gi, np.nan, 0)

    return all_sequences2


def _alignment_matrix_from_sequences(aligned_sequences):
    """Convert a list of gap-padded sequences into a matrix of vector indexes.

    Each sequence becomes a row; the value is the running index of the non-gap
    vectors, and ``-1`` marks a gap. This is the shape consumed by the
    post-processing step.
    """
    s = np.array(aligned_sequences).shape
    alignment_matrix = np.full((s[0], s[1]), -1)
    for i, aligned_seq in enumerate(aligned_sequences):
        no_gap = np.array([not all(np.isnan(e)) for e in aligned_seq])
        alignment_matrix[i][no_gap] = np.arange(sum(no_gap))
    return alignment_matrix


def _progressive_alignment(
    sequences, gap, gap_extremity_factor, start, align_range, distance, record_steps
):
    """Core progressive multiple sequence alignment.

    Shared implementation behind :func:`multi_alignment` (matrix only) and
    :func:`align_with_trace` (matrix + per-step record). Behaviour is identical
    to the historical ``multi_alignment`` body.

    Returns
    -------
    (alignment_matrix, steps)
        ``steps`` is an empty list unless ``record_steps`` is True.
    """

    assert -1 <= start <= len(sequences) - 1

    aligned_sequences = deepcopy(sequences)

    # init
    # (k_start -> 0) then (k_start -> n)
    k_start = len(aligned_sequences) - 1 if start == -1 else start
    alignment_order = np.array(
        list(range(0, k_start + 1)[::-1])
        + list(range(k_start + 1, len(aligned_sequences)))
    )

    steps = []
    for k in range(1, len(aligned_sequences)):
        xi = alignment_order[:k]  # ref
        yi = alignment_order[k]

        # select the 2 profiles to align
        xi_in_range = (
            xi
            if align_range is None
            else [val for val in xi if abs(val - k) <= align_range]
        )
        X = np.array([aligned_sequences[i] for i in xi_in_range])
        Y = np.array([aligned_sequences[yi]])

        # alignment
        rx, ry = needleman_wunsch(
            X, Y, gap, gap_extremity_factor=gap_extremity_factor, distance=distance
        )

        if record_steps:
            steps.append(
                {
                    "step": k,
                    "ref_indexes": list(xi_in_range),
                    "new_index": int(yi),
                    "rx": list(rx),
                    "ry": list(ry),
                }
            )

        # update all sequences from sq0 to sq yi
        aligned_sequences = insert_gaps(
            aligned_sequences, xi, rx
        )  # xi = sequences that all have already been aligned
        aligned_sequences = insert_gaps(aligned_sequences, [yi], ry)

    # convert list of aligned sequences (all having the same length) in a matrix
    # of vector indexes (-1 = gap)
    alignment_matrix = _alignment_matrix_from_sequences(aligned_sequences)

    return alignment_matrix, steps


def multi_alignment(
    sequences, gap, gap_extremity_factor=1.0, start=0, align_range=None, distance=None
):
    """
    Multiple sequence alignment algorithm to align n sequences, using a
    progressive method. At each step, a sequence (Y) is aligned with a matrix
    (X) corresponding to a profile (i.e. the alignment of k sequences)
    resulting in the alignment of k + 1 sequences. Each pairwise alignment
    of X vs Y is based on needleman-wunsch algorithm.

    Parameters
    ----------
    sequences : list of 2D arrays
        The list of sequences to align
    gap : float
        penalty parameter to add a gap
    gap_extremity_factor : float
        parameter to modify the gap penalty on sequence extremity positions,
        relatively to gap value.
        For example, if gap = 5 and gap_extremity_factor = 0.6,
        Then the penalty for terminal gaps equals 3.
    start : int
        sequences are progressively added to the global alignment from
        sequences[start] to sequences[0], then from
        sequences[start + 1] to sequences[-1]
    align_range : int
        When adding a new sequence to the global alignment, only the already
        aligned sequences with a distance inferior or equal to this parameter
        in the sequences order are used for the alignment.
    distance : callable, optional
        vector-to-vector scoring strategy, defaults to
        :func:`scoring.vector_distance`.

    Returns
    -------
    2D array
        alignment matrix of vector indexes (-1 = gap)
    """

    alignment_matrix, _ = _progressive_alignment(
        sequences,
        gap=gap,
        gap_extremity_factor=gap_extremity_factor,
        start=start,
        align_range=align_range,
        distance=distance,
        record_steps=False,
    )
    return alignment_matrix


def align_with_trace(
    sequences, gap, gap_extremity_factor=1.0, start=0, align_range=None, distance=None
):
    """Like :func:`multi_alignment` but also returns the per-step record.

    Additive debugging helper: the returned ``steps`` list contains, for every
    progressive step, the reference indexes, the newly added sequence index and
    the pairwise alignment (``rx``/``ry``). Use it to inspect *why* sequences
    were matched a given way instead of resorting to print statements.

    Returns
    -------
    (alignment_matrix, steps)
    """

    return _progressive_alignment(
        sequences,
        gap=gap,
        gap_extremity_factor=gap_extremity_factor,
        start=start,
        align_range=align_range,
        distance=distance,
        record_steps=True,
    )
