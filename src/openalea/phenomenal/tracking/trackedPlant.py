"""
Time-lapse tracking of leaves in a time-series of 3D maize segmentations.
//!\\ In the tracking algorithm, ranks start at 0. But in the final output,
ranks start at 1. (see get_ranks() method)

The class is organised so that *data preparation* (selecting mature leaves,
building feature vectors / feature sequences, gathering candidate leaves per
rank, picking valid snapshots) is kept in dedicated helper methods, separate
from the *algorithm execution* (the alignment call, the medoid selection, the
backwards growing-leaf search). The public methods orchestrate these steps and
keep their historical signatures and defaults; the defaults are sourced from
:data:`config.DEFAULT_CONFIG` so there is a single place to tune them.
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from openalea.phenomenal.tracking.alignment import align_with_trace
from openalea.phenomenal.tracking.alignment_postprocessing import (
    detect_abnormal_ranks,
    leaf_polylines_distance,
)
from openalea.phenomenal.tracking.config import DEFAULT_CONFIG


def check_time_intervals(
    times, discontinuity=DEFAULT_CONFIG.continuity.discontinuity
):
    """
    If a gap between two successive time steps is too high compared to the
    median interval, all time-steps after the gap are invalidated

    Parameters
    ----------
    times : list(float)
    discontinuity : float
        a gap larger than ``discontinuity * median_interval`` invalidates every
        later time-step (default sourced from config).

    Returns
    -------
    list(bool)
    """

    assert list(times) == sorted(times)

    dt_median = np.median(np.diff(times))
    valid = np.array([True] * len(times))
    for i in range(1, len(times)):
        if (times[i] - times[i - 1]) > discontinuity * dt_median:
            valid[i:] = False
    return valid


@dataclass
class MatureTrackingDebug:
    """Intermediate artefacts produced by :meth:`TrackedPlant.mature_leaf_tracking`.

    Stored on ``TrackedPlant.debug`` after each run so mis-alignments can be
    inspected without adding print statements:

    * ``feature_sequences`` -- the per-snapshot sequences of leaf feature
      vectors fed to the alignment.
    * ``alignment_matrix`` -- the raw alignment matrix (before abnormal-rank
      removal).
    * ``abnormal_ranks`` -- the ranks removed by the post-processing step.
    * ``steps`` -- the per-step pairwise alignments (see
      :func:`alignment.align_with_trace`).
    """

    feature_sequences: List = field(default_factory=list)
    alignment_matrix: Optional[np.ndarray] = None
    abnormal_ranks: List[int] = field(default_factory=list)
    steps: List = field(default_factory=list)


class TrackedLeaf:
    """Describe a leaf organ, with attributes specific to leaf tracking
    algorithm."""

    def __init__(self, polyline, features):
        """
        Parameters
        ----------
        polyline : (n, 3) numpy array
        features : dict
            {'mature': bool, 'azimuth': float, 'height': float, 'length': float}
        """

        # for mature leaf tracking
        self.features = features
        self.vec = np.array([])

        # for growing leaf tracking
        self.polyline = polyline

    @staticmethod
    def _azimuth_components(azimuth_deg):
        """Map an azimuth in degrees to a point on the unit circle.

        Returns ``(cos, sin)`` of the azimuth (converted to radians), so that
        the periodicity of the angle is respected by the euclidean distance used
        during alignment.
        """
        azimuth_rad = azimuth_deg / 360 * 2 * np.pi
        return np.cos(azimuth_rad), np.sin(azimuth_rad)

    def compute_features_vector(self, w_h, w_l):
        """for the sequence alignment of mature leaves"""

        if not self.features["mature"]:
            warnings.warn("This method is supposed to be used for mature leaves")

        cos_az, sin_az = self._azimuth_components(self.features["azimuth"])

        self.vec = np.array(
            [
                cos_az,
                sin_az,
                w_h * self.features["height"],
                w_l * self.features["length"],
            ]
        )


class TrackedSnapshot:
    """Describe the plant segmentation at a given time point, particularly
    the order of leaves, which is modified during leaf tracking."""

    def __init__(self, leaves, check):
        """
        Parameters
        ----------
        leaves : list(TrackedLeaf)
        check : list(bool)
        """

        self.leaves = leaves

        self.check_continuity = check

        # self.sequence gives the ranks of leaves in self.leaves.
        # for example, if self.order[5] = 2, it means that self.leaves[2] is associated to rank 5+1=6.
        # -1 = no leaf
        self.sequence = []

    def mature_leaf_indices(self):
        """Indexes of the mature leaves in ``self.leaves`` (data preparation)."""
        return [
            i for i, leaf in enumerate(self.leaves) if leaf.features["mature"]
        ]

    def leaf_ranks(self):
        """
        returns the ranks of leaves contained in self.leaves

        example :
        self.leaves = [leaf0, leaf1, leaf2, leaf3]
        self.sequence = [-1, -1, 0, 1, -1, 2, 3, -1]
        ===> self.leaf_ranks() returns [3, 4, 6, 7]

        WARNING : the rank of a leaf is given by its position in
        TrackedSnapshot.sequence, which starts at 0. But leaf ranks are usually
        numerated starting from 1: this second option is used in the output
        from this function.
        """
        return [
            self.sequence.index(i) + 1 if i in self.sequence else 0
            for i in range(len(self.leaves))
        ]


class TrackedPlant:
    """Main class for leaf tracking"""

    def __init__(self, snapshots):
        """
        Parameters
        ----------
        snapshots : list(TrackedSnapshot)
        """

        self.snapshots = snapshots

        # Intermediate artefacts of the last mature_leaf_tracking() run, exposed
        # for debugging. None until the first run.
        self.debug = None

    @staticmethod
    def load(segmentation_time_series):
        """
        Parameters
        ----------
        segmentation_time_series : list
            list of dict {'time': float,
                          'polylines_sequence': list of polylines,
                          'features_sequence': list of {'mature': bool, 'azimuth': float,
                                                        'height': float, 'length': float}
                          }

        Returns
        -------
        TrackedPlant
        """

        times = [seg["time"] for seg in segmentation_time_series]
        times = sorted(times)

        # verify temporal order of the time-series
        if times != sorted(times):
            raise Exception("objects need to be ordered by temporal order")

        # check if there is no big time gap in the time-series
        checks_continuity = check_time_intervals(times)

        # initialize the TrackedPlant object
        snapshots = []
        for seg, check in zip(segmentation_time_series, checks_continuity):
            leaves = []
            for polyline, features in zip(
                seg["polylines_sequence"], seg["features_sequence"]
            ):
                assert all(
                    var in features for var in ["mature", "azimuth", "height", "length"]
                )
                leaves.append(TrackedLeaf(polyline=polyline, features=features))
            snapshots.append(TrackedSnapshot(leaves, check))

        return TrackedPlant(snapshots=snapshots)

    # ------------------------------------------------------------------ #
    # Mature leaf tracking                                               #
    # ------------------------------------------------------------------ #

    def _init_mature_sequences(self):
        """Data preparation: reset each snapshot sequence to its mature indices."""
        for snapshot in self.snapshots:
            snapshot.sequence = snapshot.mature_leaf_indices()

    def _compute_mature_feature_vectors(self, w_h, w_l):
        """Data preparation: compute the feature vector of every mature leaf."""
        for snapshot in self.snapshots:
            for leaf in snapshot.leaves:
                if leaf.features["mature"]:
                    leaf.compute_features_vector(w_h=w_h, w_l=w_l)

    def _mature_feature_sequences(self):
        """Data preparation: time-series of sequences of mature feature vectors.

        Sequences may have different sizes; the vectors all share the same size.
        """
        return [
            np.array([snapshot.leaves[i].vec for i in snapshot.sequence])
            for snapshot in self.snapshots
        ]

    def _apply_alignment(self, alignment_matrix):
        """Book-keeping: rewrite each snapshot sequence from the alignment matrix.

        A ``-1`` stays a gap; any other value indexes back into the snapshot's
        pre-alignment mature-index sequence.
        """
        for t, aligned_sequence in enumerate(alignment_matrix):
            self.snapshots[t].sequence = [
                -1 if i == -1 else self.snapshots[t].sequence[i]
                for i in aligned_sequence
            ]

    def _remove_ranks(self, ranks_to_remove):
        """Post-processing: drop the given ranks (columns) from every sequence."""
        ranks_to_remove = set(ranks_to_remove)
        for snapshot in self.snapshots:
            snapshot.sequence = [
                e
                for i, e in enumerate(snapshot.sequence)
                if i not in ranks_to_remove
            ]

    def mature_leaf_tracking(
        self,
        gap=DEFAULT_CONFIG.alignment.gap,
        gap_extremity_factor=DEFAULT_CONFIG.alignment.gap_extremity_factor,
        start=DEFAULT_CONFIG.alignment.start,
        w_h=DEFAULT_CONFIG.feature_weights.height,
        w_l=DEFAULT_CONFIG.feature_weights.length,
        align_range=DEFAULT_CONFIG.alignment.align_range,
        rank_attribution=True,
    ):
        """
        alignment and rank attributions in a time-series of sequences of leaves.
        Step 1 : use a multiple sequence alignment algorithm to align the sequences.
        Step 2 (post-processing) : Detect and remove abnormal group of leaves ; final rank attribution.

        Parameters
        ----------
        gap : float
            weight  for pairwise sequence alignment
        gap_extremity_factor : float
            parameter allowing to change the value of the gap penalty for terminal gaps (terminal gap penalty = gap *
            gap_extremity_factor)
        start : int
            sequences are progressively added to the global alignment from sequences[start] to sequences[0], then from
            sequences[start + 1] to sequences[-1]
        align_range : int
            When adding a new sequence to the global alignment, only the already aligned sequences with a distance
            inferior or equal to this parameter in the sequences order are used for the alignment.
        w_h : float
            weight associated to insertion height feature in a leaf feature vector
        w_l : float
            weight associated to length feature in a leaf feature vector
        rank_attribution : bool
            choose if step 2 is done (True) or not (False)

        Returns
        -------
        """

        # _____ data preparation _____________________________________________
        self._init_mature_sequences()
        self._compute_mature_feature_vectors(w_h=w_h, w_l=w_l)
        feature_sequences = self._mature_feature_sequences()

        # _____ Step 1 - multiple sequence alignment _________________________
        alignment_matrix, steps = align_with_trace(
            sequences=feature_sequences,
            gap=gap,
            gap_extremity_factor=gap_extremity_factor,
            align_range=align_range,
            start=start,
        )
        self._apply_alignment(alignment_matrix)

        # _____ Step 2 - From relative leaf ranks to absolute leaf ranks ______
        abnormal_ranks = []
        if rank_attribution:
            abnormal_ranks = detect_abnormal_ranks(alignment_matrix)
            self._remove_ranks(abnormal_ranks)

        # _____ expose intermediate results for debugging ____________________
        self.debug = MatureTrackingDebug(
            feature_sequences=feature_sequences,
            alignment_matrix=alignment_matrix,
            abnormal_ranks=abnormal_ranks,
            steps=steps,
        )

    # ------------------------------------------------------------------ #
    # Reference (median) skeleton                                        #
    # ------------------------------------------------------------------ #

    def _mature_leaves_at_rank(self, rank, nmax):
        """Data preparation: mature leaves observed at ``rank``, capped to nmax.

        Old leaves (beyond ``nmax``) are dropped to avoid senescence artefacts
        biasing the reference shape.
        """
        leaves = [
            s.leaves[s.sequence[rank]]
            for s in self.snapshots
            if s.sequence[rank] != -1  # -1 = no leaf
        ]
        leaves = [leaf for leaf in leaves if leaf.features["mature"]]

        # remove old leaves (that could have a different shape)
        # TODO use value of times instead
        return leaves[:nmax]

    @staticmethod
    def _medoid_leaf(leaves):
        """Algorithm: pick the leaf whose feature vector is closest to the mean."""
        vectors = np.array([leaf.vec for leaf in leaves])
        mean_vector = np.mean(vectors, axis=0)
        dists = [np.sum(abs(vec - mean_vector)) for vec in vectors]
        return leaves[np.argmin(dists)]

    def get_ref_skeleton(self, nmax=DEFAULT_CONFIG.ref_skeleton.nmax):
        """
        Compute a median skeleton {rank : leaf}.
        For each rank, the leaf whose vector is less distant to all other leaves
        from the same ranks is selected.

        Parameters
        ----------
        nmax : int
            max number of leaves considered at a given rank (to avoid old leaves which can have senescence)

        Returns
        -------
        """

        ref_skeleton = {}

        ranks = range(len(self.snapshots[0].sequence))
        for rank in ranks:
            leaves = self._mature_leaves_at_rank(rank, nmax)
            if len(leaves) > 0:
                ref_skeleton[rank] = self._medoid_leaf(leaves)

        return ref_skeleton

    # ------------------------------------------------------------------ #
    # Growing leaf tracking                                              #
    # ------------------------------------------------------------------ #

    def _valid_snapshots(self):
        """Data preparation: snapshots not invalidated by a time discontinuity."""
        return [
            snapshot for snapshot in self.snapshots if snapshot.check_continuity
        ]

    @staticmethod
    def _untracked_growing_indices(snapshot):
        """Data preparation: growing leaves of a snapshot not yet tracked."""
        return [
            g
            for g, leaf in enumerate(snapshot.leaves)
            if not leaf.features["mature"]  # avoids non-tracked mature
            and g not in snapshot.sequence  # avoids already-tracked growing
        ]

    @staticmethod
    def _closest_growing_leaf(snapshot, leaf_ref):
        """Algorithm: index of the growing leaf closest to ``leaf_ref``.

        Returns None when the snapshot has no untracked growing leaf.
        """
        candidates = TrackedPlant._untracked_growing_indices(snapshot)
        if not candidates:
            return None
        dists = [
            leaf_polylines_distance(
                polyline_ref=leaf_ref.polyline,
                polyline_candidate=snapshot.leaves[g].polyline,
            )
            for g in candidates
        ]
        return candidates[np.argmin(dists)]

    def growing_leaf_tracking(self):
        """
        Tracking of growing leaves over time.
        To use AFTER self.align_mature()
        """

        # _____ data preparation _____________________________________________
        valid_snapshots = self._valid_snapshots()
        mature_ref = self.get_ref_skeleton()

        # _____ backwards search of each leaf in its growth phase ____________
        for r, leaf_ref in mature_ref.items():
            # day t when leaf starts to be mature
            t_mature = next(
                (
                    t
                    for t, snapshot in enumerate(valid_snapshots)
                    if snapshot.sequence[r] != -1
                )
            )

            # backwards tracking of this leaf
            for t in range(t_mature)[::-1]:
                snapshot = valid_snapshots[t]
                g = self._closest_growing_leaf(snapshot, leaf_ref)
                if g is not None:
                    valid_snapshots[t].sequence[r] = g

    def output(self):
        ranks = [snapshot.leaf_ranks() for snapshot in self.snapshots]
        features = [
            [leaf.features for leaf in snapshot.leaves] for snapshot in self.snapshots
        ]
        checks = np.array([snapshot.check_continuity for snapshot in self.snapshots])
        return ranks, features, checks
