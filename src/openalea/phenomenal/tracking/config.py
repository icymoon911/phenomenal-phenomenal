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
Centralized configuration for the leaf-tracking pipeline.

Every tunable that used to live as a *magic number* spread across
``alignment.py``, ``alignment_postprocessing.py`` and ``trackedPlant.py`` is
gathered here as a frozen dataclass field. Changing a weight or a penalty is now
a one-line edit in a single file instead of a hunt through the algorithm code.

The default values defined here are exactly the historical defaults of the
pipeline; importing :data:`DEFAULT_CONFIG` and reading its fields is equivalent
to the previous hard-coded behaviour.

Layering reminder
-----------------
The pipeline is organised in three stages, and the config mirrors them:

* **scoring**  -> :class:`FeatureWeights` (how a leaf becomes a vector) and the
  gap penalties used by the dissimilarity scoring (:class:`GapPenalty`).
* **alignment** -> :class:`AlignmentParams` (Needleman-Wunsch / multiple
  sequence alignment knobs).
* **post-processing** -> :class:`PostProcessingParams` (abnormal-rank removal,
  growing-leaf polyline distance) plus :class:`RefSkeletonParams` and
  :class:`ContinuityParams` used around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class FeatureWeights:
    """Weights applied when a mature leaf is turned into a feature vector.

    Historically these were the ``w_h`` and ``w_l`` arguments of
    :meth:`TrackedLeaf.compute_features_vector`.
    """

    height: float = 0.03  # w_h : weight of the insertion-height feature
    length: float = 0.004  # w_l : weight of the leaf-length feature


@dataclass(frozen=True)
class GapPenalty:
    """Gap penalties for the Needleman-Wunsch dynamic programming.

    A single object answers both questions the alignment asks: the cost of an
    *interior* gap (``gap``) and the cost of a gap sitting on a sequence
    *extremity* (``extremity``). The extremity cost is derived from ``gap``
    through ``extremity_factor`` so the two can never drift apart.
    """

    gap: float = 12.0
    extremity_factor: float = 0.2

    @property
    def extremity(self) -> float:
        """Penalty applied to gaps located on a sequence extremity."""
        return self.gap * self.extremity_factor

    def penalty(self, at_extremity: bool) -> float:
        """Return the interior or extremity penalty depending on position."""
        return self.extremity if at_extremity else self.gap


@dataclass(frozen=True)
class AlignmentParams:
    """Knobs of the multiple sequence alignment used for mature leaves."""

    gap: float = 12.0
    gap_extremity_factor: float = 0.2
    start: int = 0
    align_range: Optional[int] = None

    def gap_penalty(self) -> GapPenalty:
        """Build the :class:`GapPenalty` strategy for these parameters."""
        return GapPenalty(gap=self.gap, extremity_factor=self.gap_extremity_factor)


@dataclass(frozen=True)
class PostProcessingParams:
    """Parameters of the steps applied *after* the raw alignment."""

    # A rank (column of the alignment matrix) is flagged abnormal when it holds
    # fewer aligned leaves than this fraction of its neighbours' average.
    abnormal_min_relative_count: float = 0.5

    # Number of samples used to compare two leaf polylines (growing tracking).
    polyline_distance_samples: int = 20

    # Numerical floor used when normalising a polyline distance by a length.
    polyline_distance_eps: float = 1e-6


@dataclass(frozen=True)
class RefSkeletonParams:
    """Parameters controlling the reference (median) skeleton computation."""

    # Max number of mature leaves kept per rank, oldest leaves dropped to avoid
    # senescence artefacts biasing the median shape.
    nmax: int = 15


@dataclass(frozen=True)
class ContinuityParams:
    """Parameters of the temporal-continuity check on the time-series."""

    # A time interval larger than ``discontinuity * median_interval`` marks the
    # start of a discontinuity; every later snapshot is invalidated.
    discontinuity: float = 5.0


@dataclass(frozen=True)
class TrackingConfig:
    """Aggregate of every tunable of the tracking pipeline.

    Pass a customised instance around (or simply read :data:`DEFAULT_CONFIG`) to
    tune behaviour from one place. All public functions keep their historical
    keyword arguments; those keyword defaults are sourced from this object so
    there is a single source of truth.
    """

    feature_weights: FeatureWeights = field(default_factory=FeatureWeights)
    alignment: AlignmentParams = field(default_factory=AlignmentParams)
    post_processing: PostProcessingParams = field(default_factory=PostProcessingParams)
    ref_skeleton: RefSkeletonParams = field(default_factory=RefSkeletonParams)
    continuity: ContinuityParams = field(default_factory=ContinuityParams)


#: Shared default configuration. Public keyword defaults are sourced from this.
DEFAULT_CONFIG = TrackingConfig()
