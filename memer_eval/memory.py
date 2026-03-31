"""Deploy-style episodic memory built from predicted candidate keyframes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import median_low
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class CandidateCluster:
    """One 1D cluster of raw candidate keyframe votes."""

    start_index: int
    end_index: int
    representative_index: int
    vote_count: int


def cluster_candidate_indices(
    candidate_indices: Sequence[int],
    merge_distance: int,
) -> List[CandidateCluster]:
    """Cluster sorted 1D candidate indices using the rollout merge rule."""
    normalized = sorted(int(index) for index in candidate_indices)
    if not normalized:
        return []

    clusters: List[List[int]] = []
    current = [normalized[0]]
    for index in normalized[1:]:
        if index - current[-1] <= merge_distance:
            current.append(index)
        else:
            clusters.append(current)
            current = [index]
    clusters.append(current)

    return [
        CandidateCluster(
            start_index=cluster[0],
            end_index=cluster[-1],
            representative_index=int(median_low(cluster)),
            vote_count=len(cluster),
        )
        for cluster in clusters
    ]


def count_candidate_votes(candidate_indices: Sequence[int]) -> Dict[int, int]:
    """Count duplicate-preserving keyframe votes at each raw frame index."""
    return dict(sorted(Counter(int(index) for index in candidate_indices).items()))


@dataclass
class EpisodicMemory:
    """Cluster predicted candidate keyframes and expose stable representatives."""

    merge_distance: int
    memory_length: int
    _candidate_indices: List[int] = field(default_factory=list)

    def add_candidates(self, absolute_indices: Sequence[int]) -> List[int]:
        """Add new candidate frame indices and return the accepted subset."""
        accepted = [int(index) for index in absolute_indices if int(index) >= 0]
        if not accepted:
            return []
        self._candidate_indices.extend(accepted)
        self._candidate_indices.sort()
        return accepted

    def all_candidates(self) -> List[int]:
        """Return all candidate indices including duplicates."""
        return list(self._candidate_indices)

    def selected_indices(self) -> List[int]:
        """Return median representative indices for all current clusters."""
        representatives = [
            cluster.representative_index
            for cluster in cluster_candidate_indices(self._candidate_indices, self.merge_distance)
        ]
        return representatives[-self.memory_length :]

    def visible_indices(self, current_context_indices: Sequence[int]) -> List[int]:
        """Return selected memory indices that are no longer in the recent context."""
        context_set = set(int(index) for index in current_context_indices)
        visible = [index for index in self.selected_indices() if index not in context_set]
        return visible[-self.memory_length :]
