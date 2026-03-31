"""Shared helpers for data export and evaluation."""

from typing import List


def align_frames_with_subsampling(
    all_keyframe_indices: List[int],
    start_idx: int,
    end_idx: int,
    frame_subsample: int,
) -> List[int]:
    """Align keyframes to subsample boundaries and remove duplicates while preserving order."""
    frame_indices = [
        ((k - start_idx) // frame_subsample) * frame_subsample + start_idx
        for k in all_keyframe_indices
        if start_idx <= k < end_idx
    ]
    return list(dict.fromkeys(frame_indices))
