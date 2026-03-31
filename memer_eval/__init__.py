"""Release-ready MemER offline rollout evaluation utilities."""

from .camera_layout import CameraLayout, load_camera_layout
from .contract import (
    DEFAULT_SYSTEM_PROMPT,
    KEYFRAME_POSITIONS_KEY,
    PREDICTION_ALIASES,
    PREDICTION_KEY,
    build_system_prompt,
    build_human_prompt,
    build_user_prompt,
    build_user_message,
    compute_target_index,
    normalize_subtask_label,
)
from .deploy import (
    DeploymentConfig,
    DeploymentStepResult,
    MemERDeploymentPolicy,
    predict_subtask_from_observation_history,
)
from .inference import ModelPrediction, QwenStructuredPredictor, StructuredPredictor
from .memory import CandidateCluster, EpisodicMemory, cluster_candidate_indices, count_candidate_votes
from .rollout import RolloutConfig, RolloutSummary, evaluate_rollout

__all__ = [
    "CameraLayout",
    "CandidateCluster",
    "DeploymentConfig",
    "DeploymentStepResult",
    "DEFAULT_SYSTEM_PROMPT",
    "EpisodicMemory",
    "KEYFRAME_POSITIONS_KEY",
    "MemERDeploymentPolicy",
    "ModelPrediction",
    "PREDICTION_ALIASES",
    "PREDICTION_KEY",
    "QwenStructuredPredictor",
    "RolloutConfig",
    "RolloutSummary",
    "StructuredPredictor",
    "build_system_prompt",
    "build_human_prompt",
    "build_user_prompt",
    "build_user_message",
    "cluster_candidate_indices",
    "compute_target_index",
    "count_candidate_votes",
    "evaluate_rollout",
    "load_camera_layout",
    "normalize_subtask_label",
    "predict_subtask_from_observation_history",
]
