"""Online deployment helpers for MemER high-level subtask prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .camera_layout import load_camera_layout
from .contract import build_human_prompt
from .inference import QwenStructuredPredictor, StructuredPredictor
from .memory import EpisodicMemory
from .rollout import (
    DEFAULT_VIEW_HEIGHT,
    DEFAULT_VIEW_WIDTH,
    PREFERRED_STACKED_CAMERA_KEYS,
    build_recent_context_indices,
    effective_merge_distance_raw_frames,
    map_relative_positions_to_absolute,
    stack_frame_images,
    to_numpy_image,
)


@dataclass
class DeploymentConfig:
    frame_subsample: int = 5
    recent_frames_length: int = 8
    memory_length: int = 8
    merge_distance: int = 5
    view_width: Optional[int] = None
    view_height: Optional[int] = None
    camera_key: Optional[str] = None
    camera_keys: Optional[List[str]] = None
    camera_layout_config: Optional[str] = None


@dataclass
class DeploymentStepResult:
    timestep: int
    current_subtask: Optional[str]
    parse_ok: bool
    parse_error: Optional[str]
    predicted_keyframe_positions: List[int]
    mapped_keyframe_indices: List[int]
    invalid_keyframe_positions: List[int]
    context_indices: List[int]
    memory_indices_before: List[int]
    memory_indices_after: List[int]
    all_candidate_indices: List[int]
    raw_text: str


class MemERDeploymentPolicy:
    """Stateful deploy-time MemER wrapper around a structured predictor."""

    def __init__(
        self,
        predictor: StructuredPredictor,
        *,
        instruction: Optional[str] = None,
        config: Optional[DeploymentConfig] = None,
    ) -> None:
        self.predictor = predictor
        self.config = config or DeploymentConfig()
        self.instruction = instruction
        self._camera_keys, self._view_width, self._view_height = _resolve_camera_settings(self.config)
        self.reset(instruction=instruction)

    @classmethod
    def from_qwen_checkpoint(
        cls,
        model_path: str,
        *,
        processor_path: Optional[str] = None,
        instruction: Optional[str] = None,
        config: Optional[DeploymentConfig] = None,
        device: Optional[str] = None,
        dtype: str = "auto",
        attn_implementation: Optional[str] = None,
        max_new_tokens: int = 128,
    ) -> "MemERDeploymentPolicy":
        predictor = QwenStructuredPredictor(
            model_path=model_path,
            processor_path=processor_path,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            max_new_tokens=max_new_tokens,
        )
        return cls(predictor, instruction=instruction, config=config)

    def reset(self, instruction: Optional[str] = None) -> None:
        if instruction is not None:
            self.instruction = instruction
        self._frames: List[Any] = []
        self._memory = EpisodicMemory(
            merge_distance=effective_merge_distance_raw_frames(
                self.config.merge_distance,
                self.config.frame_subsample,
            ),
            memory_length=self.config.memory_length,
        )

    def step(self, *observations: Any, instruction: Optional[str] = None) -> DeploymentStepResult:
        """Add one or more observations and run inference once at the final timestep.

        When multiple observations are provided, all frames are buffered but the
        VLM is only invoked once — at the last timestep.  Memory and context are
        computed relative to that final timestep, keeping the policy fully stateful
        across calls.
        """
        if not observations:
            raise ValueError("step requires at least one observation.")
        if instruction is not None:
            self.instruction = instruction
        if not self.instruction:
            raise ValueError("MemERDeploymentPolicy.step requires an instruction. Pass it to the constructor, reset, or step.")

        for obs in observations:
            frame = _observation_to_pil_image(
                obs,
                camera_keys=self._camera_keys,
                view_width=self._view_width,
                view_height=self._view_height,
            )
            self._frames.append(frame)

        timestep = len(self._frames) - 1

        context_indices = build_recent_context_indices(
            timestep=timestep,
            frame_subsample=self.config.frame_subsample,
            recent_frames_length=self.config.recent_frames_length,
        )
        memory_indices_before = self._memory.visible_indices(context_indices)
        prompt = build_human_prompt(
            instruction=self.instruction,
            memory_count=len(memory_indices_before),
            recent_count=len(context_indices),
        )
        images = [self._frames[index] for index in memory_indices_before + context_indices]
        prediction = self.predictor.predict(prompt, images)

        mapped_keyframes: List[int] = []
        invalid_positions: List[int] = []
        if prediction.parse_ok:
            mapped_keyframes, invalid_positions = map_relative_positions_to_absolute(
                prediction.keyframe_positions,
                context_indices,
            )
            self._memory.add_candidates(mapped_keyframes)

        memory_indices_after = self._memory.visible_indices(context_indices)
        return DeploymentStepResult(
            timestep=timestep,
            current_subtask=prediction.current_subtask,
            parse_ok=prediction.parse_ok,
            parse_error=prediction.parse_error,
            predicted_keyframe_positions=list(prediction.keyframe_positions),
            mapped_keyframe_indices=mapped_keyframes,
            invalid_keyframe_positions=invalid_positions,
            context_indices=context_indices,
            memory_indices_before=memory_indices_before,
            memory_indices_after=memory_indices_after,
            all_candidate_indices=self._memory.all_candidates(),
            raw_text=prediction.raw_text,
        )

    def predict_from_history(
        self,
        observations: Sequence[Any],
        *,
        instruction: Optional[str] = None,
        reset: bool = True,
    ) -> DeploymentStepResult:
        if not observations:
            raise ValueError("predict_from_history requires at least one observation.")
        if reset:
            self.reset(instruction=instruction)
        elif instruction is not None:
            self.instruction = instruction

        result: Optional[DeploymentStepResult] = None
        for observation in observations:
            result = self.step(observation)
        assert result is not None
        return result


def predict_subtask_from_observation_history(
    observations: Sequence[Any],
    *,
    predictor: StructuredPredictor,
    instruction: str,
    config: Optional[DeploymentConfig] = None,
) -> DeploymentStepResult:
    policy = MemERDeploymentPolicy(
        predictor=predictor,
        instruction=instruction,
        config=config,
    )
    return policy.predict_from_history(observations, reset=False)


def _resolve_camera_settings(config: DeploymentConfig) -> tuple[List[str], int, int]:
    camera_layout = load_camera_layout(config.camera_layout_config)
    if camera_layout and (config.camera_key or config.camera_keys):
        raise ValueError(
            "Use only one of camera_layout_config or camera_key/camera_keys for deployment."
        )

    if camera_layout:
        camera_keys = list(camera_layout.camera_keys)
    elif config.camera_keys:
        camera_keys = list(config.camera_keys)
    elif config.camera_key:
        camera_keys = [config.camera_key]
    else:
        camera_keys = list(PREFERRED_STACKED_CAMERA_KEYS)

    view_width = (
        int(config.view_width)
        if config.view_width is not None
        else (
            camera_layout.view_width
            if camera_layout and camera_layout.view_width is not None
            else DEFAULT_VIEW_WIDTH
        )
    )
    view_height = (
        int(config.view_height)
        if config.view_height is not None
        else (
            camera_layout.view_height
            if camera_layout and camera_layout.view_height is not None
            else DEFAULT_VIEW_HEIGHT
        )
    )
    return camera_keys, view_width, view_height


def _observation_to_pil_image(
    observation: Any,
    *,
    camera_keys: Sequence[str],
    view_width: int,
    view_height: int,
) -> Any:
    from PIL import Image

    if isinstance(observation, dict):
        missing = [key for key in camera_keys if key not in observation]
        if missing:
            available = ", ".join(sorted(observation.keys()))
            raise ValueError(f"Observation is missing camera key(s) {missing}. Available keys: {available}")
        image_values = [observation[key] for key in camera_keys]
    elif len(camera_keys) == 1:
        image_values = [observation]
    else:
        raise ValueError(
            "Multi-camera deployment expects each observation to be a dict keyed by camera name."
        )

    if len(image_values) == 1:
        array = to_numpy_image(image_values[0])
        return Image.fromarray(array)

    stacked = stack_frame_images(image_values)
    return Image.fromarray(stacked).resize(
        (view_width, view_height * len(camera_keys)),
        resample=_get_resample(),
    )


def _get_resample() -> Any:
    from PIL import Image

    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC
