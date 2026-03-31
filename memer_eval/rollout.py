"""Offline deploy-style MemER rollout evaluation on LeRobot datasets."""

from __future__ import annotations

import io
import json
import logging
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera_layout import load_camera_layout
from .contract import (
    build_human_prompt,
    compute_target_index,
    normalize_subtask_label,
)
from .dataset_metadata import (
    build_index_mapping_from_dataframe,
    extract_instruction,
    extract_subtask_label,
    resolve_dataset_args,
    scalar_to_int,
)
from .inference import ModelPrediction, QwenStructuredPredictor, StructuredPredictor
from .memory import EpisodicMemory


DEFAULT_VIEW_WIDTH = 320
DEFAULT_VIEW_HEIGHT = 180
PREFERRED_STACKED_CAMERA_KEYS = (
    "observation.images.wrist_left",
    "observation.images.exterior_1_left",
)


@dataclass
class DatasetContext:
    repo_id: str
    dataset_root: Optional[Path]
    camera_keys: List[str]
    task_map: Dict[int, str]
    subtask_map: Dict[int, str]
    episode_indices: List[int]


@dataclass
class RolloutConfig:
    model_path: str
    output_dir: str
    lerobot_path: Optional[str] = None
    repo_id: Optional[str] = None
    processor_path: Optional[str] = None
    system_role: str = "system"
    camera_key: Optional[str] = None
    camera_keys: Optional[List[str]] = None
    camera_layout_config: Optional[str] = None
    high_level_instruction: Optional[str] = None
    frame_subsample: int = 5
    recent_frames_length: int = 8
    memory_length: int = 8
    prediction_horizon: int = 2
    merge_distance: int = 5
    view_width: Optional[int] = None
    view_height: Optional[int] = None
    max_new_tokens: int = 128
    device: Optional[str] = None
    dtype: str = "auto"
    attn_implementation: Optional[str] = None
    max_episodes: Optional[int] = None
    episode_indices: Optional[List[int]] = None
    save_raw_responses: bool = False
    log_level: str = "INFO"


@dataclass
class EpisodeSummary:
    episode_index: int
    timesteps: int
    parse_failures: int
    raw_accuracy: float
    normalized_accuracy: float


@dataclass
class RolloutSummary:
    total_examples: int
    raw_accuracy: float
    normalized_accuracy: float
    parse_failure_rate: float
    episode_summaries: List[EpisodeSummary] = field(default_factory=list)
def load_dataset_context(lerobot_path: Optional[str], repo_id: Optional[str]) -> DatasetContext:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    resolved_repo_id, dataset_root = resolve_dataset_args(lerobot_path, repo_id)
    metadata = LeRobotDatasetMetadata(
        repo_id=resolved_repo_id,
        root=dataset_root,
        force_cache_sync=False,
    )

    task_map = build_index_mapping_from_dataframe(getattr(metadata, "tasks", None), "task_index")
    subtask_map = build_index_mapping_from_dataframe(getattr(metadata, "subtasks", None), "subtask_index")
    if not subtask_map:
        raise ValueError("Dataset metadata does not include subtasks.")

    info = metadata.info
    camera_keys = list(getattr(metadata, "camera_keys", []) or [])
    if not camera_keys:
        features = info.get("features", {})
        camera_keys = [
            key
            for key, value in features.items()
            if isinstance(value, dict) and value.get("dtype") in {"image", "video"}
        ]
    if not camera_keys:
        raise ValueError("Dataset metadata does not expose image camera keys.")

    return DatasetContext(
        repo_id=resolved_repo_id,
        dataset_root=dataset_root,
        camera_keys=camera_keys,
        task_map=task_map,
        subtask_map=subtask_map,
        episode_indices=list(range(int(metadata.total_episodes))),
    )


def resolve_camera_settings(
    context: DatasetContext,
    config: RolloutConfig,
) -> Tuple[List[str], int, int]:
    camera_layout = load_camera_layout(config.camera_layout_config)
    if camera_layout and (config.camera_key or config.camera_keys):
        raise ValueError(
            "Use only one of camera_layout_config or camera_key/camera_keys for rollout eval."
        )

    if camera_layout:
        selected = list(camera_layout.camera_keys)
    elif config.camera_keys:
        selected = list(config.camera_keys)
    elif config.camera_key:
        selected = [config.camera_key]
    elif all(key in context.camera_keys for key in PREFERRED_STACKED_CAMERA_KEYS):
        selected = list(PREFERRED_STACKED_CAMERA_KEYS)
    else:
        selected = [context.camera_keys[0]]

    missing = [key for key in selected if key not in context.camera_keys]
    if missing:
        available = ", ".join(context.camera_keys)
        raise ValueError(f"Camera key(s) not found: {missing}. Available: {available}")
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
    return selected, view_width, view_height

def to_numpy_image(value: Any) -> np.ndarray:
    if hasattr(value, "convert") and callable(getattr(value, "convert")):
        from PIL import Image

        if isinstance(value, Image.Image):
            return np.asarray(value.convert("RGB"), dtype=np.uint8)

    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes:
            from PIL import Image

            with Image.open(io.BytesIO(image_bytes)) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        if image_path:
            from PIL import Image

            with Image.open(image_path) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()

    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        return np.repeat(array, 3, axis=2).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def get_resample():
    from PIL import Image

    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def stack_frame_images(image_values: Sequence[Any]) -> np.ndarray:
    arrays = [to_numpy_image(image_value) for image_value in image_values]
    max_width = max(array.shape[1] for array in arrays)
    padded_arrays = []

    for array in arrays:
        height, width, channels = array.shape
        padded = np.zeros((height, max_width, channels), dtype=np.uint8)
        padded[:height, :width, :channels] = array
        padded_arrays.append(padded)

    return np.concatenate(padded_arrays, axis=0)


class EpisodeFrameCache:
    """Lazy in-memory frame renderer for one episode."""

    def __init__(
        self,
        items: Sequence[Dict[str, Any]],
        camera_keys: Sequence[str],
        view_width: int,
        view_height: int,
    ) -> None:
        self.items = items
        self.camera_keys = list(camera_keys)
        self.view_width = view_width
        self.view_height = view_height
        self._cache: Dict[int, Any] = {}

    def get(self, index: int) -> Any:
        if index in self._cache:
            return self._cache[index]

        from PIL import Image

        item = self.items[index]
        missing = [key for key in self.camera_keys if key not in item]
        if missing:
            available = ", ".join(sorted(item.keys()))
            raise ValueError(f"Missing camera key(s) {missing} for frame {index}. Available: {available}")

        if len(self.camera_keys) == 1:
            array = to_numpy_image(item[self.camera_keys[0]])
        else:
            array = stack_frame_images([item[key] for key in self.camera_keys])

        image = Image.fromarray(array)
        if len(self.camera_keys) > 1:
            image = image.resize(
                (self.view_width, self.view_height * len(self.camera_keys)),
                resample=get_resample(),
            )

        self._cache[index] = image
        return image


def load_episode_items(context: DatasetContext, episode_index: int) -> List[Dict[str, Any]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_kwargs: Dict[str, Any] = {
        "repo_id": context.repo_id,
        "episodes": [episode_index],
        "force_cache_sync": False,
    }
    if context.dataset_root is not None:
        dataset_kwargs["root"] = context.dataset_root
    dataset = LeRobotDataset(**dataset_kwargs)
    return [dataset[index] for index in range(len(dataset))]


def build_recent_context_indices(
    timestep: int,
    frame_subsample: int,
    recent_frames_length: int,
) -> List[int]:
    start_frame_offset = timestep % frame_subsample
    all_context_indices = list(range(start_frame_offset, timestep + 1, frame_subsample))
    return all_context_indices[-recent_frames_length:]


def effective_merge_distance_raw_frames(
    merge_distance: int,
    frame_subsample: int,
) -> int:
    """Convert merge distance from subsampled steps to raw frame units."""
    if frame_subsample <= 0:
        raise ValueError("frame_subsample must be positive.")
    if merge_distance < 0:
        raise ValueError("merge_distance must be non-negative.")
    return int(merge_distance) * int(frame_subsample)


def map_relative_positions_to_absolute(
    relative_positions: Sequence[int],
    context_indices: Sequence[int],
) -> Tuple[List[int], List[int]]:
    absolute_indices: List[int] = []
    invalid_positions: List[int] = []

    for position in relative_positions:
        relative_index = position - 1
        if 0 <= relative_index < len(context_indices):
            absolute_indices.append(int(context_indices[relative_index]))
        else:
            invalid_positions.append(position)

    return absolute_indices, invalid_positions


def validate_rollout_config(config: RolloutConfig) -> None:
    model_path = Path(config.model_path)
    if not model_path.exists():
        raise ValueError(f"--model-path does not exist: {model_path}")

    processor_path = Path(config.processor_path) if config.processor_path else model_path
    if not processor_path.exists():
        raise ValueError(f"--processor-path does not exist: {processor_path}")
    if config.frame_subsample <= 0:
        raise ValueError("--frame-subsample must be positive.")
    if config.recent_frames_length <= 0:
        raise ValueError("--recent-frames-length must be positive.")
    if config.memory_length < 0:
        raise ValueError("--memory-length must be non-negative.")
    if config.prediction_horizon < 0:
        raise ValueError("--prediction-horizon must be non-negative.")
    if config.merge_distance < 0:
        raise ValueError("--merge-distance must be non-negative.")
    if config.camera_layout_config and (config.camera_key or config.camera_keys):
        raise ValueError("Use only one of --camera-layout-config or --camera-key/--camera-keys.")
    if config.view_width is not None and config.view_width <= 0:
        raise ValueError("--view-width must be positive.")
    if config.view_height is not None and config.view_height <= 0:
        raise ValueError("--view-height must be positive.")
    load_camera_layout(config.camera_layout_config)

    if config.lerobot_path is not None:
        lerobot_path = Path(config.lerobot_path)
        if not lerobot_path.exists():
            raise ValueError(f"--lerobot-path does not exist: {lerobot_path}")
        if not (lerobot_path / "meta" / "info.json").is_file():
            raise ValueError(f"--lerobot-path is not a LeRobot dataset root: {lerobot_path}")

    if config.system_role not in {"system", "assistant"}:
        raise ValueError(f"--system-role must be 'system' or 'assistant', got: {config.system_role}")


def make_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [make_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: make_serializable(item) for key, item in value.items()}
    return value


def evaluate_rollout(
    config: RolloutConfig,
    predictor: Optional[StructuredPredictor] = None,
) -> RolloutSummary:
    from tqdm import tqdm

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_rollout_config(config)
    context = load_dataset_context(config.lerobot_path, config.repo_id)
    selected_camera_keys, view_width, view_height = resolve_camera_settings(context, config)
    raw_merge_distance = effective_merge_distance_raw_frames(
        config.merge_distance,
        config.frame_subsample,
    )

    episode_indices = context.episode_indices
    if config.episode_indices:
        requested = set(config.episode_indices)
        episode_indices = [index for index in episode_indices if index in requested]
    if config.max_episodes is not None:
        episode_indices = episode_indices[: config.max_episodes]

    if predictor is None:
        predictor = QwenStructuredPredictor(
            config.model_path,
            processor_path=config.processor_path,
            system_role=config.system_role,
            device=config.device,
            dtype=config.dtype,
            attn_implementation=config.attn_implementation,
            max_new_tokens=config.max_new_tokens,
        )

    overall_total = 0
    overall_raw_correct = 0
    overall_normalized_correct = 0
    overall_parse_failures = 0
    label_totals: Counter[str] = Counter()
    label_raw_correct: Counter[str] = Counter()
    label_normalized_correct: Counter[str] = Counter()
    episode_summaries: List[EpisodeSummary] = []

    predictions_path = output_dir / "predictions.jsonl"
    run_started_at = time.time()

    with predictions_path.open("w", encoding="utf-8") as predictions_fp:
        for episode_index in episode_indices:
            items = load_episode_items(context, episode_index)
            if not items:
                continue

            labels = [extract_subtask_label(item, context.subtask_map) for item in items]
            instructions = [
                extract_instruction(item, context.task_map, config.high_level_instruction)
                for item in items
            ]
            frame_cache = EpisodeFrameCache(
                items,
                selected_camera_keys,
                view_width,
                view_height,
            )
            memory = EpisodicMemory(
                merge_distance=raw_merge_distance,
                memory_length=config.memory_length,
            )

            episode_total = 0
            episode_raw_correct = 0
            episode_normalized_correct = 0
            episode_parse_failures = 0

            iterator = tqdm(
                range(len(items)),
                desc=f"Episode {episode_index:03d}",
                leave=False,
            )

            for timestep in iterator:
                context_indices = build_recent_context_indices(
                    timestep,
                    config.frame_subsample,
                    config.recent_frames_length,
                )
                memory_indices_before = memory.visible_indices(context_indices)
                prompt = build_human_prompt(
                    instruction=instructions[timestep],
                    memory_count=len(memory_indices_before),
                    recent_count=len(context_indices),
                )
                images = [
                    frame_cache.get(index)
                    for index in memory_indices_before + context_indices
                ]
                prediction = predictor.predict(prompt, images)

                target_index = compute_target_index(
                    timestep=timestep,
                    total_steps=len(labels),
                    frame_subsample=config.frame_subsample,
                    prediction_horizon=config.prediction_horizon,
                )
                target_label = labels[target_index]
                normalized_target = normalize_subtask_label(target_label)
                normalized_prediction = normalize_subtask_label(prediction.current_subtask)

                raw_correct = bool(prediction.parse_ok and prediction.current_subtask == target_label)
                normalized_correct = bool(prediction.parse_ok and normalized_prediction == normalized_target)

                mapped_keyframes: List[int] = []
                invalid_positions: List[int] = []
                if prediction.parse_ok:
                    mapped_keyframes, invalid_positions = map_relative_positions_to_absolute(
                        prediction.keyframe_positions,
                        context_indices,
                    )
                    memory.add_candidates(mapped_keyframes)
                else:
                    episode_parse_failures += 1
                    overall_parse_failures += 1

                memory_indices_after = memory.visible_indices(context_indices)

                row = {
                    "episode_index": episode_index,
                    "timestep": timestep,
                    "target_timestep": target_index,
                    "instruction": instructions[timestep],
                    "predicted_subtask": prediction.current_subtask,
                    "target_subtask": target_label,
                    "raw_correct": raw_correct,
                    "normalized_correct": normalized_correct,
                    "parse_ok": prediction.parse_ok,
                    "parse_error": prediction.parse_error,
                    "predicted_keyframe_positions": prediction.keyframe_positions,
                    "mapped_keyframe_indices": mapped_keyframes,
                    "invalid_keyframe_positions": invalid_positions,
                    "context_indices": context_indices,
                    "memory_indices_before": memory_indices_before,
                    "memory_indices_after": memory_indices_after,
                    "all_candidate_indices": memory.all_candidates(),
                    "raw_text": prediction.raw_text if config.save_raw_responses else None,
                }
                predictions_fp.write(json.dumps(row, ensure_ascii=False) + "\n")

                overall_total += 1
                episode_total += 1
                label_totals[target_label] += 1
                if raw_correct:
                    overall_raw_correct += 1
                    episode_raw_correct += 1
                    label_raw_correct[target_label] += 1
                if normalized_correct:
                    overall_normalized_correct += 1
                    episode_normalized_correct += 1
                    label_normalized_correct[target_label] += 1

                if hasattr(iterator, "set_postfix") and overall_total > 0:
                    iterator.set_postfix(
                        raw_acc=f"{overall_raw_correct / overall_total:.3f}",
                        parse_fail=f"{overall_parse_failures / overall_total:.3f}",
                    )

            episode_summaries.append(
                EpisodeSummary(
                    episode_index=episode_index,
                    timesteps=episode_total,
                    parse_failures=episode_parse_failures,
                    raw_accuracy=(episode_raw_correct / episode_total) if episode_total else 0.0,
                    normalized_accuracy=(
                        episode_normalized_correct / episode_total
                    )
                    if episode_total
                    else 0.0,
                )
            )

    summary = RolloutSummary(
        total_examples=overall_total,
        raw_accuracy=(overall_raw_correct / overall_total) if overall_total else 0.0,
        normalized_accuracy=(overall_normalized_correct / overall_total) if overall_total else 0.0,
        parse_failure_rate=(overall_parse_failures / overall_total) if overall_total else 0.0,
        episode_summaries=episode_summaries,
    )

    label_metrics = {}
    for label, total in sorted(label_totals.items()):
        label_metrics[label] = {
            "support": total,
            "raw_accuracy": (label_raw_correct[label] / total) if total else 0.0,
            "normalized_accuracy": (label_normalized_correct[label] / total) if total else 0.0,
        }

    serialized_config = make_serializable(asdict(config))
    serialized_config["camera_keys"] = selected_camera_keys
    serialized_config["view_width"] = view_width
    serialized_config["view_height"] = view_height

    summary_payload = {
        "config": serialized_config,
        "dataset": {
            "repo_id": context.repo_id,
            "lerobot_path": str(context.dataset_root) if context.dataset_root else None,
            "camera_keys": selected_camera_keys,
            "num_episodes": len(episode_indices),
        },
        "runtime": {
            "started_at_unix": run_started_at,
            "finished_at_unix": time.time(),
            "duration_seconds": time.time() - run_started_at,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "metrics": {
            "total_examples": summary.total_examples,
            "raw_accuracy": summary.raw_accuracy,
            "normalized_accuracy": summary.normalized_accuracy,
            "parse_failure_rate": summary.parse_failure_rate,
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "episodes.json").write_text(
        json.dumps([make_serializable(asdict(item)) for item in episode_summaries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "label_metrics.json").write_text(
        json.dumps(label_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logging.info(
        "Finished rollout eval on %d examples: raw_acc=%.4f normalized_acc=%.4f parse_failure_rate=%.4f",
        summary.total_examples,
        summary.raw_accuracy,
        summary.normalized_accuracy,
        summary.parse_failure_rate,
    )
    return summary
