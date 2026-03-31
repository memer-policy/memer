import argparse
import io
import json
import logging
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **_: Any):
        return iterable

from memer_eval.camera_layout import load_camera_layout
from memer_eval.contract import DEFAULT_SYSTEM_PROMPT, PREDICTION_KEY, build_human_prompt
from memer_eval.dataset_metadata import (
    build_index_mapping_from_dataframe,
    extract_instruction,
    extract_subtask_label,
    resolve_dataset_args,
    scalar_to_int,
)
from memer_eval.utils import align_frames_with_subsampling


SUPPORTED_LEROBOT_MAJOR = 3
VALID_KEYFRAME_SELECT = {"first", "last", "both", "none"}
DEFAULT_VIEW_WIDTH = 320
DEFAULT_VIEW_HEIGHT = 180
DEFAULT_JPEG_QUALITY = 90


@dataclass
class DatasetContext:
    repo_id: str
    dataset_root: Optional[Path]
    info: Dict[str, Any]
    camera_keys: List[str]
    task_map: Dict[int, str]
    subtask_map: Dict[int, str]
    episode_indices: List[int]


@dataclass
class EpisodeJob:
    repo_id: str
    dataset_root: Optional[str]
    episode_index: int
    trajectory_name: str
    output_dir: str
    camera_keys: List[str]
    subtask_map: Dict[int, str]
    task_map: Dict[int, str]
    high_level_instruction: Optional[str]
    frame_subsample: int
    recent_frames_length: int
    keyframes_length: int
    prediction_horizon: int
    overwrite: bool
    default_keyframe_select: str
    keyframe_rules: List[Dict[str, Any]]
    view_width: int
    view_height: int
    jpeg_quality: int
    system_prompt: str
    prediction_key: str


def load_sft_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    with open(config_path, "r") as fp:
        return json.load(fp)


def parse_major_version(version_str: Optional[str]) -> Optional[int]:
    if not version_str:
        return None
    match = re.search(r"(\d+)", str(version_str))
    return int(match.group(1)) if match else None


def validate_codebase_version(codebase_version: Optional[str], strict: bool) -> None:
    major = parse_major_version(codebase_version)
    if major is None:
        logging.warning("Could not parse codebase_version=%s", codebase_version)
        return
    if major != SUPPORTED_LEROBOT_MAJOR:
        msg = (
            f"Dataset codebase_version={codebase_version} (major={major}) "
            f"is not supported. Expected v{SUPPORTED_LEROBOT_MAJOR}.x."
        )
        if strict:
            raise ValueError(msg)
        logging.warning(msg)


def validate_select(select: str) -> str:
    value = str(select).strip().lower()
    if value not in VALID_KEYFRAME_SELECT:
        raise ValueError(
            f"Invalid keyframe select mode '{value}'. "
            f"Expected one of: {sorted(VALID_KEYFRAME_SELECT)}"
        )
    return value


def validate_args(args: argparse.Namespace) -> None:
    if args.frame_subsample <= 0:
        raise ValueError("--frame_subsample must be positive.")
    if args.recent_frames_length <= 0:
        raise ValueError("--recent_frames_length must be positive.")
    if args.keyframes_length < 0:
        raise ValueError("--keyframes_length must be non-negative.")
    if args.prediction_horizon < 0:
        raise ValueError("--prediction_horizon must be non-negative.")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be positive.")
    if args.camera_key and args.camera_keys:
        raise ValueError("Use only one of --camera_key or --camera_keys.")
    if getattr(args, "camera_layout_config", None) and (args.camera_key or args.camera_keys):
        raise ValueError("Use only one of --camera_layout_config or --camera_key/--camera_keys.")
    if args.view_width is not None and args.view_width <= 0:
        raise ValueError("--view_width must be positive.")
    if args.view_height is not None and args.view_height <= 0:
        raise ValueError("--view_height must be positive.")
    if args.jpeg_quality is not None and (args.jpeg_quality <= 0 or args.jpeg_quality > 100):
        raise ValueError("--jpeg_quality must be in the range [1, 100].")


def normalize_rule(rule_item: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(rule_item, dict):
        raise ValueError(f"Rule entries must be objects, got: {type(rule_item)}")

    select = validate_select(rule_item.get("select", "none"))
    ignore_case = bool(rule_item.get("ignore_case", True))

    if "pattern" in rule_item:
        return {
            "type": "regex",
            "pattern": str(rule_item["pattern"]),
            "select": select,
            "ignore_case": ignore_case,
        }
    if "exact" in rule_item:
        return {
            "type": "exact",
            "exact": str(rule_item["exact"]),
            "select": select,
            "ignore_case": ignore_case,
        }
    if "label" in rule_item:
        return {
            "type": "exact",
            "exact": str(rule_item["label"]),
            "select": select,
            "ignore_case": ignore_case,
        }
    raise ValueError(
        "Each rule must provide one of: 'pattern', 'exact', or 'label'. "
        f"Found keys: {sorted(rule_item.keys())}"
    )


def load_keyframe_rules(
    keyframe_rule_file: Optional[str], default_select: str
) -> Tuple[str, List[Dict[str, Any]]]:
    default_select = validate_select(default_select)
    if not keyframe_rule_file:
        return default_select, []

    with open(keyframe_rule_file, "r") as fp:
        config = json.load(fp)

    if isinstance(config, dict):
        default_select = validate_select(config.get("default_select", default_select))
        rules_data = config.get("rules", [])
    elif isinstance(config, list):
        rules_data = config
    else:
        raise ValueError(
            "Invalid keyframe rule file. Use {'default_select': ..., 'rules': [...]} or a rules list."
        )

    return default_select, [normalize_rule(item) for item in rules_data]


def resolve_select_for_label(
    label: str, default_select: str, rules: Sequence[Dict[str, Any]]
) -> str:
    for rule in rules:
        if rule["type"] == "regex":
            flags = re.IGNORECASE if rule.get("ignore_case", True) else 0
            if re.search(rule["pattern"], label, flags=flags):
                return rule["select"]
            continue

        exact = str(rule["exact"])
        if rule.get("ignore_case", True):
            if label.lower() == exact.lower():
                return rule["select"]
        elif label == exact:
            return rule["select"]
    return default_select


def get_rule_based_keyframes(
    labels: Sequence[str], default_select: str, rules: Sequence[Dict[str, Any]]
) -> List[int]:
    keyframe_indices: List[int] = []
    index = 0

    while index < len(labels):
        label = labels[index]
        if label is None or str(label).strip() == "" or str(label) == "None":
            index += 1
            continue

        next_index = index + 1
        while next_index < len(labels) and labels[next_index] == label:
            next_index += 1

        select = resolve_select_for_label(str(label), default_select, rules)
        if select == "first":
            keyframe_indices.append(index)
        elif select == "last":
            keyframe_indices.append(next_index - 1)
        elif select == "both":
            keyframe_indices.extend([index, next_index - 1])

        index = next_index

    return list(dict.fromkeys(keyframe_indices))
def load_dataset_metadata(
    lerobot_path: Optional[str], repo_id: Optional[str], strict_version_check: bool
) -> DatasetContext:
    resolved_repo_id, dataset_root = resolve_dataset_args(lerobot_path, repo_id)
    meta = LeRobotDatasetMetadata(
        repo_id=resolved_repo_id,
        root=dataset_root,
        force_cache_sync=False,
    )

    info = meta.info
    validate_codebase_version(info.get("codebase_version"), strict_version_check)

    task_map = build_index_mapping_from_dataframe(getattr(meta, "tasks", None), "task_index")
    subtask_map = build_index_mapping_from_dataframe(getattr(meta, "subtasks", None), "subtask_index")
    if not subtask_map:
        raise ValueError(
            "No subtasks found in dataset metadata. Expected a LeRobot dataset with subtasks."
        )

    camera_keys = list(getattr(meta, "camera_keys", []) or [])
    if not camera_keys:
        features = info.get("features", {})
        camera_keys = [
            key
            for key, value in features.items()
            if isinstance(value, dict) and value.get("dtype") in {"image", "video"}
        ]
    if not camera_keys:
        raise ValueError("No image or video camera keys found in dataset metadata.")

    resolved_dataset_root = Path(meta.root).resolve()
    episode_indices = list(range(int(meta.total_episodes)))
    return DatasetContext(
        repo_id=resolved_repo_id,
        dataset_root=resolved_dataset_root,
        info=info,
        camera_keys=camera_keys,
        task_map=task_map,
        subtask_map=subtask_map,
        episode_indices=episode_indices,
    )


def warm_dataset_cache_for_workers(
    repo_id: str,
    dataset_root: Optional[Path],
    episode_indices: Sequence[int],
    use_repo_id_resolution: bool,
    num_workers: int,
) -> None:
    if not use_repo_id_resolution or dataset_root is None or num_workers <= 1 or not episode_indices:
        return

    logging.info(
        "Priming shared LeRobot cache under %s before spawning %d workers.",
        dataset_root,
        num_workers,
    )
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=list(episode_indices),
        force_cache_sync=False,
        download_videos=True,
    )
    _ = len(dataset)


def detect_default_camera_key(camera_keys: Sequence[str]) -> str:
    if not camera_keys:
        raise ValueError("No image or video camera keys found in dataset metadata.")
    if "exterior_image_1_left" in camera_keys:
        return "exterior_image_1_left"
    return str(camera_keys[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot subtasks into Qwen3-VL finetune annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lerobot_path",
        default=None,
        help="Local LeRobot dataset root. If omitted, LeRobot resolves data from cache via --repo_id.",
    )
    parser.add_argument(
        "--repo_id",
        default=None,
        help="Dataset identifier for LeRobot metadata/cache lookup. Inferred from --lerobot_path when omitted.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Writes train.json and media/ here.",
    )
    parser.add_argument(
        "--camera_key",
        default=None,
        help=(
            "Single camera view to export. Mutually exclusive with --camera_keys; "
            "defaults to exterior_image_1_left when available, else the first detected camera."
        ),
    )
    parser.add_argument(
        "--camera_keys",
        nargs="+",
        default=None,
        help="Multiple camera views to vertically stack into one JPEG per timestep. Mutually exclusive with --camera_key.",
    )
    parser.add_argument(
        "--camera_layout_config",
        default=None,
        help=(
            "JSON camera-layout config that defines the ordered camera_keys and optional per-view image size. "
            "Mutually exclusive with --camera_key and --camera_keys."
        ),
    )
    parser.add_argument(
        "--sft_config",
        default=None,
        help=(
            "Optional JSON overrides for the generated Qwen export contract "
            "(system_prompt, prediction_key)."
        ),
    )
    parser.add_argument(
        "--keyframe_rule_file",
        default=None,
        help="JSON rules that map subtask labels or regexes to memory keyframe selection.",
    )
    parser.add_argument(
        "--high_level_instruction",
        default=None,
        help="Use one task instruction for every sample instead of dataset task/task_index fields.",
    )
    parser.add_argument(
        "--frame_subsample",
        type=int,
        default=5,
        help="Keep every Nth frame in recent context; prediction horizon uses the same stride.",
    )
    parser.add_argument(
        "--recent_frames_length",
        type=int,
        default=8,
        help="Number of subsampled frames kept in the rolling recent context window.",
    )
    parser.add_argument(
        "--keyframes_length",
        type=int,
        default=8,
        help="Maximum number of older memory keyframes prepended before recent frames. Use 0 to disable memory images.",
    )
    parser.add_argument(
        "--prediction_horizon",
        type=int,
        default=2,
        help="Predict the subtask at t + horizon * frame_subsample.",
    )
    parser.add_argument(
        "--view_width",
        type=int,
        default=None,
        help=(
            "Per-camera-view width in pixels. Each camera is resized to this width. "
            "Defaults to camera_layout_config.view_width when set, else 320."
        ),
    )
    parser.add_argument(
        "--view_height",
        type=int,
        default=None,
        help=(
            "Per-camera-view height in pixels. Each camera is resized to this height; "
            "the stacked image height is view_height * number_of_cameras. "
            "Defaults to camera_layout_config.view_height when set, else 180."
        ),
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=None,
        help="JPEG quality for exported frame images. Defaults to 90.",
    )
    parser.add_argument(
        "--default_keyframe_select",
        choices=sorted(VALID_KEYFRAME_SELECT),
        default="last",
        help="Fallback keyframe pick for each contiguous subtask segment when no rule matches.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Only process the first N episodes from dataset order.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="Episode-level worker processes. Use 1 for easier debugging.",
    )
    parser.add_argument(
        "--strict_version_check",
        action="store_true",
        help="Fail instead of warning when dataset codebase_version is not LeRobot v3.x.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_dir first and re-export all frames.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging verbosity.",
    )
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> List[EpisodeJob]:
    validate_args(args)
    sft_config = load_sft_config(args.sft_config)
    camera_layout = load_camera_layout(args.camera_layout_config)

    context = load_dataset_metadata(args.lerobot_path, args.repo_id, args.strict_version_check)
    if camera_layout:
        selected_camera_keys = list(camera_layout.camera_keys)
    elif args.camera_keys:
        selected_camera_keys = list(args.camera_keys)
    else:
        selected_camera_keys = [args.camera_key or detect_default_camera_key(context.camera_keys)]

    missing_camera_keys = [key for key in selected_camera_keys if key not in context.camera_keys]
    if missing_camera_keys:
        available = ", ".join(context.camera_keys)
        missing = ", ".join(missing_camera_keys)
        raise ValueError(
            f"Camera key(s) '{missing}' not found. Available keys: {available}"
        )

    view_width = (
        args.view_width if args.view_width is not None
        else (
            camera_layout.view_width
            if camera_layout and camera_layout.view_width is not None
            else DEFAULT_VIEW_WIDTH
        )
    )
    view_height = (
        args.view_height if args.view_height is not None
        else (
            camera_layout.view_height
            if camera_layout and camera_layout.view_height is not None
            else DEFAULT_VIEW_HEIGHT
        )
    )

    jpeg_quality = (
        args.jpeg_quality if args.jpeg_quality is not None
        else DEFAULT_JPEG_QUALITY
    )
    system_prompt = sft_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    prediction_key = sft_config.get("prediction_key", PREDICTION_KEY)

    default_select, rules = load_keyframe_rules(
        args.keyframe_rule_file,
        args.default_keyframe_select,
    )

    episode_indices = context.episode_indices
    if args.max_episodes is not None:
        episode_indices = episode_indices[: args.max_episodes]

    warm_dataset_cache_for_workers(
        repo_id=context.repo_id,
        dataset_root=context.dataset_root,
        episode_indices=episode_indices,
        use_repo_id_resolution=args.lerobot_path is None,
        num_workers=args.num_workers,
    )

    jobs: List[EpisodeJob] = []
    for episode_index in episode_indices:
        jobs.append(
            EpisodeJob(
                repo_id=context.repo_id,
                dataset_root=str(context.dataset_root) if context.dataset_root else None,
                episode_index=episode_index,
                trajectory_name=f"episode_{episode_index:06d}",
                output_dir=os.path.abspath(args.output_dir),
                camera_keys=selected_camera_keys,
                subtask_map=context.subtask_map,
                task_map=context.task_map,
                high_level_instruction=args.high_level_instruction,
                frame_subsample=args.frame_subsample,
                recent_frames_length=args.recent_frames_length,
                keyframes_length=args.keyframes_length,
                prediction_horizon=args.prediction_horizon,
                overwrite=args.overwrite,
                default_keyframe_select=default_select,
                keyframe_rules=rules,
                view_width=view_width,
                view_height=view_height,
                jpeg_quality=jpeg_quality,
                system_prompt=system_prompt,
                prediction_key=prediction_key,
            )
        )
    return jobs


def get_lerobot_dataset(job: EpisodeJob):
    dataset_kwargs: Dict[str, Any] = {
        "repo_id": job.repo_id,
        "episodes": [job.episode_index],
        "force_cache_sync": False,
    }
    if job.dataset_root:
        dataset_kwargs["root"] = Path(job.dataset_root)
    return LeRobotDataset(**dataset_kwargs)

def to_numpy_image(value: Any) -> np.ndarray:
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
        pass
    elif array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        pass
    else:
        raise ValueError(f"Unsupported image array shape: {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def load_pillow_image(image_source: Any):
    if isinstance(image_source, (str, os.PathLike)):
        image = Image.open(image_source)
    else:
        image = Image.open(io.BytesIO(image_source))
    return image


def get_pillow_resample():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def array_to_rgb(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim == 3 and array.shape[-1] == 1:
        return np.repeat(array, 3, axis=2)
    if array.ndim == 3 and array.shape[-1] == 4:
        return array[:, :, :3]
    if array.ndim == 3 and array.shape[-1] == 3:
        return array
    raise ValueError(f"Unsupported image shape: {array.shape}")


def maybe_resize_image_array(
    array: np.ndarray,
    width: Optional[int],
    height: Optional[int],
) -> np.ndarray:
    if width is None or height is None:
        return array

    pil_image = Image.fromarray(array_to_rgb(array))
    resized = pil_image.resize((width, height), resample=get_pillow_resample())
    return np.asarray(resized)


def write_jpeg_image(
    image_value: Any,
    output_path: Path,
    quality: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    if hasattr(image_value, "save") and callable(image_value.save):
        image_array = np.asarray(image_value.convert("RGB"))
    else:
        image_array = to_numpy_image(image_value)

    image_array = maybe_resize_image_array(image_array, width, height)
    image = Image.fromarray(array_to_rgb(image_array))
    image.save(output_path, format="JPEG", quality=quality, optimize=True)


def resolve_source_path(path_value: Any, dataset_root: Optional[str]) -> Path:
    source_path = Path(path_value)
    if source_path.is_absolute() or dataset_root is None:
        return source_path
    return Path(dataset_root) / source_path


def normalize_image_for_stacking(array: np.ndarray) -> np.ndarray:
    return array_to_rgb(array)


def load_image_array(
    image_value: Any,
    dataset_root: Optional[str],
) -> np.ndarray:
    if isinstance(image_value, dict):
        image_bytes = image_value.get("bytes")
        image_path = image_value.get("path")
        if image_bytes is not None:
            image_source = image_bytes.tobytes() if hasattr(image_bytes, "tobytes") else bytes(image_bytes)
            with load_pillow_image(image_source) as image:
                return normalize_image_for_stacking(np.asarray(image.convert("RGB")))
        if image_path:
            with load_pillow_image(resolve_source_path(image_path, dataset_root)) as image:
                return normalize_image_for_stacking(np.asarray(image.convert("RGB")))
        raise ValueError("Image dict did not contain 'bytes' or 'path'.")

    if isinstance(image_value, (str, os.PathLike)):
        with load_pillow_image(resolve_source_path(image_value, dataset_root)) as image:
            return normalize_image_for_stacking(np.asarray(image.convert("RGB")))

    if isinstance(image_value, (bytes, bytearray)):
        with load_pillow_image(bytes(image_value)) as image:
            return normalize_image_for_stacking(np.asarray(image.convert("RGB")))

    return normalize_image_for_stacking(to_numpy_image(image_value))


def stack_frame_images(image_values: Sequence[Any], dataset_root: Optional[str]) -> np.ndarray:
    image_arrays = [load_image_array(image_value, dataset_root) for image_value in image_values]
    max_width = max(array.shape[1] for array in image_arrays)

    padded_arrays = []
    for array in image_arrays:
        height, width, channels = array.shape
        padded = np.zeros((height, max_width, channels), dtype=np.uint8)
        padded[:height, :width, :channels] = array
        padded_arrays.append(padded)

    return np.concatenate(padded_arrays, axis=0)


def export_frame_image(
    image_value: Any,
    output_path: Path,
    dataset_root: Optional[str],
    quality: int,
) -> None:
    image_array = load_image_array(image_value, dataset_root)
    write_jpeg_image(image_array, output_path, quality=quality)


def build_frame_relative_path(trajectory_name: str, frame_index: int) -> str:
    return f"media/{trajectory_name}/frame_{frame_index:06d}.jpg"


def export_episode_frames(items: Sequence[Dict[str, Any]], job: EpisodeJob) -> None:
    frames_dir = Path(job.output_dir) / "media" / job.trajectory_name
    frames_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, item in enumerate(items):
        missing_keys = [camera_key for camera_key in job.camera_keys if camera_key not in item]
        if missing_keys:
            available = ", ".join(sorted(item.keys()))
            missing = ", ".join(missing_keys)
            raise ValueError(
                f"Camera key(s) '{missing}' not found in episode {job.episode_index}. "
                f"Available keys: {available}"
            )

        output_path = frames_dir / f"frame_{frame_index:06d}.jpg"
        if not job.overwrite and output_path.exists():
            continue
        if len(job.camera_keys) == 1:
            export_frame_image(
                item[job.camera_keys[0]],
                output_path,
                job.dataset_root,
                quality=job.jpeg_quality,
            )
            continue

        stacked_image = stack_frame_images(
            [item[camera_key] for camera_key in job.camera_keys],
            job.dataset_root,
        )
        write_jpeg_image(
            stacked_image,
            output_path,
            quality=job.jpeg_quality,
            width=job.view_width,
            height=job.view_height * len(job.camera_keys),
        )

def build_qwen_example(
    labels: Sequence[str],
    instructions: Sequence[str],
    all_keyframe_indices: Sequence[int],
    job: EpisodeJob,
    timestep: int,
) -> Dict[str, Any]:
    start_frame_offset = timestep % job.frame_subsample
    all_context_indices = list(range(start_frame_offset, timestep + 1, job.frame_subsample))
    recent_frame_indices = all_context_indices[-job.recent_frames_length :]
    start_context_idx = recent_frame_indices[0]

    candidate_keyframes = align_frames_with_subsampling(
        list(all_keyframe_indices),
        start_context_idx,
        timestep + 1,
        job.frame_subsample,
    )
    relative_positions = [recent_frame_indices.index(idx) + 1 for idx in candidate_keyframes]

    memory_keyframes = align_frames_with_subsampling(
        list(all_keyframe_indices),
        start_frame_offset,
        start_context_idx,
        job.frame_subsample,
    )
    if job.keyframes_length > 0:
        memory_keyframes = memory_keyframes[-job.keyframes_length :]
    else:
        memory_keyframes = []

    memory_paths = [
        build_frame_relative_path(job.trajectory_name, frame_index)
        for frame_index in memory_keyframes
    ]
    recent_paths = [
        build_frame_relative_path(job.trajectory_name, frame_index)
        for frame_index in recent_frame_indices
    ]

    target_idx = min(
        timestep + job.prediction_horizon * job.frame_subsample,
        len(labels) - 1,
    )
    answer = {
        job.prediction_key: labels[target_idx],
        "keyframe_positions": relative_positions,
    }
    system_prompt = job.system_prompt

    return {
        "id": f"{job.trajectory_name}_t{timestep:06d}",
        "image": memory_paths + recent_paths,
        "conversations": [
            {"from": "system", "value": system_prompt},
            {
                "from": "human",
                "value": build_human_prompt(
                    instruction=instructions[timestep],
                    memory_count=len(memory_paths),
                    recent_count=len(recent_paths),
                ),
            },
            {"from": "gpt", "value": json.dumps(answer)},
        ],
        "metadata": {
            "repo_id": job.repo_id,
            "episode_index": job.episode_index,
            "timestep": timestep,
            "camera_keys": job.camera_keys,
            "memory_indices": memory_keyframes,
            "context_indices": recent_frame_indices,
            "system_prompt": system_prompt,
            "instruction": instructions[timestep],
            "answer": answer,
        },
    }


def load_episode_items(job: EpisodeJob) -> List[Dict[str, Any]]:
    dataset = get_lerobot_dataset(job)
    return [dataset[index] for index in range(len(dataset))]


def process_episode(job: EpisodeJob) -> List[Dict[str, Any]]:
    items = load_episode_items(job)
    if not items:
        return []

    export_episode_frames(items, job)

    labels = [extract_subtask_label(item, job.subtask_map) for item in items]
    instructions = [extract_instruction(item, job.task_map, job.high_level_instruction) for item in items]
    all_keyframe_indices = get_rule_based_keyframes(
        labels,
        job.default_keyframe_select,
        job.keyframe_rules,
    )

    examples = [
        build_qwen_example(
            labels=labels,
            instructions=instructions,
            all_keyframe_indices=all_keyframe_indices,
            job=job,
            timestep=timestep,
        )
        for timestep in range(len(labels))
    ]
    logging.info("Generated %d SFT entries for %s", len(examples), job.trajectory_name)
    return examples


def iter_episode_examples(jobs: Sequence[EpisodeJob], num_workers: int) -> Iterable[List[Dict[str, Any]]]:
    if num_workers == 1:
        iterator = (process_episode(job) for job in jobs)
        yield from tqdm(iterator, total=len(jobs), desc="Generating Qwen entries")
        return

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        yield from tqdm(
            executor.map(process_episode, jobs),
            total=len(jobs),
            desc="Generating Qwen entries",
        )


def write_train_json(output_dir: str, examples: Sequence[Dict[str, Any]]) -> Path:
    train_path = Path(output_dir) / "train.json"
    with train_path.open("w") as fp:
        json.dump(list(examples), fp, indent=2)
    return train_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    output_dir = os.path.abspath(args.output_dir)
    if args.overwrite:
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    jobs = build_jobs(args)
    logging.info("Processing %d episodes with %d workers.", len(jobs), args.num_workers)

    all_examples: List[Dict[str, Any]] = []
    for episode_examples in iter_episode_examples(jobs, args.num_workers):
        all_examples.extend(episode_examples)

    train_path = write_train_json(output_dir, all_examples)
    logging.info("Wrote %d Qwen entries to %s", len(all_examples), train_path)


if __name__ == "__main__":
    main()
