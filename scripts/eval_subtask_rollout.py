#!/usr/bin/env python3
"""Offline deploy-style MemER rollout evaluation CLI."""

from __future__ import annotations

import argparse
from typing import List, Optional

from memer_eval.rollout import RolloutConfig, evaluate_rollout


def parse_episode_indices(value: Optional[str]) -> Optional[List[int]]:
    if value is None or not value.strip():
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> RolloutConfig:
    parser = argparse.ArgumentParser(
        description="Run deploy-style offline MemER subtask rollout evaluation on a LeRobot dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="Path to a Qwen3-VL checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for rollout metrics and predictions.")
    parser.add_argument(
        "--lerobot-path",
        default=None,
        help="Local LeRobot dataset root containing meta/info.json, tasks, and subtasks metadata.",
    )
    parser.add_argument("--repo-id", default=None, help="LeRobot repo id; inferred from --lerobot-path when omitted.")
    parser.add_argument(
        "--processor-path",
        default=None,
        help="Optional local processor bundle path. Defaults to --model-path, which must include the processor files.",
    )
    parser.add_argument(
        "--system-role",
        default="system",
        choices=["system", "assistant"],
        help="Role used for the MemER system prompt when building the Qwen chat message.",
    )
    parser.add_argument("--camera-key", default=None, help="Single camera to evaluate.")
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        default=None,
        help="Multiple camera views to vertically stack per timestep.",
    )
    parser.add_argument(
        "--camera-layout-config",
        default=None,
        help=(
            "JSON camera-layout config that defines the ordered camera_keys and optional per-view image size. "
            "Mutually exclusive with --camera-key and --camera-keys."
        ),
    )
    parser.add_argument(
        "--high-level-instruction",
        default=None,
        help="Override the dataset task text with one instruction for all frames.",
    )
    parser.add_argument("--frame-subsample", type=int, default=5, help="Recent-context stride in raw frames.")
    parser.add_argument("--recent-frames-length", type=int, default=8, help="Recent context length after subsampling.")
    parser.add_argument("--memory-length", type=int, default=8, help="Max exposed memory keyframes.")
    parser.add_argument("--prediction-horizon", type=int, default=2, help="Future target horizon in subsampled steps.")
    parser.add_argument(
        "--merge-distance",
        type=int,
        default=5,
        help="1D clustering distance in subsampled timesteps; internally scaled by --frame-subsample.",
    )
    parser.add_argument(
        "--view-width",
        type=int,
        default=None,
        help="Per-camera-view width in pixels. Defaults to --camera-layout-config when set, else 320.",
    )
    parser.add_argument(
        "--view-height",
        type=int,
        default=None,
        help="Per-camera-view height in pixels. Stacked height is view_height * num_cameras. Defaults to --camera-layout-config when set, else 180.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max generated tokens per prediction.")
    parser.add_argument("--device", default=None, help="Torch device, for example 'cuda' or 'cpu'.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model weight dtype.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Attention backend passed to transformers, for example 'flash_attention_2' or 'sdpa'.",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Evaluate only the first N episodes.")
    parser.add_argument(
        "--episode-indices",
        default=None,
        help="Comma-separated explicit episode indices to evaluate.",
    )
    parser.add_argument("--save-raw-responses", action="store_true", help="Persist raw model text in predictions.jsonl.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level.",
    )
    args = parser.parse_args()

    return RolloutConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        lerobot_path=args.lerobot_path,
        repo_id=args.repo_id,
        processor_path=args.processor_path,
        system_role=args.system_role,
        camera_key=args.camera_key,
        camera_keys=args.camera_keys,
        camera_layout_config=args.camera_layout_config,
        high_level_instruction=args.high_level_instruction,
        frame_subsample=args.frame_subsample,
        recent_frames_length=args.recent_frames_length,
        memory_length=args.memory_length,
        prediction_horizon=args.prediction_horizon,
        merge_distance=args.merge_distance,
        view_width=args.view_width,
        view_height=args.view_height,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_episodes=args.max_episodes,
        episode_indices=parse_episode_indices(args.episode_indices),
        save_raw_responses=args.save_raw_responses,
        log_level=args.log_level,
    )


def main() -> None:
    config = parse_args()
    evaluate_rollout(config)


if __name__ == "__main__":
    main()
