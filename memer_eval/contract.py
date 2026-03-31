"""Prompt and output contract for MemER rollout evaluation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

PREDICTION_KEY = "current_subtask"
PREDICTION_ALIASES = (PREDICTION_KEY, "current_primitive")
KEYFRAME_POSITIONS_KEY = "keyframe_positions"
DEFAULT_SYSTEM_PROMPT = (
    "You are a robot program that predicts actions.\n"
    "The video input from the egocentric camera shows the most recent actions the robot has executed. "
    "The images are selected frames of particular importance from all the actions the robot has executed so far. "
    "Based on these, output the current subtask the robot should execute and nothing else.\n\n"
    "Return a JSON with:\n"
    "- current_subtask: the action that should be executed at the current timestep\n"
    "- keyframe_positions: list of frame positions (1-indexed) from the video input where actions change\n"
)

_IMAGE_PATTERN = re.compile(r"(<image>)")


def build_system_prompt() -> str:
    """Build the paper-aligned system prompt for MemER."""
    return DEFAULT_SYSTEM_PROMPT


def build_user_prompt(instruction: str, memory_count: int, recent_count: int) -> str:
    """Build the paper-aligned user prompt for MemER."""
    lines = [f"Task: {instruction}"]

    if memory_count > 0:
        lines.append(
            "Here are the selected frames from the entirety of the full video that are of particular importance:"
        )
        lines.extend(["<image>"] * memory_count)

    lines.append("Here is a video of the most recent actions the robot has executed:")
    lines.extend(["<image>"] * recent_count)
    return "\n".join(lines)


def build_human_prompt(instruction: str, memory_count: int, recent_count: int) -> str:
    """Backward-compatible alias for the user prompt portion of the chat contract."""
    return build_user_prompt(
        instruction=instruction,
        memory_count=memory_count,
        recent_count=recent_count,
    )


def split_prompt_on_images(prompt: str) -> List[str]:
    """Split a user prompt into text segments around <image> placeholders."""
    return _IMAGE_PATTERN.split(prompt)


def build_user_message(
    prompt: str,
    images: Sequence[Any],
    *,
    system_prompt: str | None = None,
    system_role: str = "system",
) -> List[Dict[str, Any]]:
    """Create paper-aligned system+user chat messages with interleaved images."""
    if system_role not in {"system", "assistant"}:
        raise ValueError(f"Unsupported system_role: {system_role}")

    parts = split_prompt_on_images(prompt)
    content: List[Dict[str, Any]] = []
    image_index = 0

    for part in parts:
        if part == "<image>":
            if image_index >= len(images):
                raise ValueError("Prompt contains more <image> placeholders than provided images.")
            content.append({"type": "image", "image": images[image_index]})
            image_index += 1
            continue

        if part:
            content.append({"type": "text", "text": part})

    if image_index != len(images):
        raise ValueError("Provided more images than the prompt consumes.")

    return [
        {"role": system_role, "content": [{"text": system_prompt or build_system_prompt()}]},
        {"role": "user", "content": content},
    ]


def normalize_subtask_label(value: Any) -> str:
    """Normalize whitespace for exact-match comparisons."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def compute_target_index(
    timestep: int,
    total_steps: int,
    frame_subsample: int,
    prediction_horizon: int,
) -> int:
    """Return the shifted target index used during training."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    target = timestep + prediction_horizon * frame_subsample
    return min(target, total_steps - 1)
