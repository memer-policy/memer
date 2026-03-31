"""Shared dataset-metadata helpers used by export and rollout eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def scalar_to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    return int(value)


def build_index_mapping_from_dataframe(df: Any, index_column: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if df is None or not hasattr(df, "iterrows"):
        return mapping

    label_column = "__index_level_0__" if hasattr(df, "columns") and "__index_level_0__" in df.columns else None
    for label, row in df.iterrows():
        index = row[index_column] if index_column in row else None
        if index is not None:
            resolved_label = row[label_column] if label_column and label_column in row else label
            mapping[int(index)] = str(resolved_label)
    return mapping


def extract_instruction(
    item: Dict[str, Any],
    task_map: Dict[int, str],
    high_level_instruction: Optional[str] = None,
) -> str:
    if high_level_instruction:
        return high_level_instruction

    task = item.get("task")
    task_index = scalar_to_int(item.get("task_index"))
    if task_index is None:
        task_index = scalar_to_int(task)
    if task_index is not None and task_index in task_map:
        return task_map[task_index]

    if task is not None and str(task).strip():
        return str(task)

    return "What subtask should the robot execute next?"


def extract_subtask_label(item: Dict[str, Any], subtask_map: Dict[int, str]) -> str:
    label = item.get("subtask")
    if label is not None and str(label).strip():
        return str(label)

    subtask_index = scalar_to_int(item.get("subtask_index"))
    if subtask_index is not None and subtask_index in subtask_map:
        return subtask_map[subtask_index]

    raise ValueError("Missing subtask label for a rollout frame.")


def resolve_dataset_args(
    lerobot_path: Optional[str], repo_id: Optional[str]
) -> Tuple[str, Optional[Path]]:
    resolved_path = Path(lerobot_path).resolve() if lerobot_path else None
    if resolved_path is None and not repo_id:
        raise ValueError("Provide at least one of --lerobot-path/--lerobot_path or --repo-id/--repo_id.")
    if repo_id:
        return repo_id, resolved_path
    assert resolved_path is not None
    return resolved_path.name, resolved_path
