"""Camera layout config helpers for stacked multi-view inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass(frozen=True)
class CameraLayout:
    camera_keys: List[str]
    view_width: Optional[int] = None
    view_height: Optional[int] = None
    notes: Optional[str] = None


def load_camera_layout(config_path: Optional[str]) -> Optional[CameraLayout]:
    if not config_path:
        return None

    path = Path(config_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Camera layout config must be a JSON object: {path}")

    camera_keys = payload.get("camera_keys")
    if not isinstance(camera_keys, list) or not camera_keys:
        raise ValueError(f"Camera layout config must define a non-empty camera_keys list: {path}")

    normalized_camera_keys = [str(key) for key in camera_keys]
    width = payload.get("view_width")
    height = payload.get("view_height")
    if width is not None and int(width) <= 0:
        raise ValueError(f"view_width must be positive in {path}")
    if height is not None and int(height) <= 0:
        raise ValueError(f"view_height must be positive in {path}")

    return CameraLayout(
        camera_keys=normalized_camera_keys,
        view_width=int(width) if width is not None else None,
        view_height=int(height) if height is not None else None,
        notes=str(payload["notes"]) if "notes" in payload and payload["notes"] is not None else None,
    )
