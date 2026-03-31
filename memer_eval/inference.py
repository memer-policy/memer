"""Model loading and structured prediction parsing for MemER rollout eval."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .contract import (
    KEYFRAME_POSITIONS_KEY,
    PREDICTION_ALIASES,
    build_user_message,
)


@dataclass
class ModelPrediction:
    """Structured model output for one rollout timestep."""

    current_subtask: Optional[str]
    keyframe_positions: list[int]
    raw_text: str
    parse_ok: bool
    parse_error: Optional[str] = None


class StructuredPredictor(ABC):
    """Minimal interface for one-step structured prediction."""

    @abstractmethod
    def predict(self, prompt: str, images: Sequence[Any]) -> ModelPrediction:
        raise NotImplementedError


def _extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first balanced JSON object found in text."""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_prediction_text(text: str) -> ModelPrediction:
    """Parse a structured MemER prediction from raw model output text."""
    stripped = text.strip()
    candidates = [stripped]

    if "```" in stripped:
        fence_parts = stripped.split("```")
        for part in fence_parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith("json"):
                part = part[4:].strip()
            candidates.append(part)

    json_fragment = _extract_first_json_object(stripped)
    if json_fragment is not None:
        candidates.append(json_fragment)

    parse_errors = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            parse_errors.append(str(exc))
            continue

        if not isinstance(payload, dict):
            parse_errors.append("Parsed JSON is not an object.")
            continue

        label = None
        for key in PREDICTION_ALIASES:
            if key in payload:
                label = payload[key]
                break

        if label is None:
            parse_errors.append("Prediction JSON did not contain a current_subtask/current_primitive field.")
            continue

        positions = payload.get(KEYFRAME_POSITIONS_KEY, [])
        if positions is None:
            positions = []
        if not isinstance(positions, list):
            parse_errors.append("keyframe_positions must be a list.")
            continue

        normalized_positions: list[int] = []
        for position in positions:
            normalized_position = _coerce_keyframe_position(position)
            if normalized_position is None:
                parse_errors.append(f"Invalid keyframe position: {position!r}")
                normalized_positions = []
                break
            normalized_positions.append(normalized_position)
        else:
            return ModelPrediction(
                current_subtask=str(label),
                keyframe_positions=normalized_positions,
                raw_text=text,
                parse_ok=True,
            )

    error = parse_errors[-1] if parse_errors else "No JSON object found in model output."
    return ModelPrediction(
        current_subtask=None,
        keyframe_positions=[],
        raw_text=text,
        parse_ok=False,
        parse_error=error,
    )


def _coerce_keyframe_position(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("-"):
            digits = stripped[1:]
        else:
            digits = stripped
        if digits.isdigit():
            return int(stripped)
    return None


class QwenStructuredPredictor(StructuredPredictor):
    """Load a Qwen3-VL checkpoint and emit structured one-step predictions."""

    def __init__(
        self,
        model_path: str,
        *,
        processor_path: Optional[str] = None,
        system_role: str = "system",
        device: Optional[str] = None,
        dtype: str = "auto",
        attn_implementation: Optional[str] = None,
        max_new_tokens: int = 128,
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.max_new_tokens = int(max_new_tokens)
        self.processor_path = processor_path or self.model_path
        self.system_role = system_role

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from qwen_vl_utils import process_vision_info

        self._torch = torch
        self._process_vision_info = process_vision_info

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = resolved_device
        if resolved_device == "auto":
            device_map: Any = "auto"
        elif resolved_device.startswith("cuda"):
            device_map = {"": resolved_device}
        else:
            device_map = None

        torch_dtype: Any
        if dtype == "auto":
            torch_dtype = "auto"
        else:
            if not hasattr(torch, dtype):
                raise ValueError(f"Unsupported dtype '{dtype}'.")
            torch_dtype = getattr(torch, dtype)

        if attn_implementation is None:
            use_flash_attention = resolved_device == "auto" or resolved_device.startswith("cuda")
            attn_implementation = "flash_attention_2" if use_flash_attention and torch.cuda.is_available() else "sdpa"

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            trust_remote_code=True,
        )
        if resolved_device != "auto" and device_map is None:
            self.model = self.model.to(resolved_device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.processor_path,
            trust_remote_code=True,
        )

    def predict(self, prompt: str, images: Sequence[Any]) -> ModelPrediction:
        messages = build_user_message(
            prompt,
            images,
            system_role=self.system_role,
        )
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 16)
        image_inputs, video_inputs, video_kwargs = self._process_vision_info(
            messages,
            image_patch_size=image_patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        video_metadata = None
        if video_inputs is not None:
            video_inputs, video_metadata = zip(*video_inputs)
            video_inputs = list(video_inputs)
            video_metadata = list(video_metadata)

        processor_kwargs: Dict[str, Any] = {
            "text": [text],
            "images": image_inputs,
            "padding": True,
            "return_tensors": "pt",
            "do_resize": False,
            **video_kwargs,
        }
        if video_inputs is not None:
            processor_kwargs["videos"] = video_inputs
            processor_kwargs["video_metadata"] = video_metadata
        inputs = self.processor(**processor_kwargs)

        model_device = getattr(self.model, "device", None)
        tensor_inputs: Dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to") and model_device is not None:
                tensor_inputs[key] = value.to(model_device)
            else:
                tensor_inputs[key] = value

        with self._torch.no_grad():
            output_ids = self.model.generate(
                **tensor_inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        input_ids = tensor_inputs["input_ids"]
        trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)
        ]
        text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return parse_prediction_text(text)
