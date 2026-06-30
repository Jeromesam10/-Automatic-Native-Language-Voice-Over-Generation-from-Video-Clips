from __future__ import annotations

import json
from typing import Any


class PromptValidationError(ValueError):
    """Raised when request prompt data is invalid."""


def build_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        text = prompt.strip()
        if text:
            return text
        raise PromptValidationError("Field 'prompt' is required.")

    if not isinstance(prompt, dict):
        raise PromptValidationError("Field 'prompt' must be a string or object.")

    task = str(prompt.get("task", "")).strip()
    data = prompt.get("data")

    if task and isinstance(data, dict):
        return _build_structured_prompt(task, data)

    return json.dumps(prompt, indent=2)


def _build_structured_prompt(task: str, data: dict[str, Any]) -> str:
    lines = [f"Task: {task}"]

    objects = data.get("objects")
    if isinstance(objects, list) and objects:
        object_names = ", ".join(str(item) for item in objects)
        lines.append(f"Objects: {object_names}")

    frames = data.get("frames")
    if isinstance(frames, list) and frames:
        lines.append("")
        lines.append("Frame descriptions:")
        for index, frame in enumerate(frames, start=1):
            if isinstance(frame, dict):
                description = str(frame.get("description", "")).strip()
            else:
                description = str(frame).strip()
            if description:
                lines.append(f"{index}. {description}")

    lines.append("")
    lines.append("Respond with only the final result for the task.")

    return "\n".join(lines)
