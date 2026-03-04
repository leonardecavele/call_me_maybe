import json
import logging

from pathlib import Path
from typing import Any

from .errors import PromptError

logger: logging.Logger = logging.getLogger(__name__)


def parse_prompts(input_path: Path) -> list[str]:
    try:
        with input_path.open("r", encoding="utf-8") as f:
            try:
                data: list[dict[str, str]] = json.load(f)
            except json.JSONDecodeError as e:
                raise PromptError(e)
    except OSError as e:
        raise PromptError(e)

    return [" ".join(d.values()) for d in data]


def format_function(fn_desc: dict[str, Any]) -> str:
    return (
        f"NAME: \"{fn_desc['name']}\"."
    )


def get_prompt_context(functions_definition_path: Path) -> str:
    try:
        with functions_definition_path.open("r", encoding="utf-8") as f:
            try:
                data: list[dict[str, Any]] = json.load(f)
            except json.JSONDecodeError as e:
                raise PromptError(e)
        return "\n".join(format_function(fn) for fn in data)
    except OSError as e:
        raise PromptError(e)


def augment_prompts(
    prompts: list[str],
    context: str
) -> list[str]:
    augmented_prompts: list[str] = []
    for p in prompts:
        augmented_prompt: str = "\n".join([
            context,
            f"User: {p}",
        ])
        augmented_prompts.append(augmented_prompt)
    return augmented_prompts
