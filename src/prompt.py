import json
import logging

from pathlib import Path
from typing import Any

from .errors import PromptError

logger: logging.Logger = logging.getLogger(__name__)

DIRECTIVE: str = (
    """
    You are a JSON generator.

    Task: For each user prompt, output exactly one JSON object with these keys
    only, in this exact order:
    1) "prompt" (string): the original prompt unchanged
    2) "name" (string): the function name to call
    3) "parameters" (object): required arguments only, with correct types

    Output format (STRICT):
    - Return ONLY a JSON array of objects.
    - The output MUST be valid JSON.
    - The JSON MUST be minified on a single line (no newlines, no indentation).
    - Do NOT write any extra text, explanations, or prose.
    - Do NOT wrap the JSON in Markdown code fences.
    - Do NOT add extra keys.
    - Every required parameter must be present.
    - Parameter types must match the function definitions exactly.
    """
)


def parse_prompts(input_path: Path) -> list[dict[str, str]]:
    try:
        with input_path.open("r", encoding="utf-8") as f:
            try:
                data: list[dict[str, str]] = json.load(f)
            except json.JSONDecodeError as e:
                raise PromptError(e)
    except OSError as e:
        raise PromptError(e)
    return data


def format_function(fn_desc: dict[str, Any]) -> str:
    parameters: str = ", ".join(
        f"\'{param_name}\' ({param_info.get('type', 'unknown')})"
        for param_name, param_info in fn_desc.get('parameters', {}).items()
    )
    return_value: str = fn_desc.get("returns", {}).get("type", "unknown")
    return (
        f"NAME: \"{fn_desc['name']}\", "
        f"DESCRIPTION: \"{fn_desc['description']}\", "
        f"PARAMETERS: \"{parameters}\", "
        f"RETURNS: \"{return_value}\"."
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
    pre_prompts: list[dict[str, str]],
    context: str
) -> list[str]:
    prompts: list[str] = []
    for p in pre_prompts:
        prompt: str = "\n".join(str(v) for v in p.values())
        augmented_prompt: str = (
            context + DIRECTIVE + "\nUser: " + prompt + "\nAssistant: "
        )
        prompts.append(augmented_prompt)
    return prompts
