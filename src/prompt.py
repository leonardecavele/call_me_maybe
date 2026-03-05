import logging

from typing import Any

from .errors import PromptError
from .typing import JsonData

logger: logging.Logger = logging.getLogger(__name__)


DIRECTIVE: str = "Pick the appropriate function:"


def format_fn(fn_desc: dict[str, Any]) -> str:
    return f"NAME: \"{fn_desc['name']}\"."


def get_prompt_context(fns: JsonData) -> str:
    try:
        if not all(isinstance(fn, dict) for fn in fns):
            raise PromptError("Expected list[dict] for function definitions")
        return "\n".join(format_fn(fn) for fn in fns)
    except Exception as e:
        raise PromptError(e)


def augment_prompts(prompts: list[str], fns: JsonData) -> list[str]:
    context: str = get_prompt_context(fns)
    augmented_prompts: list[str] = []
    for p in prompts:
        augmented_prompt: str = "\n".join([DIRECTIVE, context, f"User: {p}"])
        augmented_prompts.append(augmented_prompt)
    return augmented_prompts
