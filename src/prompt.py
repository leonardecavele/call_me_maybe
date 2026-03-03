import json
import logging

from pathlib import Path

from .errors import PromptError

logger: logging.Logger = logging.getLogger(__name__)


def parse_prompts(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8") as f:
        try:
            data: list[dict[str, str]] = json.load(f)
        except json.JSONDecodeError as e:
            raise PromptError(e)
    return data


def augment_prompts(pre_prompts: list[dict[str, str]]) -> list[str]:
    prompts: list[str] = []
    for p in pre_prompts:
        final_string: str = "\n".join(str(v) for v in p.values())
        # augment prompts using args, functions definition path and stuff
        prompts.append(final_string)
    return prompts
