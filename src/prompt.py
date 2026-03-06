import logging

from typing import Any

from .typing import JsonData

logger: logging.Logger = logging.getLogger(__name__)


DIRECTIVE: str = "Pick the appropriate function:"


def format_fn(fn_desc: dict[str, Any]) -> str:
    """
    Format one function description for the prompt.

    Return the prompt line that exposes the function name.

    Parameters
    ----------
    fn_desc
        A function description mapping.

    Returns
    -------
    str
        A formatted prompt line for the function.
    """
    return f"NAME: \"{fn_desc['name']}\"."


def get_prompt_context(fns: JsonData) -> str:
    """
    Build the function list context for prompting.

    Join all formatted function names into one prompt context string.

    Parameters
    ----------
    fns
        The validated function definitions.

    Returns
    -------
    str
        The prompt context listing the available functions.
    """
    return "\n".join(format_fn(fn) for fn in fns)


def augment_prompts(prompts: list[str], fns: JsonData) -> list[str]:
    """
    Augment prompts with the function context.

    Prepend the directive and function list to each user prompt.

    Parameters
    ----------
    prompts
        The original user prompts.
    fns
        The validated function definitions.

    Returns
    -------
    list[str]
        The augmented prompts ready for decoding.
    """
    context: str = get_prompt_context(fns)
    augmented_prompts: list[str] = []
    for p in prompts:
        augmented_prompt: str = "\n".join([DIRECTIVE, context, f"User: {p}"])
        augmented_prompts.append(augmented_prompt)
    return augmented_prompts
