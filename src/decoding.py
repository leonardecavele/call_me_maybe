import logging

from enum import IntEnum, auto
from typing import Any

from llm_sdk import Small_LLM_Model

logger: logging.Logger = logging.getLogger(__name__)

MAX_TOKENS: int = 256
NEGATIVE_INF: float = -1e30


class Step(IntEnum):
    FUNCTION_NAME = 0
    PARAMETERS = auto()


def generate_function_name(
    model: Small_LLM_Model, prompt_ids: list[int], function_names
) -> list[int]:
    function_name_ids: list[int] = []
    next_token: str = ""
    while "\"" not in next_token:
        logits: list[float] = model.get_logits_from_input_ids(prompt_ids)
        function_name_ids.append(
            max(range(len(logits)), key=logits.__getitem__)
        )
    return function_name_ids


def generate_parameters(
    model: Small_LLM_Model, prompt_ids: list[int]
) -> list[int]:
    logits: list[float] = model.get_logits_from_input_ids(prompt_ids)
    return [max(range(len(logits)), key=logits.__getitem__)]


def get_answers(
    model: Small_LLM_Model, augmented_prompts: list[str], prompts: list[str],
    functions: list[dict[str, Any]]
) -> str:
    answer_ids: list[int] = []
    answer_ids.append(model.encode("[")[0].tolist()[0])

    TOOL_CALL: int = model.encode("<tool_call>")[0].tolist()[0]

    prompts_ids: list[list[int]] = [
        model.encode(p)[0].tolist() for p in augmented_prompts
    ]

    function_names: list[str] = [f['name'] for f in functions]

    for prompt_i, _ in enumerate(prompts_ids):

        pattern: list[int] = model.encode(
            f"{{\"prompt\":\"{prompts[prompt_i]}\","
            f"\"name\":\"<tool_call>\","
            f"\"parameters\":\"{{<tool_call>}}\"}}"
        )[0].tolist()

        step: int = 0
        for token_id in pattern:
            if token_id == TOOL_CALL:
                new_ids: list[int] = []
                if step == Step.FUNCTION_NAME:
                    logger.info("generating function name")
                    new_ids += generate_function_name(
                        model, prompts_ids[prompt_i], function_names
                    )
                elif step == Step.PARAMETERS:
                    logger.info("generating parameters")
                    new_ids += generate_parameters(
                        model, prompts_ids[prompt_i]
                    )
                answer_ids += new_ids
            else:
                logger.debug(
                    "token_id=%d piece=%r",
                    token_id, model.decode([token_id])
                )
                answer_ids.append(token_id)
                prompts_ids[prompt_i].append(token_id)
        answer_ids.append(model.encode(",")[0].tolist()[0])

    answer_ids.append(model.encode("]")[0].tolist()[0])
    return model.decode(answer_ids)
