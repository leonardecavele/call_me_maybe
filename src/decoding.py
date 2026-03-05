import logging
import re

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
    model: Small_LLM_Model, prompt_ids: list[int]
) -> list[int]:
    function_name_ids: list[int] = []

    next_id: int = model.encode("fn_")[0].tolist()[0]
    next_token: str = ""

    while "\"" not in next_token:
        function_name_ids.append(next_id)
        prompt_ids.append(next_id)
        logger.debug("next_id=%d piece=%r", next_id, next_token)

        logits: list[float] = model.get_logits_from_input_ids(prompt_ids)
        next_id = max(range(len(logits)), key=logits.__getitem__)
        next_token = model.decode([next_id])

    return function_name_ids


def generate_parameters(
    model: Small_LLM_Model, prompt_ids: list[int],
    functions: list[dict[str, Any]], TOOL_CALL: int
) -> list[int]:
    parameters_ids: list[int] = []

    # TODO PROTECT TYPE HINT ALL
    function_name: str = re.findall(
        r'"name":"([^"]+)"', model.decode(prompt_ids)
    )[-1]
    fn: dict[str, Any] = next(
        (f for f in functions if f.get("name") == function_name), {}
    )
    parameters: dict[str, dict[str, str]] = fn.get("parameters", {})
    pattern: list[int] = model.encode(",".join(
        [f"\"{k}\":\"<tool_call>\"" for k in parameters.keys()]
    ))[0].tolist()

    for token_id in pattern:
        if token_id == TOOL_CALL:
            next_str: str = ""
            while True:
                logits: list[float] = model.get_logits_from_input_ids(
                    prompt_ids
                )
                next_id: int = max(range(len(logits)), key=logits.__getitem__)
                next_str = model.decode([next_id])

                if "}" in next_str or "\"" in next_str:
                    break

                logger.debug(
                    "token_id=%d piece=%r", next_id, next_str
                )
                parameters_ids.append(next_id)
                prompt_ids.append(next_id)
        else:
            logger.debug(
                "token_id=%d piece=%r",
                token_id, model.decode([token_id])
            )
            parameters_ids.append(token_id)
            prompt_ids.append(token_id)

    return parameters_ids


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

    for prompt_i, _ in enumerate(prompts_ids):

        pattern: list[int] = model.encode(
            f"{{\"prompt\":\"{prompts[prompt_i]}\","
            f"\"name\":\"<tool_call>\","
            f"\"parameters\":{{<tool_call>}}}}"
        )[0].tolist()

        step: int = 0
        for token_id in pattern:
            if token_id == TOOL_CALL:
                new_ids: list[int] = []
                if step == Step.FUNCTION_NAME:
                    logger.info("generating function name")
                    new_ids += generate_function_name(
                        model, prompts_ids[prompt_i]
                    )
                    step += 1
                elif step == Step.PARAMETERS:
                    logger.info("generating parameters")
                    new_ids += generate_parameters(
                        model, prompts_ids[prompt_i], functions, TOOL_CALL
                    )
                    step += 1
                answer_ids += new_ids
            else:
                logger.debug(
                    "token_id=%d piece=%r",
                    token_id, model.decode([token_id])
                )
                answer_ids.append(token_id)
                prompts_ids[prompt_i].append(token_id)
        if prompt_i < len(prompts_ids) - 1:
            answer_ids.append(model.encode(",")[0].tolist()[0])

    answer_ids.append(model.encode("]")[0].tolist()[0])
    return model.decode(answer_ids)
