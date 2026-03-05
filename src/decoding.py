import logging
import json
import re

from enum import IntEnum, auto
from typing import Any

from llm_sdk import Small_LLM_Model

from .errors import DecodeError
from .typing import JsonData

logger: logging.Logger = logging.getLogger(__name__)


class Step(IntEnum):
    FUNCTION_NAME = 0
    PARAMETERS = auto()


def generate_fn_name(
    model: Small_LLM_Model, prompt_ids: list[int], fn_first_ids: list[int]
) -> list[int]:
    fn_name_ids: list[int] = []

    logits: list[float] = model.get_logits_from_input_ids(prompt_ids)
    for token_id in range(len(logits)):
        if token_id not in fn_first_ids:
            logits[token_id] = float("-inf")

    next_id: int = max(range(len(logits)), key=logits.__getitem__)
    next_token: str = model.decode([next_id])

    while "\"" not in next_token:
        fn_name_ids.append(next_id)
        prompt_ids.append(next_id)
        logger.debug("next_id=%d piece=%r", next_id, next_token)

        logits = model.get_logits_from_input_ids(prompt_ids)
        next_id = max(range(len(logits)), key=logits.__getitem__)
        next_token = model.decode([next_id])

    return fn_name_ids


def generate_parameters(
    model: Small_LLM_Model, prompt_ids: list[int],
    fns: JsonData, TOOL_CALL: int
) -> list[int]:
    parameters_ids: list[int] = []

    try:
        fn_name: str = re.findall(
            r'"name":"([^"]+)"', model.decode(prompt_ids)
        )[-1]
        if not fn_name:
            raise DecodeError("empty function name")
    except IndexError:
        raise DecodeError("could not extract function name")

    fn: dict[str, Any] = next(
        (f for f in fns if f.get("name") == fn_name), {}
    )
    if not fn:
        raise DecodeError(f"unknown function: {fn_name}")

    parameters: dict[str, dict[str, str]] = fn.get("parameters", {})
    if not parameters:
        return []

    pattern: list[int] = model.encode(",".join(
        (
            f"\"{k}\":<tool_call>"
            if parameters[k].get("type") == "number"
            else f"\"{k}\":\"<tool_call>\""
        )
        for k in parameters.keys()
    ))[0].tolist()

    for token_id in pattern:
        if token_id == TOOL_CALL:
            first: bool = True
            next_str: str = ""
            while True:
                logits: list[float] = model.get_logits_from_input_ids(
                    prompt_ids
                )
                if first:
                    first = False
                    for token_id in range(len(logits)):
                        token_str: str = model.decode([token_id])
                        if "}" in token_str or "\"" in token_str:
                            logits[token_id] = float("-inf")
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
    fns: JsonData
) -> str:
    answer_ids: list[int] = []
    answer_ids.append(model.encode("[")[0].tolist()[0])

    TOOL_CALL: int = model.encode("<tool_call>")[0].tolist()[0]

    prompts_ids: list[list[int]] = [
        model.encode(p)[0].tolist() for p in augmented_prompts
    ]

    fn_names: list[str] = [fn.get("name", "") for fn in fns]
    fn_first_ids: list[int] = [
        model.encode(name)[0].tolist()[0] for name in fn_names if name
    ]

    for prompt_i, _ in enumerate(prompts_ids):

        try:
            prompt_json: str = json.dumps(
                prompts[prompt_i], ensure_ascii=False
            )
        except (TypeError, ValueError, RecursionError):
            raise DecodeError("cannot dump prompt")

        pattern: list[int] = model.encode(
            f"{{\"prompt\":{prompt_json},"
            f"\"name\":\"<tool_call>\","
            f"\"parameters\":{{<tool_call>}}}}"
        )[0].tolist()

        step: int = 0
        for token_id in pattern:
            if token_id == TOOL_CALL:
                new_ids: list[int] = []
                if step == Step.FUNCTION_NAME:
                    logger.info("generating function name")
                    new_ids += generate_fn_name(
                        model, prompts_ids[prompt_i], fn_first_ids
                    )
                    step += 1
                elif step == Step.PARAMETERS:
                    logger.info("generating parameters")
                    new_ids += generate_parameters(
                        model, prompts_ids[prompt_i], fns, TOOL_CALL
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
    return str(model.decode(answer_ids))
