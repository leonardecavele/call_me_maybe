import logging
import json

from enum import IntEnum, auto
from typing import Any

from llm_sdk import Small_LLM_Model

from .errors import DecodeError
from .typing import JsonData

logger: logging.Logger = logging.getLogger(__name__)


class Step(IntEnum):
    """
    Generation steps for constrained decoding.

    Attributes
    ----------
    FUNCTION_NAME : int
        Step that generates the function name.
    PARAMETERS : int
        Step that generates the function parameters.
    """
    FUNCTION_NAME = 0
    PARAMETERS = auto()


def generate_fn_name(
    model: Small_LLM_Model, ids: list[int], fn_first_ids: list[int]
) -> list[int]:
    """
    Generate a function name token by token.

    Restrict the first token to known functions, then decode greedily.

    Parameters
    ----------
    model
        The language model used for decoding.
    ids
        The current token ids for the prompt and partial output.
    fn_first_ids
        Token ids that can start a valid function name.

    Returns
    -------
    list[int]
        The token ids of the generated function name.
    """
    fn_name_ids: list[int] = []

    logits: list[float] = model.get_logits_from_input_ids(ids)
    for token_id in range(len(logits)):
        if token_id not in fn_first_ids:
            logits[token_id] = float("-inf")

    next_id: int = max(range(len(logits)), key=logits.__getitem__)
    next_token: str = model.decode([next_id])

    while "\"" not in next_token:
        fn_name_ids.append(next_id)
        ids.append(next_id)
        logger.debug("next_id=%d piece=%r", next_id, next_token)

        logits = model.get_logits_from_input_ids(ids)
        next_id = max(range(len(logits)), key=logits.__getitem__)
        next_token = model.decode([next_id])

    return fn_name_ids


def generate_parameters(
    model: Small_LLM_Model, ids: list[int], fn_name_ids: list[int],
    fns: JsonData, TOOL_CALL: int
) -> list[int]:
    """
    Generate parameter values for the selected function.

    Follow the parameter pattern and greedily decode each value.

    Parameters
    ----------
    model
        The language model used for decoding.
    ids
        The current token ids for the prompt and partial output.
    fn_name_ids
        The token ids of the selected function name.
    fns
        The validated function definitions.
    TOOL_CALL
        The placeholder token id used during pattern generation.

    Returns
    -------
    list[int]
        The token ids of the generated parameters.

    Raises
    ------
    DecodeError:
        Raised when the generated function name is unknown.
    """
    parameters_ids: list[int] = []

    fn_name_str: str = model.decode(fn_name_ids)
    fn: dict[str, Any] = next(
        (f for f in fns if f.get("name") == fn_name_str), {}
    )
    if not fn:
        raise DecodeError(f"unknown function: {fn_name_str}")

    parameters: dict[str, dict[str, str]] = fn["parameters"]

    pattern: list[int] = model.encode(",".join(
        (
            f"\"{k}\":<tool_call>"
            if parameters[k].get("type") != "string"
            else f"\"{k}\":\"<tool_call>\""
        )
        for k in parameters.keys()
    ))[0].tolist()

    for token_id in pattern:
        if token_id == TOOL_CALL:
            first: bool = True
            next_str: str = ""
            while True:
                logits: list[float] = model.get_logits_from_input_ids(ids)
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

                logger.debug("token_id=%d piece=%r", next_id, next_str)
                parameters_ids.append(next_id)
                ids.append(next_id)
        else:
            logger.debug(
                "token_id=%d piece=%r",
                token_id, model.decode([token_id])
            )
            parameters_ids.append(token_id)
            ids.append(token_id)

    return parameters_ids


def get_answers(
    model: Small_LLM_Model, augmented_prompts: list[str], prompts: list[str],
    fns: JsonData
) -> str:
    """
    Generate JSON answers for all prompts.

    Build one function call per prompt with constrained greedy decoding.

    Parameters
    ----------
    model
        The language model used for decoding.
    augmented_prompts
        Prompts augmented with the function list context.
    prompts
        Original user prompts to include in the output JSON.
    fns
        The validated function definitions.

    Returns
    -------
    str
        A JSON string containing all generated answers.
    """
    answer_ids: list[int] = []
    answer_ids.append(model.encode("[")[0].tolist()[0])

    TOOL_CALL: int = model.encode("<tool_call>")[0].tolist()[0]

    prompts_ids: list[list[int]] = [
        model.encode(p)[0].tolist() for p in augmented_prompts
    ]

    fn_names: list[str] = [fn["name"] for fn in fns]
    fn_first_ids: list[int] = [
        model.encode(name)[0].tolist()[0] for name in fn_names if name
    ]

    for i, _ in enumerate(prompts_ids):

        prompt_json: str = json.dumps(prompts[i], ensure_ascii=False)

        pattern: list[int] = model.encode(
            f"{{\"prompt\":{prompt_json},"
            f"\"name\":\"<tool_call>\","
            f"\"parameters\":{{<tool_call>}}}}"
        )[0].tolist()

        fn_name_ids: list[int] = []

        step: int = 0
        for token_id in pattern:
            if token_id == TOOL_CALL:
                new_ids: list[int] = []
                if step == Step.FUNCTION_NAME:
                    logger.info("generating function name")
                    fn_name_ids = generate_fn_name(
                        model, prompts_ids[i], fn_first_ids
                    )
                    new_ids += fn_name_ids
                    step += 1
                elif step == Step.PARAMETERS:
                    logger.info("generating parameters")
                    new_ids += generate_parameters(
                        model, prompts_ids[i], fn_name_ids, fns, TOOL_CALL
                    )
                    step += 1
                answer_ids += new_ids
            else:
                logger.debug(
                    "token_id=%d piece=%r",
                    token_id, model.decode([token_id])
                )
                answer_ids.append(token_id)
                prompts_ids[i].append(token_id)
        if i < len(prompts_ids) - 1:
            answer_ids.append(model.encode(",")[0].tolist()[0])

    answer_ids.append(model.encode("]")[0].tolist()[0])
    return str(model.decode(answer_ids))
