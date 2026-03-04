import logging
import json

from typing import Any

from llm_sdk import Small_LLM_Model

from .errors import DecodeError

logger: logging.Logger = logging.getLogger(__name__)

MAX_TOKENS: int = 256
NEGATIVE_INF: float = -1e30


def get_dict_ids(model: Small_LLM_Model) -> dict[str, int]:
    #path: str = model.get_path_to_vocab_file()
    #try:
    #    with open(path, "r", encoding="utf-8") as f:
    #        try:
    #            vocab: dict[str, Any] = json.load(f)
    #        except json.JSONDecodeError as e:
    #            raise DecodeError(e)
    #except OSError as e:
    #    raise DecodeError(e)

    #print("len(vocab) =", len(vocab))
    #print("max_id =", max(vocab.values()))
    #print("min_id =", min(vocab.values()))
    return {
        "QUOTE": model.encode("\"")[0].tolist()[0],
        "OPEN_SQB": model.encode("[")[0].tolist()[0],
        "CLOSE_SQB": model.encode("]")[0].tolist()[0],
        "OPEN_CRLB": model.encode("{")[0].tolist()[0],
        "NAME": model.encode("name")[0].tolist()[0],
        "PROMPT": model.encode("prompt")[0].tolist()[0],
        "PARAMETERS": model.encode("parameters")[0].tolist()[0],
        "COLON": model.encode(":")[0].tolist()[0],
        "COMMA": model.encode(",")[0].tolist()[0],
        "EOS": 55940
    }


def constraint(
    i: int, ids: dict[str, int]
) -> tuple[list[int], list[int], bool]:
    id_pool: list[int] = []
    id_ban: list[int] = []
    in_string: bool = False
    match i:
        case 0:
            id_pool.append(ids["OPEN_SQB"])
        case 1 | 20:
            id_pool.append(ids["OPEN_CRLB"])
        case 2 | 4 | 6 | 9 | 11 | 13 | 16 | 18:
            id_pool.append(ids["QUOTE"])
        case 3:
            id_pool.append(ids["PROMPT"])
        case 5 | 12 | 19:
            id_pool.append(ids["COLON"])
        case 7 | 14:
            in_string = True
        case 8 | 15:
            id_pool.append(ids["COMMA"])
        case 10:
            id_pool.append(ids["NAME"])
        case 17:
            id_pool.append(ids["PARAMETERS"])

    return id_pool, id_ban, in_string


def greedy_constrained_id(
    logits: list[float], id_pool: list[int], id_ban: list[int]
) -> int:
    filtered: list[float] = []
    if id_pool:
        filtered = [NEGATIVE_INF] * len(logits)
        for token_id in id_pool:
            if 0 <= token_id < len(logits) and token_id not in id_ban:
                filtered[token_id] = logits[token_id]
    else:
        filtered = logits[:]
        for token_id in id_ban:
            if 0 <= token_id < len(filtered):
                filtered[token_id] = NEGATIVE_INF

    return max(range(len(filtered)), key=filtered.__getitem__)


def get_answers(
    model: Small_LLM_Model, augmented_prompts: list[str], prompts: list[str]
) -> list[str]:
    ids: dict[str, int] = get_dict_ids(model)

    answers: list[str] = []
    prompts_ids: list[list[int]] = [
        model.encode(p)[0].tolist() for p in augmented_prompts
    ]

    for prompt_i, prompt_ids in enumerate(prompts_ids):
        i: int = 0

        pattern: list[int] = model.encode(
            f"[{{\"prompt\":\"{prompts[prompt_i]}\","
            f"\"name\":\"<tool_call>\","
            f"\"parameters\":\"<tool_call>\"}}]"
        )[0].tolist()

        in_string: bool = False
        input_ids: list[int] = prompt_ids[:]
        answer_ids: list[int] = []

        for _ in range(MAX_TOKENS):
            logits: list[float] = model.get_logits_from_input_ids(input_ids)
            pool_id, id_ban, in_string = constraint(i, ids)
            next_id: int = greedy_constrained_id(logits, pool_id, id_ban)
            input_ids.append(next_id)
            answer_ids.append(next_id)

            logger.debug(
                "i=%d next_id=%d piece=%r", i, next_id, model.decode([next_id])
            )

            if "\"" in model.decode([next_id]):
                if "," in model.decode([next_id]):
                    i += 1
                in_string = False
            if not in_string:
                i += 1

            if next_id == ids["EOS"]:
                break
            if "]" in model.decode([next_id]):
                break

        answers.append(model.decode(answer_ids))

    return answers
