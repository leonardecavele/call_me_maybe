import sys
import json
import logging

from pathlib import Path
from typing import Any, TextIO

import colorlog

from llm_sdk import Small_LLM_Model

from .decoding import get_answers
from .prompt import augment_prompts
from .parsing import parse_args, JsonParsingHandler
from .errors import ErrorCode, DecodeError, PromptError, ParseError
from .typing import JsonData

logger: logging.Logger = logging.getLogger(__name__)


def set_up_handler() -> "logging.StreamHandler[TextIO]":
    handler: logging.StreamHandler[TextIO] = logging.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            (
                "%(asctime)s %(log_color)s%(levelname)s%(reset)s "
                "%(name)s:%(lineno)d - %(message)s"
            ),
            log_colors={
                "DEBUG": "blue",
                "INFO": "yellow",
                "ERROR": "red",
            },
        )
    )
    return handler


def main() -> int:
    # parse and handle arguments
    try:
        args: dict[str, Any] = parse_args()
    except OSError:
        return ErrorCode.ARGS_ERROR
    except SystemExit as e:
        if e.code == 0 or e.code is None:
            return ErrorCode.NO_ERROR
        return ErrorCode.ARGS_ERROR

    logging.getLogger().setLevel(level=getattr(logging, args["lib_log_level"]))
    logging.basicConfig(
        level=getattr(logging, args["lib_log_level"]),
        handlers=[set_up_handler()],
        force=True
    )

    logger.setLevel(level=getattr(logging, args["log_level"]))

    paths: dict[str, Path] = {
        k: a for k, a in args.items() if isinstance(a, Path)
    }
    logger.info("arguments parsed")
    logger.debug(args)

    # parsing and prompt augentation
    try:
        fn_data: JsonData = JsonParsingHandler.parse_fn_def(paths['fn_def'])
    except ParseError as e:
        logger.error(e)
        return ErrorCode.PARSE_ERROR

    try:
        prompts: list[str] = JsonParsingHandler.parse_prompts(paths['input'])
    except PromptError as e:
        logger.error(e)
        return ErrorCode.PROMPT_ERROR

    augmented_prompts: list[str] = augment_prompts(prompts, fn_data)
    logger.info("prompts parsed and augmented")
    logger.debug("\n".join(augmented_prompts))

    # constrained decoding
    model: Small_LLM_Model = Small_LLM_Model()
    try:
        answers: str = get_answers(model, augmented_prompts, prompts, fn_data)
    except DecodeError as e:
        logger.error(e)
        return ErrorCode.DECODE_ERROR
    logger.info("got answers from llm")
    logger.debug(answers)

    # export answers
    output_directory: Path = paths['output'].parent
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(e)
        return ErrorCode.EXPORT_ERROR

    try:
        formatted: Any = json.loads(answers)
    except json.JSONDecodeError as e:
        logger.error(e)
        return ErrorCode.DECODE_ERROR

    try:
        with open(paths['output'], "w", encoding="utf-8") as f:
            json.dump(formatted, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.error(e)
        return ErrorCode.EXPORT_ERROR
    logger.info(f"json exported to {paths['output']}")

    return ErrorCode.NO_ERROR


sys.exit(main())
