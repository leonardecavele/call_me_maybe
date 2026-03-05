import sys
import json
import logging
import argparse

from pathlib import Path
from typing import Any

import colorlog

from llm_sdk import Small_LLM_Model

from .decoding import get_answers
from .prompt import parse_prompts, augment_prompts
from .errors import (
    DecodeError, ErrorCode, PromptError, ParseError
)

logger: logging.Logger = logging.getLogger(__name__)


def set_up_handler() -> logging.StreamHandler:
    handler: logging.StreamHandler = logging.StreamHandler()
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


def parse_args() -> dict[str, Path]:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="path to the function definitions JSON file"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="path to the input JSON file"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="path to the output JSON file"
    )
    parser.add_argument(
        "--log-level",
        default="ERROR",
        choices=("INFO", "DEBUG", "ERROR"),
        help="logging level"
    )
    parser.add_argument(
        "--lib-log-level",
        default="WARNING",
        choices=("CRITICAL", "ERROR", "INFO", "DEBUG", "WARNING"),
        help="libraries logging level"
    )

    args: argparse.Namespace = parser.parse_args()
    functions_definition_path: Path = Path(args.functions_definition)
    input_path: Path = Path(args.input)
    output_path: Path = Path(args.output)

    if not functions_definition_path.is_file():
        raise FileNotFoundError(
            f"Missing functions definition file: {functions_definition_path}"
        )
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Missing input file: {input_path}"
        )

    output_directory: Path = output_path.parent
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(e)

    return {
        'lib_log_level': args.lib_log_level,
        'log_level': args.log_level,
        'functions': functions_definition_path,
        'input': input_path,
        'output': output_path
    }


def get_functions_definition(functions: Path) -> list[dict[str, Any]]:
    try:
        with functions.open("r", encoding="utf-8") as f:
            try:
                data: Any = json.load(f)
            except json.JSONDecodeError as e:
                raise ParseError(e) from e

        if not isinstance(data, list) or not all(
            isinstance(x, dict) for x in data
        ):
            raise ParseError(
                "functions_definition.json must be a JSON array of objects"
            )

        return data
    except OSError as e:
        raise PromptError(e) from e


def main() -> int:
    # parse and treat arguments
    try:
        args: dict[str, Any] = parse_args()
    except OSError:
        return ErrorCode.ARGS_ERROR
    except SystemExit as e:
        if e.code == 0 or e.code is None:
            return ErrorCode.NO_ERROR
        return ErrorCode.ARGS_ERROR

    logging.basicConfig(
        level=getattr(logging, args["log_level"]),
        handlers=[set_up_handler()],
        force=True
    )
    for lib in (
        "transformers",
        "huggingface_hub",
        "accelerate",
        "torch",
        "urllib3",
        "httpx",
        "httpcore",
    ):
        logging.getLogger(lib).setLevel(
            getattr(logging, args["lib_log_level"])
        )

    paths: dict[str, Path] = {
        k: a for k, a in args.items() if isinstance(a, Path)
    }
    logger.info("arguments parsed")
    logger.debug(args)

    # parse and augment prompts
    try:
        functions: list[dict[str, Any]] = get_functions_definition(
            paths['functions']
        )
        prompts: list[str] = parse_prompts(paths['input'])
    except PromptError as e:
        logger.error(e)
        return ErrorCode.PROMPT_ERROR
    except ParseError as e:
        logger.error(e)
        return ErrorCode.PARSE_ERROR

    augmented_prompts: list[str] = augment_prompts(prompts, functions)
    logger.info("prompts parsed and augmented")
    logger.debug("\n".join(augmented_prompts))

    # constrained decoding
    model: Small_LLM_Model = Small_LLM_Model()
    try:
        answers: str = get_answers(
            model, augmented_prompts, prompts, functions
        )
    except DecodeError as e:
        logger.error(e)
        return ErrorCode.DECODE_ERROR
    logger.info("got answers from llm")
    logger.debug(answers)

    # export
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
