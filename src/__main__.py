import sys
import logging
import argparse

from pathlib import Path
from typing import Any

import colorlog

from llm_sdk import Small_LLM_Model

from .decoding import get_answers
from .prompt import parse_prompts, augment_prompts, get_prompt_context
from .errors import ErrorCode, PromptError

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
        default="data/output/function_calls.json",
        help="path to the output JSON file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("INFO", "DEBUG"),
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
        'functions_definition_path': functions_definition_path,
        'input_path': input_path,
        'output_path': output_path
    }


def main() -> int:
    # parse and treat arguments
    try:
        args: dict[str, Any] = parse_args()
    except OSError as e:
        logger.error(e)
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
        pre_prompts: list[dict[str, str]] = parse_prompts(paths['input_path'])
        context: str = get_prompt_context(paths['functions_definition_path'])
    except PromptError as e:
        logger.error(e)
        return ErrorCode.PROMPT_ERROR

    prompts: list[str] = augment_prompts(pre_prompts, context)
    logger.info("prompts parsed and augmented")
    logger.debug("".join(prompts[0])) # edit with \n etc

    # constrained decoding
    model: Small_LLM_Model = Small_LLM_Model()
    answers: list[str] = get_answers(model, prompts)
    logger.info("got answers from llm")
    logger.debug(
        "\n".join(f"answer{i}: {a}" for i, a in enumerate(answers, start=1))
    )

    # export

    return ErrorCode.NO_ERROR


sys.exit(main())
