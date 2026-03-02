import os
import sys
import json
import logging
import argparse

from pathlib import Path

import numpy
from llm_sdk import Small_LLM_Model


def available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


half = max(1, available_cpus() // 2)

os.environ["OMP_NUM_THREADS"] = str(half)
os.environ["MKL_NUM_THREADS"] = str(half)
os.environ["OPENBLAS_NUM_THREADS"] = str(half)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(half)
os.environ["NUMEXPR_NUM_THREADS"] = str(half)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


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
        'functions_definition_path': functions_definition_path,
        'input_path': input_path,
        'output_path': output_path
    }


def parse_prompts(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8") as f:
        try:
            data: list[dict[str, str]] = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(e)
    return data


def augment_prompts(pre_prompts: list[dict[str, str]]) -> list[str]:
    prompts: list[str] = []
    for p in pre_prompts:
        final_string: str = ""
        final_string.join(p)
        # augment prompts using args, functions definition path and stuff
        prompts.append(final_string)
    return prompts


def main() -> int:
    model = Small_LLM_Model()

    paths: dict[str, Path] = parse_args()

    pre_prompts: list[dict[str, str]] = parse_prompts(paths['input_path'])
    prompts: list[str] = augment_prompts(pre_prompts)

    # for eeach prompt
    # constrained decoding
    prompt_ids: list[int] = model.encode("User: Hello\nAssistant:")[0].tolist()
    logits: list[float] = model.get_logits_from_input_ids(prompt_ids)
    top = numpy.argsort(logits)[-10:][::-1]

    for i in top:
        print(i, repr(model.decode([int(i)])), float(logits[i]))
    return 0


sys.exit(main())
