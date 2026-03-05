import json
import argparse

from pydantic import (
    BaseModel, field_validator, StrictStr, ConfigDict, Field, ValidationError
)

from pathlib import Path
from typing import Any

from .errors import ParseError
from .typing import JsonData


ARGUMENTS: list[tuple[tuple[str, ...], dict[str, Any]]] = [
    (
        ("--functions_definition",),
        {"default": "data/input/functions_definition.json",
         "help": "path to the function definitions JSON file"}
    ),
    (
        ("--input",),
        {"default": "data/input/function_calling_tests.json",
         "help": "path to the input JSON file"}
    ),
    (
        ("--output",),
        {"default": "data/output/function_calling_results.json",
         "help": "path to the output JSON file"}
    ),
    (
        ("--log-level",),
        {"default": "ERROR", "choices": ("INFO", "DEBUG", "ERROR"),
         "help": "logging level"}
    )
]


def parse_args() -> dict[str, Any]:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    for flags, kwargs in ARGUMENTS:
        parser.add_argument(*flags, **kwargs)

    args: argparse.Namespace = parser.parse_args()

    fn_def_path: Path = Path(args.functions_definition)
    input_path: Path = Path(args.input)
    output_path: Path = Path(args.output)

    if not fn_def_path.is_file():
        raise FileNotFoundError(f"not found functions file: {fn_def_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"not found input file: {input_path}")

    return {
        'log_level': args.log_level,
        'fn_def': fn_def_path,
        'input': input_path,
        'output': output_path
    }


class JsonParsingHandler:
    class ValidatePrompts(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        prompt: StrictStr = Field(..., min_length=1)

        @field_validator("prompt")
        @classmethod
        def not_blank(cls, value: str) -> str:
            value = value.strip()
            if not value:
                raise ValueError("prompt must be non-empty")
            return value

    def parse_prompts(self, input_path: Path) -> list[str]:
        try:
            with input_path.open("r", encoding="utf-8") as f:
                data: Any = json.load(f)

            if not isinstance(data, list):
                raise ParseError("prompt json must be a list")

            for d in data:
                if not isinstance(d, dict):
                    raise ParseError("prompt json items must be dicts")
                try:
                    self.ValidatePrompts(**d)
                except ValidationError as e:
                    raise ParseError(e)

        except (OSError, json.JSONDecodeError, AttributeError, TypeError) as e:
            raise ParseError(e)

        return [" ".join(d.values()) for d in data]

    class ValidateFn(BaseModel):
        pass

    def parse_fn_def(self, fns_path: Path) -> JsonData:
        try:
            with fns_path.open("r", encoding="utf-8") as f:
                try:
                    data: Any = json.load(f)
                except json.JSONDecodeError as e:
                    raise ParseError(e) from e

            try:
                if not all(isinstance(x, dict) for x in data):
                    raise ParseError("functions_definition.json is invalid")
            except TypeError:
                raise ParseError("functions_definition.json must be a list")

            self.ValidateFn(data)
            return data

        except OSError as e:
            raise ParseError(e) from e
