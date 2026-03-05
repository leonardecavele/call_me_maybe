import json
import argparse

from pydantic import (
    BaseModel, field_validator, StrictStr, ConfigDict, Field, ValidationError
)

from pathlib import Path
from typing import Any, Literal

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
                raw: str = f.read()

                if not raw.strip():
                    raise ParseError(f"empty json file: {input_path}")

                data: Any = json.loads(raw)

            for d in data:
                if not isinstance(d, dict):
                    raise ParseError("prompt json objects must be dicts")
                try:
                    self.ValidatePrompts(**d)
                except ValidationError as e:
                    raise ParseError(e.errors()[0]["msg"])

            if not data:
                raise ParseError("prompt json must not be empty")

            return [" ".join(d.values()) for d in data]

        except (OSError, json.JSONDecodeError, AttributeError, TypeError) as e:
            raise ParseError(e)

    class ValidateFn(BaseModel):
        class ValidateParam(BaseModel):
            model_config = ConfigDict(extra="forbid", strict=True)
            type: Literal["number", "string"]

        class ValidateReturns(BaseModel):
            model_config = ConfigDict(extra="forbid", strict=True)
            type: Literal["number", "string"]

        model_config = ConfigDict(extra="forbid", strict=True)

        name: StrictStr = Field(..., min_length=1)
        description: StrictStr = Field(..., min_length=1)

        parameters: dict[StrictStr, ValidateParam]
        returns: ValidateReturns

        @field_validator("name", "description")
        @classmethod
        def not_blank_str(cls, value: str) -> str:
            value = value.strip()
            if not value:
                raise ValueError("must be non-empty")
            return value

        @field_validator("parameters", mode="before")
        @classmethod
        def validate_parameters(cls, value: Any) -> Any:
            if not isinstance(value, dict):
                raise TypeError("parameters must be an object")

            for key, spec in value.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("parameter name must non-empty string")
                if not isinstance(spec, dict):
                    raise TypeError("each parameter spec must be an object")
            return value

    def parse_fn_def(self, fns_path: Path) -> JsonData:
        try:
            with fns_path.open("r", encoding="utf-8") as f:
                raw: str = f.read()

                if not raw.strip():
                    raise ParseError(f"empty json file: {fns_path}")

                data: Any = json.loads(raw)

            for d in data:
                if not isinstance(d, dict):
                    raise ParseError("function json objects must be dicts")
                try:
                    self.ValidateFn(**d)
                except ValidationError as e:
                    raise ParseError(e.errors()[0]["msg"])

            if not data:
                raise ParseError("functions json must not be empty")

            return data

        except (OSError, json.JSONDecodeError, AttributeError, TypeError) as e:
            raise ParseError(e)
