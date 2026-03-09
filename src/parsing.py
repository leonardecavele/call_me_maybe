import json
import argparse

from pydantic import (
    BaseModel, field_validator, StrictStr, ConfigDict, Field, ValidationError
)

from pathlib import Path
from typing import Any, Literal, cast

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
    """
    Parse and validate command-line arguments.

    Build CLI arguments and ensure required input files exist.

    Returns
    -------
    dict[str, Any]
        A mapping of parsed options and resolved paths.

    Raises
    ------
    FileNotFoundError:
        Raised when the functions file does not exist.
    FileNotFoundError:
        Raised when the input file does not exist.
    """
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
    """
    Parse and validate input JSON files.
    """
    class ValidatePrompts(BaseModel):
        """
        Validate one prompt entry from the input file.

        Attributes
        ----------
        model_config : ConfigDict
            Pydantic configuration for strict validation.
        prompt
            The user prompt to validate.
        """
        model_config = ConfigDict(extra="forbid", strict=True)
        prompt: StrictStr = Field(..., min_length=1)

        @field_validator("prompt")
        @classmethod
        def not_blank(cls, value: str) -> str:
            """
            Reject blank prompts after trimming whitespace.

            Validate and normalize a prompt string.

            Parameters
            ----------
            cls : type
                The model class running the validator.
            value
                The prompt string to validate.

            Returns
            -------
            str
                The trimmed prompt string.

            Raises
            ------
            ValueError:
                Raised when the prompt is empty after trimming.
            """
            value = value.strip()
            if not value:
                raise ValueError("prompt must be non-empty")
            return value

    def parse_prompts(self, input_path: Path) -> list[str]:
        """
        Parse and validate the prompt input file.

        Load prompt objects from JSON and return their prompt strings.

        Parameters
        ----------
        input_path
            Path to the prompt JSON file.

        Returns
        -------
        list[str]
            The validated prompt strings.

        Raises
        ------
        ParseError:
            Raised when the file cannot be read.
        ParseError:
            Raised when the file is empty.
        ParseError:
            Raised when the JSON is invalid.
        ParseError:
            Raised when an entry has an invalid structure.
        ParseError:
            Raised when the prompt list is empty.
        """
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
        """
        Validate one function definition entry.

        Attributes
        ----------
        model_config : ConfigDict
            Pydantic configuration for strict validation.
        name
            The function name.
        description
            A short description of the function.
        parameters
            The function parameter definitions.
        returns
            The function return specification.
        """
        class ValidateParam(BaseModel):
            """
            Validate one parameter specification.

            Attributes
            ----------
            model_config : ConfigDict
                Pydantic configuration for strict validation.
            type
                The declared parameter type.
            """
            model_config = ConfigDict(extra="forbid", strict=True)
            type: str = Field(min_length=1)

        class ValidateReturns(BaseModel):
            """
            Validate the return value specification.

            Attributes
            ----------
            model_config : ConfigDict
                Pydantic configuration for strict validation.
            type
                The declared return type.
            """
            model_config = ConfigDict(extra="forbid", strict=True)
            type: str = Field(min_length=1)

        model_config = ConfigDict(extra="forbid", strict=True)

        name: StrictStr = Field(..., min_length=1)
        description: StrictStr = Field(..., min_length=1)

        parameters: dict[StrictStr, ValidateParam]
        returns: ValidateReturns

        @field_validator("name", "description")
        @classmethod
        def not_blank_str(cls, value: str) -> str:
            """
            Reject blank strings after trimming whitespace.

            Validate and normalize a required string.

            Parameters
            ----------
            cls : type
                The model class running the validator.
            value
                The string value to validate.

            Returns
            -------
            str
                The trimmed string.

            Raises
            ------
            ValueError:
                Raised when the string is empty after trimming.
            """
            value = value.strip()
            if not value:
                raise ValueError("must be non-empty")
            return value

        @field_validator("parameters", mode="before")
        @classmethod
        def validate_parameters(cls, value: Any) -> Any:
            """
            Validate the raw parameters mapping.

            Ensure names are non-empty and specs are objects.

            Parameters
            ----------
            cls : type
                The model class running the validator.
            value
                The raw parameters value from the JSON object.

            Returns
            -------
            Any
                The validated raw parameters mapping.

            Raises
            ------
            TypeError:
                Raised when parameters is not an object.
            ValueError:
                Raised when a parameter name is blank.
            TypeError:
                Raised when a parameter spec is not an object.
            """
            if not isinstance(value, dict):
                raise TypeError("parameters must be an object")

            for key, spec in value.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("parameter name must non-empty string")
                if not isinstance(spec, dict):
                    raise TypeError("each parameter spec must be an object")
            return value

    def parse_fn_def(self, fns_path: Path) -> JsonData:
        """
        Parse and validate the functions definition file.

        Load function definitions from JSON and validate their schema.

        Parameters
        ----------
        fns_path
            Path to the functions JSON file.

        Returns
        -------
        JsonData
            The validated function definitions.

        Raises
        ------
        ParseError:
            Raised when the file cannot be read.
        ParseError:
            Raised when the file is empty.
        ParseError:
            Raised when the JSON is invalid.
        ParseError:
            Raised when an entry has an invalid structure.
        ParseError:
            Raised when the functions list is empty.
        """
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

            return cast(JsonData, data)

        except (OSError, json.JSONDecodeError, AttributeError, TypeError) as e:
            raise ParseError(e)
