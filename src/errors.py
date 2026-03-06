from enum import IntEnum, auto


class ErrorCode(IntEnum):
    """
    Exit codes returned by the command-line program.

    Attributes
    ----------
    NO_ERROR : int
        Execution completed successfully.
    ARGS_ERROR : int
        Argument parsing or validation failed.
    DECODE_ERROR : int
        Model decoding or output validation failed.
    PARSE_ERROR : int
        Input JSON parsing or validation failed.
    EXPORT_ERROR : int
        Writing the output file failed.
    """
    NO_ERROR = 0
    ARGS_ERROR = auto()
    DECODE_ERROR = auto()
    PARSE_ERROR = auto()
    EXPORT_ERROR = auto()


class DecodeError(Exception):
    """
    Raised when constrained decoding fails.
    """
    pass


class ParseError(Exception):
    """
    Raised when input parsing or validation fails.
    """
    pass
