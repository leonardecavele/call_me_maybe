from enum import IntEnum, auto


class ErrorCode(IntEnum):
    NO_ERROR = 0
    ARGS_ERROR = auto()
    PROMPT_ERROR = auto()
    DECODE_ERROR = auto()
    PARSE_ERROR = auto()
    EXPORT_ERROR = auto()


class PromptError(Exception):
    pass


class DecodeError(Exception):
    pass


class ParseError(Exception):
    pass
