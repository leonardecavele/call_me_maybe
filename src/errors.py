from enum import IntEnum, auto


class ErrorCode(IntEnum):
    NO_ERROR = 0
    ARGS_ERROR = auto()
    PROMPT_ERROR = auto()


class PromptError(Exception):
    pass
