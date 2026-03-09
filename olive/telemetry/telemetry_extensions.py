# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import functools
import inspect
import time
from types import TracebackType
from typing import Any, Callable, Optional, TypeVar

from olive.telemetry.telemetry import ACTION_EVENT_NAME, ERROR_EVENT_NAME, _get_logger
from olive.telemetry.utils import _format_exception_message

_TFunc = TypeVar("_TFunc", bound=Callable[..., Any])


def log_action(
    invoked_from: str,
    action_name: str,
    duration_ms: float,
    success: bool,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    telemetry = _get_logger()
    attributes = {
        "invoked_from": invoked_from,
        "action_name": action_name,
        "duration_ms": duration_ms,
        "success": success,
    }
    telemetry.log(ACTION_EVENT_NAME, attributes, metadata)


def log_error(
    exception_type: str,
    exception_message: str,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    telemetry = _get_logger()
    attributes = {
        "exception_type": exception_type,
        "exception_message": exception_message,
    }
    telemetry.log(ERROR_EVENT_NAME, attributes, metadata)


def _resolve_invoked_from(skip_frames: int = 0) -> str:
    """Resolve how Olive was invoked by examining the call stack.

    Walks up the stack to find the first frame outside the olive package,
    which indicates how the user invoked Olive (CLI, script, interactive, etc.).

    :param skip_frames: Number of additional frames to skip (for internal use).
    :return: A string indicating how Olive was invoked.
    """
    for frame_info in inspect.stack()[2 + skip_frames :]:  # skip this function and caller
        module = inspect.getmodule(frame_info.frame)
        if module is None:
            # Could be interactive or dynamically generated code
            continue
        module_name = module.__name__
        # Skip olive internals to find user code
        if module_name.startswith("olive."):
            continue
        if module_name == "__main__":
            return "Script"
        return module_name
    return "Interactive"


class ActionContext:
    """Context manager for recording telemetry around a block of work."""

    def __init__(
        self,
        action_name: str,
        invoked_from: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.action_name = action_name
        self.invoked_from = invoked_from if invoked_from is not None else _resolve_invoked_from()
        self.metadata = metadata or {}
        self._start_time: Optional[float] = None

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def __enter__(self) -> "ActionContext":
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        duration_ms = int((time.perf_counter() - (self._start_time or time.perf_counter())) * 1000)
        success = exc_type is None

        log_action(
            invoked_from=self.invoked_from,
            action_name=self.action_name,
            duration_ms=duration_ms,
            success=success,
            metadata=self.metadata,
        )

        if exc_type is not None and exc_val is not None:
            log_error(
                exception_type=exc_type.__name__,
                exception_message=_format_exception_message(exc_val, exc_tb),
                metadata=self.metadata,
            )

        # Do not suppress exceptions
        return False


def action(func: _TFunc) -> _TFunc:
    """Record telemetry around a function call."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        invoked_from = _resolve_invoked_from()
        action_name = func.__name__
        if args and hasattr(args[0], "__class__"):
            cls_name = args[0].__class__.__name__
            cls_name = cls_name[: -len("Command")] if cls_name.endswith("Command") else cls_name
            if cls_name:
                action_name = cls_name if action_name == "run" else f"{cls_name}.{action_name}"

        start_time = time.perf_counter()
        success = True
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            success = False
            log_error(
                exception_type=type(exc).__name__,
                exception_message=_format_exception_message(exc, exc.__traceback__),
            )
            raise
        finally:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_action(
                invoked_from=invoked_from,
                action_name=action_name,
                duration_ms=duration_ms,
                success=success,
            )

    return wrapper  # type: ignore[return-value]
