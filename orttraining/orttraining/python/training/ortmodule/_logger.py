# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import os
import sys
import tempfile
import textwrap
import time
from contextlib import contextmanager
from enum import IntEnum
from functools import partial
from typing import Callable, Dict, List, Optional

from onnxruntime.capi._pybind_state import Severity

from ._utils import get_rank, get_world_size


class LogLevel(IntEnum):
    VERBOSE = 0
    DEVINFO = 1  # For ORT developers.
    INFO = 1  # For ORT users.
    WARNING = 2
    ERROR = 3
    FATAL = 4


ORTMODULE_LOG_LEVEL_MAP: Dict[LogLevel, List[int]] = {
    LogLevel.VERBOSE: [Severity.VERBOSE, logging.DEBUG],
    LogLevel.DEVINFO: [Severity.INFO, logging.INFO],
    # ONNX Runtime has too many INFO logs, so we map it to WARNING for better user experience.
    LogLevel.INFO: [Severity.WARNING, logging.INFO],
    LogLevel.WARNING: [Severity.WARNING, logging.WARNING],
    LogLevel.ERROR: [Severity.ERROR, logging.ERROR],
    LogLevel.FATAL: [Severity.FATAL, logging.FATAL],
}


def ortmodule_loglevel_to_onnxruntime_c_loglevel(loglevel: LogLevel) -> int:
    return ORTMODULE_LOG_LEVEL_MAP.get(loglevel, [Severity.WARNING, logging.WARNING])[0]


def ortmodule_loglevel_to_python_loglevel(loglevel: LogLevel) -> int:
    return ORTMODULE_LOG_LEVEL_MAP.get(loglevel, [Severity.WARNING, logging.WARNING])[1]


def configure_ortmodule_logger(log_level: LogLevel) -> logging.Logger:
    """Configure the logger for ortmodule according to following rules.
    1. If multiple processes are used, the rank will be appended
       to the logger name.
    2. If the log level is equal to or greater than INFO, the logger will be
       disabled for non-zero ranks.
    """
    rank_info = f".rank-{get_rank()}" if get_world_size() > 1 else ""
    logger = logging.getLogger(f"orttraining{rank_info}")
    # Disable the logger for non-zero ranks when level >= info
    logger.disabled = log_level >= LogLevel.INFO and get_rank() != 0
    logger.setLevel(ortmodule_loglevel_to_python_loglevel(log_level))
    return logger


class LogColor:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ORTModuleInitPhase(IntEnum):
    EndToEnd = 0  # The end to end of ORT first-time initialization
    EXPORT = 1  # The phase of preparing and exporting the model to ONNX
    GRAPH_BUILDER_INIT = 2  # The phase of initializing the graph builder
    DETECTION = 3  # The phase of runtime detection
    BUILD_GRAPH = 4  # The phase of optimizing forward graph (and building the gradient graph for training).
    CREATE_SESSION = 5  # The phase of creating the session

    def to_string(self) -> str:
        if self == ORTModuleInitPhase.EndToEnd:
            return "end to end"
        if self == ORTModuleInitPhase.EXPORT:
            return "export"
        elif self == ORTModuleInitPhase.GRAPH_BUILDER_INIT:
            return "graph builder init"
        elif self == ORTModuleInitPhase.DETECTION:
            return "runtime detection"
        elif self == ORTModuleInitPhase.BUILD_GRAPH:
            return "graph building"
        elif self == ORTModuleInitPhase.CREATE_SESSION:
            return "session creation"
        else:
            return "invalid"


class TimeTracker:
    """A simple class to track time spent in different phases of ORT backend first-time initialization."""

    NOT_RECORD = -1.0

    def __init__(
        self,
    ):
        self.starts_: List[float] = [TimeTracker.NOT_RECORD] * len(ORTModuleInitPhase)
        self.ends_: List[float] = [TimeTracker.NOT_RECORD] * len(ORTModuleInitPhase)

    def start(self, phase: ORTModuleInitPhase):
        self.starts_[phase] = time.time()

    def end(self, phase: ORTModuleInitPhase):
        self.ends_[phase] = time.time()

    def _get_duration(self, phase: ORTModuleInitPhase):
        if self.ends_[phase] == TimeTracker.NOT_RECORD or self.starts_[phase] == TimeTracker.NOT_RECORD:
            return TimeTracker.NOT_RECORD
        return self.ends_[phase] - self.starts_[phase]

    def to_string(self, log_details=False) -> str:
        end_to_end_str = self._get_duration(ORTModuleInitPhase.EndToEnd)
        end_to_end_str = f"{end_to_end_str:.2f}" if end_to_end_str != TimeTracker.NOT_RECORD else "N/A"
        export_str = self._get_duration(ORTModuleInitPhase.EXPORT)
        export_str = f"{export_str:.2f}" if export_str != TimeTracker.NOT_RECORD else "N/A"
        overhead_title_str = (
            f"Total ORT initialization overhead is {end_to_end_str}s where export takes {export_str}s.\n"
        )

        if log_details is False:
            return overhead_title_str

        duration_summaries = []
        for phase in ORTModuleInitPhase:
            _get_duration = self._get_duration(phase)
            if phase in [ORTModuleInitPhase.EndToEnd, ORTModuleInitPhase.EXPORT]:
                continue

            val = (
                f" {phase.to_string()} takes {_get_duration:.2f}s" if _get_duration != TimeTracker.NOT_RECORD else "N/A"
            )
            duration_summaries.append(f"{val}")

        return f"{overhead_title_str}Other overhead details: {','.join(duration_summaries)}\n"


class TrackTime:
    """A function decorator to track time spent in different phases of ORT backend first-time initialization."""

    def __init__(self, phase: ORTModuleInitPhase):
        self.phase = phase

    def __call__(self, func: Callable):
        def wrapper(graph_execution_manager, *args, **kwargs):
            if not hasattr(graph_execution_manager, "time_tracker"):
                raise RuntimeError("The class of the function to be tracked must have a 'time_tracker' attribute.")
            graph_execution_manager.time_tracker.start(self.phase)
            result = func(graph_execution_manager, *args, **kwargs)
            graph_execution_manager.time_tracker.end(self.phase)
            return result

        return wrapper


@contextmanager
def _suppress_os_stream_output(enable=True, on_exit: Optional[Callable] = None):
    """Suppress output from being printed to stdout and stderr.

    If on_exit is not None, it will be called when the context manager exits.
    """
    if enable:
        # stdout and stderr is written to a tempfile instead
        with tempfile.TemporaryFile() as fp:
            old_stdout = None
            old_stderr = None
            try:
                # Store original stdout and stderr file no.
                old_stdout = os.dup(sys.stdout.fileno())
                old_stderr = os.dup(sys.stderr.fileno())

                # Redirect stdout and stderr (printed from Python or C++) to the file.
                os.dup2(fp.fileno(), sys.stdout.fileno())
                os.dup2(fp.fileno(), sys.stderr.fileno())
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()

                # Restore stdout and stderr.
                if old_stdout is not None:
                    os.dup2(old_stdout, sys.stdout.fileno())
                if old_stderr is not None:
                    os.dup2(old_stderr, sys.stderr.fileno())

                # Close file descriptors
                os.close(old_stdout)
                os.close(old_stderr)

                if on_exit:
                    on_exit(fp)

    else:
        yield


def _log_with_filter(logger: logging.Logger, record_filters: Optional[List[str]], name: Optional[str], fo):
    """Log the content by filtering with list of string patterns.
    Args:
        logger: The logger to log the content.
        record_filters: The list of string patterns to filter the content.
            If record_filters is None, the full content will be logged.
        name: The name of log filter.
        fo: The file object to read the content.
    """
    if fo.tell() > 0:
        if logger.disabled:
            return

        fo.seek(0)
        suppress_output_messages = fo.readlines()
        if record_filters:
            filtered_messages = []
            filtered_lines = 0
            for suppressed_message in suppress_output_messages:
                msg = suppressed_message.decode("utf-8")
                found = False
                for warning in record_filters:
                    if warning in msg:
                        found = True
                        filtered_lines += 1
                        break

                if not found:
                    filtered_messages.extend(textwrap.wrap(msg, 180))
            if filtered_messages:
                filtered_messages.insert(0, f"[{name}] Filtered logs ({filtered_lines} records suppressed):")
                logger.warning("\n    ".join(filtered_messages))
        else:
            out_messages = []
            for suppressed_message in suppress_output_messages:
                out_messages.extend(textwrap.wrap(suppressed_message.decode("utf-8"), 180))
            if out_messages:
                out_messages.insert(0, f"[{name}] Full logs:")
                logger.warning("\n    ".join(out_messages))


class SuppressLogs:
    """A function decorator to suppress in different phases of ORT backend first-time initialization."""

    def __init__(self, phase: ORTModuleInitPhase, is_ort_filter=True):
        self.phase = phase
        self.is_ort_filter = is_ort_filter

    def __call__(self, func: Callable):
        def wrapper(graph_execution_manager, *args, **kwargs):
            if not hasattr(graph_execution_manager, "_logger"):
                raise RuntimeError("The class of the function to be tracked must have a '_logger' attribute.")

            if not hasattr(graph_execution_manager, "_debug_options"):
                raise RuntimeError("The class of the function to be tracked must have a '_debug_options' attribute.")

            with _suppress_os_stream_output(
                enable=graph_execution_manager._debug_options.log_level >= LogLevel.INFO,
                on_exit=partial(
                    _log_with_filter,
                    graph_execution_manager._logger,
                    graph_execution_manager._debug_options.onnxruntime_log_filter
                    if self.is_ort_filter
                    else graph_execution_manager._debug_options.torch_exporter_filter,
                    self.phase.to_string(),
                ),
            ):
                result = func(graph_execution_manager, *args, **kwargs)
            return result

        return wrapper
