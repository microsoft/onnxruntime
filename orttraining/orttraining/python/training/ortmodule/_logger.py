# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import io
import logging
import sys
import time
from contextlib import contextmanager
from enum import IntEnum
from typing import Dict, List

from onnxruntime.capi._pybind_state import Severity


class LogLevel(IntEnum):
    VERBOSE = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    FATAL = 4


@contextmanager
def suppress_os_stream_output(suppress_stdout=True, suppress_stderr=True, log_level=LogLevel.WARNING):
    """Suppress output from being printed to stdout and stderr if log_level is WARNING or higher.

    If there is any output detected, a single warning is issued in the context
    """

    # stdout and stderr is written to a tempfile instead
    stdout = sys.stdout
    stderr = sys.stderr

    suppress_logs = log_level >= LogLevel.WARNING

    fo = io.StringIO()

    try:
        if suppress_stdout and suppress_logs:
            sys.stdout = fo
        if suppress_stderr and suppress_logs:
            sys.stderr = fo
        yield fo
    finally:
        if suppress_stdout:
            sys.stdout = stdout
        if suppress_stderr:
            sys.stderr = stderr


ORTMODULE_LOG_LEVEL_MAP: Dict[LogLevel, List[int]] = {
    LogLevel.VERBOSE: [Severity.VERBOSE, logging.DEBUG],
    LogLevel.INFO: [Severity.INFO, logging.INFO],
    LogLevel.WARNING: [Severity.WARNING, logging.WARNING],
    LogLevel.ERROR: [Severity.ERROR, logging.ERROR],
    LogLevel.FATAL: [Severity.FATAL, logging.FATAL],
}


def ortmodule_loglevel_to_onnxruntime_c_loglevel(loglevel: LogLevel) -> int:
    return ORTMODULE_LOG_LEVEL_MAP.get(loglevel, [Severity.WARNING, logging.WARNING])[0]


def ortmodule_loglevel_to_python_loglevel(loglevel: LogLevel) -> int:
    return ORTMODULE_LOG_LEVEL_MAP.get(loglevel, [Severity.WARNING, logging.WARNING])[1]


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


class TimeTrackerPhase(IntEnum):
    EndToEnd = 0
    EXPORT = 1
    GRAPH_BUILDER_INIT = 2
    DETECTION = 3
    BUILD_GRAPH = 4
    CREATE_SESSION = 5

    def to_string(self) -> str:
        if self == TimeTrackerPhase.EndToEnd:  # The total overhead of ORT first-time initialization
            return "end to end"
        if self == TimeTrackerPhase.EXPORT:
            # The latency of preparing and exporting the model to ONNX
            return "export"
        elif self == TimeTrackerPhase.GRAPH_BUILDER_INIT:
            # The latency of initializing the graph builder
            return "graph builder init"
        elif self == TimeTrackerPhase.DETECTION:
            # The latency of runtime detection
            return "runtime detection"
        elif self == TimeTrackerPhase.BUILD_GRAPH:
            # The latency of optimizing forward graph (and building the gradient graph for training).
            return "building grad"
        elif self == TimeTrackerPhase.CREATE_SESSION:
            # The latency of creating the session
            return "session creation"
        else:
            return "invalid"


class TimeTracker:
    """A simple class to track time spent in different phases of ORT backend first-time initialization."""

    NOT_RECORD = -1.0

    def __init__(self):
        self.starts_: List[float] = [TimeTracker.NOT_RECORD] * len(TimeTrackerPhase)
        self.ends_: List[float] = [TimeTracker.NOT_RECORD] * len(TimeTrackerPhase)

    def start(self, phase: TimeTrackerPhase):
        self.starts_[phase] = time.time()

    def end(self, phase: TimeTrackerPhase):
        self.ends_[phase] = time.time()

    def duration(self, phase: TimeTrackerPhase):
        if self.ends_[phase] == TimeTracker.NOT_RECORD or self.starts_[phase] == TimeTracker.NOT_RECORD:
            return TimeTracker.NOT_RECORD
        return self.ends_[phase] - self.starts_[phase]

    def to_string(self, log_details=False) -> str:
        end_to_end_str = self.duration(TimeTrackerPhase.EndToEnd)
        end_to_end_str = f"{end_to_end_str * 1000:.0f}" if end_to_end_str != TimeTracker.NOT_RECORD else "N/A"
        export_str = self.duration(TimeTrackerPhase.EXPORT)
        export_str = f"{export_str * 1000:.0f}" if export_str != TimeTracker.NOT_RECORD else "N/A"
        overhead_title_str = f"Total overhead: {end_to_end_str}ms where export takes {export_str}ms.\n"

        if log_details is False:
            return overhead_title_str

        duration_summaries = []
        for phase in TimeTrackerPhase:
            duration = self.duration(phase)
            if phase in [TimeTrackerPhase.EndToEnd, TimeTrackerPhase.EXPORT]:
                continue

            val = f" {phase.to_string()} takes {duration * 1000:.0f}ms" if duration != TimeTracker.NOT_RECORD else "N/A"
            duration_summaries.append(f"{val}")

            return f"{overhead_title_str}Other overhead details: {','.join(duration_summaries)}\n"


@contextmanager
def collect_timer(tracker: TimeTracker, phase: TimeTrackerPhase):
    try:
        tracker.start(phase)
        yield
    finally:
        tracker.end(phase)
