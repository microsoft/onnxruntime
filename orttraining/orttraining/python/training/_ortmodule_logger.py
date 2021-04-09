# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from contextlib import contextmanager
from enum import IntEnum
import io
import sys
import warnings

class LogLevel(IntEnum):
    VERBOSE = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    FATAL = 4


@contextmanager 
def suppress_os_stream_output(suppress_stdout=False, suppress_stderr=False, log_level=LogLevel.VERBOSE):
    """Supress output from being printed to stdout and stderr if log_level is WARNING or higher.

    If there is any output detected, a single warning is issued at of the context
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
        yield
    finally:
        if suppress_stdout:
            sys.stdout = stdout
        if suppress_stderr:
            sys.stderr = stderr

        if fo.tell() > 0 and suppress_logs:
            # If anything was captured in fo, raise a single user warning letting users know that there was
            # some warning or error that was raised
            warnings.warn("There were one or more warnings or errors raised while exporting the PyTorch "
            "model. Please enable INFO level logging to view all warnings and errors.", UserWarning)
