# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# debug_options.py

from enum import IntEnum
import os
from typing import Type

from ._configuration_option import ConfigurationOption

class SaveOnnxOptions(ConfigurationOption):
    """Configurable option to save ORTModule intermediate onnx models."""

    def __init__(self):
        self.reset()

    def configure(self, save=True, prefix='', dst_directory=os.getcwd()):
        self._validate(save, prefix, dst_directory)
        self._save = save
        self._prefix = prefix
        self._directory = dst_directory

    def reset(self):
        self._save = False
        self._prefix = ''
        self._directory = os.getcwd()

    def _validate(self, save, prefix, dst_directory):
        # check if directory is writable
        if not os.access(dst_directory, os.W_OK):
            raise OSError(f"Directory {dst_directory} is not writable.")

        # check if input prefix is a string
        if not isinstance(prefix, str):
            raise TypeError(f"Expected prefix of type str, got {type(prefix)}.")

    @property
    def save(self):
        return self._save

    @property
    def prefix(self):
        return self._prefix

    @property
    def directory(self):
        return self._directory


class LogLevel(IntEnum):
    VERBOSE = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    FATAL = 4

class LoggingOptions(ConfigurationOption):
    """Configurable option to set the log level in ORTModule."""

    def __init__(self):
        self.reset()

    def configure(self, loglevel=LogLevel.INFO):
        self._validate(loglevel)
        self._loglevel = loglevel

    def reset(self):
        self._loglevel = LogLevel.WARNING

    def _validate(self, loglevel):
        if not isinstance(loglevel, LogLevel):
            raise TypeError(f"Expected loglevel of type LogLevel, got {type(loglevel)}.")

    @property
    def loglevel(self):
        return self._loglevel

class DebugOptions:
    """Configurable debugging options for ORTModule"""

    def __init__(self):
        self._save_intermediate_onnx_models = SaveOnnxOptions()
        self._logging = LoggingOptions()

    @property
    def save_intermediate_onnx_models(self):
        """Configurable option to save ORTModule intermediate onnx models."""
        return self._save_intermediate_onnx_models

    @property
    def logging(self):
        """Configurable option to set the log level in ORTModule."""
        return self._logging
