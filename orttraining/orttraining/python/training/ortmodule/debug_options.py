# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# debug_options.py

import os

from ._logger import LogLevel

class _SaveOnnxOptions:
    """Configurable option to save ORTModule intermediate onnx models."""

    def __init__(self, save, prefix):
        self._dir_env_key = 'ORTModuleSaveONNXDirectory'
        self._save, self._prefix, self._directory = self._extract_info(save, prefix)

    def _extract_info(self, save, prefix):
        # get the destination directory from os env variable
        dst_dir = os.getenv(self._dir_env_key, os.getcwd())
        self._validate(save, prefix, dst_dir)
        return save, prefix, dst_dir

    def _validate(self, save, prefix, dst_dir):
        # check if directory is writable
        if not os.access(dst_dir, os.W_OK):
            raise OSError(f"Directory {dst_dir} is not writable. Please set the {self._dir_env_key} environment variable to a writable directory.")

        # check if input prefix is a string
        if not isinstance(prefix, str):
            raise TypeError(f"Expected prefix of type str, got {type(prefix)}.")

        # if save_onnx is set, save_onnx_prefix must be a non empty string
        if save:
            if not prefix:
                raise ValueError("save_onnx_prefix must be provided when save_onnx is set.")

    @property
    def save(self):
        return self._save

    @property
    def prefix(self):
        return self._prefix

    @property
    def directory(self):
        return self._directory


class _LoggingOptions:
    """Configurable option to set the log level in ORTModule."""

    def __init__(self, log_level):
        self._log_level_mapping = {
            "VERBOSE": LogLevel.VERBOSE,
            "INFO": LogLevel.INFO,
            "WARNING": LogLevel.WARNING,
            "ERROR": LogLevel.ERROR,
            "FATAL": LogLevel.FATAL
        }
        self._log_level_env_key = 'ORTModuleLogLevel'
        self._log_level = self._extract_info(log_level)

    def _extract_info(self, log_level):
        # get the log_level from os env variable
        # os env variable log level supercededs the locally provided one
        log_level = self._log_level_mapping.get(os.getenv(self._log_level_env_key), log_level)
        self._validate(log_level)
        return log_level

    def _validate(self, log_level):
        # check if log_level is an instance of LogLevel
        if not isinstance(log_level, LogLevel):
            raise TypeError(f"Expected log_level of type LogLevel, got {type(log_level)}.")

    @property
    def log_level(self):
        return self._log_level

class DebugOptions:
    """Configurable debugging options for ORTModule.

    DebugOptions provides a way to configure ORTModule debug flags.
    :param log_level: Configure ORTModule log level
    :type log_level: class: `LogLevel`
    :param save_onnx: Configure saving of ORTModule ONNX models.
    :type save_onnx: bool
    :param save_onnx_prefix: Prefix to the ORTModule ONNX models saved file names.
    :type save_onnx_prefix: str
    """

    def __init__(self, log_level=LogLevel.WARNING, save_onnx=False, save_onnx_prefix=''):
        self._save_onnx_models = _SaveOnnxOptions(save_onnx, save_onnx_prefix)
        self._logging = _LoggingOptions(log_level)

    @property
    def save_onnx_models(self):
        """Accessor for the save_onnx_models debug flag."""
        return self._save_onnx_models

    @property
    def logging(self):
        """Accessor for the logging debug flag."""
        return self._logging
