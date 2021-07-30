# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# debug_options.py

import os

from ._logger import LogLevel
from ._fallback import _FallbackPolicy


class _SaveOnnxOptions:
    """Configurable option to save ORTModule intermediate onnx models."""

    # class variable
    _path_environment_key = 'ORTMODULE_SAVE_ONNX_PATH'

    def __init__(self, save, name_prefix):
        self._save, self._name_prefix, self._path = self._extract_info(save, name_prefix)

    def _extract_info(self, save, name_prefix):
        # get the destination path from os env variable
        destination_path = os.getenv(_SaveOnnxOptions._path_environment_key, os.getcwd())
        # perform validation only when save is True
        if save:
            self._validate(save, name_prefix, destination_path)
        return save, name_prefix, destination_path

    def _validate(self, save, name_prefix, destination_path):
        # check if directory is writable
        if not os.access(destination_path, os.W_OK):
            raise OSError(f"Directory {destination_path} is not writable. Please set the {_SaveOnnxOptions._path_environment_key} environment variable to a writable path.")

        # check if input prefix is a string
        if not isinstance(name_prefix, str):
            raise TypeError(f"Expected name prefix of type str, got {type(name_prefix)}.")

        # if save_onnx is set, save_onnx_prefix must be a non empty string
        if not name_prefix:
            raise ValueError("onnx_prefix must be provided when save_onnx is set.")

    @property
    def save(self):
        return self._save

    @property
    def name_prefix(self):
        return self._name_prefix

    @property
    def path(self):
        return self._path


class _LoggingOptions:
    """Configurable option to set the log level in ORTModule."""

    # class variable
    _log_level_environment_key = 'ORTMODULE_LOG_LEVEL'

    def __init__(self, log_level):
        self._log_level = self._extract_info(log_level)

    def _extract_info(self, log_level):
        # get the log_level from os env variable
        # OS environment variable log level superseeds the locally provided one
        self._validate(log_level)
        log_level = LogLevel[os.getenv(_LoggingOptions._log_level_environment_key, log_level.name)]
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

    Args:
        log_level (:obj:`LogLevel`, optional): Configure ORTModule log level. Defaults to LogLevel.WARNING.
            log_level can also be set by setting the environment variable "ORTMODULE_LOG_LEVEL" to one of
            "VERBOSE", "INFO", "WARNING", "ERROR", "FATAL". In case both are set, the environment variable
            takes precedence.
        save_onnx (:obj:`bool`, optional): Configure ORTModule to save onnx models. Defaults to False.
            The output directory of the onnx models by default is set to the current working directory.
            To change the output directory, the environment variable "ORTMODULE_SAVE_ONNX_PATH" can be
            set to the destination directory path.
        onnx_prefix (:obj:`str`, optional): Name prefix to the ORTModule ONNX models saved file names.
            Must be provided if save_onnx is True
        fallback_policy (`_FallbackPolicy`, optional): Policy to configure PyTorch fallback strategy.
            Defaults to FALLBACK_DISABLE

    Raises:
        OSError: If save_onnx is True and output directory is not writable.
        TypeError: If save_onnx is True and name_prefix is not a valid string. Or if
            log_level is not an instance of LogLevel.
        ValueError: If save_onnx is True and name_prefix is an empty string.

    """

    def __init__(self,
                 log_level=LogLevel.WARNING,
                 save_onnx=False,
                 onnx_prefix='',
                 fallback_policy=_FallbackPolicy.FALLBACK_DISABLE):
        self._save_onnx_models = _SaveOnnxOptions(save_onnx, onnx_prefix)
        self._logging = _LoggingOptions(log_level)
        self._fallback_policy = fallback_policy

    @property
    def save_onnx_models(self):
        """Accessor for the ONNX saving configuration."""

        return self._save_onnx_models

    @property
    def logging(self):
        """Accessor for the logging configuration."""

        return self._logging

    @property
    def fallback_policy(self):
        """Accessor for the Fallback policy."""

        return self._fallback_policy