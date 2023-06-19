# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _options.py

import os
from enum import IntFlag
from functools import reduce
from logging import Logger
from typing import List, Tuple

from onnxruntime.capi import _pybind_state as C
from onnxruntime.training import ortmodule

from ._fallback import _FallbackPolicy
from ._logger import LogLevel
from ._utils import parse_os_env_skip_check_flags


class _SaveOnnxOptions:
    """Configurable option to save ORTModule intermediate onnx models."""

    # class variable
    _path_environment_key = "ORTMODULE_SAVE_ONNX_PATH"

    def __init__(self, save, name_prefix, path: str):
        self._save, self._name_prefix, self._path = self._extract_info(save, name_prefix, path)

    def _extract_info(self, save, name_prefix, path: str):
        # get the destination path from os env variable
        default_path = path if len(path) > 0 else os.getcwd()
        destination_path = os.getenv(_SaveOnnxOptions._path_environment_key, default_path)
        # perform validation only when save is True
        if save:
            self._validate(save, name_prefix, destination_path)
        return save, name_prefix, destination_path

    def _validate(self, save, name_prefix, destination_path):
        # check if directory is writable
        if not os.access(destination_path, os.W_OK):
            raise OSError(
                f"Directory {destination_path} is not writable. Please set the "
                f"{_SaveOnnxOptions._path_environment_key} environment variable to a writable path."
            )

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
    _log_level_environment_key = "ORTMODULE_LOG_LEVEL"

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
    def log_level(self) -> LogLevel:
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

    Raises:
        OSError: If save_onnx is True and output directory is not writable.
        TypeError: If save_onnx is True and name_prefix is not a valid string. Or if
            log_level is not an instance of LogLevel.
        ValueError: If save_onnx is True and name_prefix is an empty string.

    """

    def __init__(self, log_level=LogLevel.WARNING, save_onnx=False, onnx_prefix="", save_path="", config=None):
        self.log_level = log_level
        self.save_onnx = save_onnx
        self.onnx_prefix = onnx_prefix

        self._save_onnx_models = _SaveOnnxOptions(self.save_onnx, self.onnx_prefix, save_path)
        self._logging = _LoggingOptions(self.log_level)

    @property
    def save_onnx_models(self):
        """Accessor for the ONNX saving configuration."""

        return self._save_onnx_models

    @property
    def logging(self):
        """Accessor for the logging configuration."""

        return self._logging


class _SkipCheck(IntFlag):
    """Enumeration to specify which checks should be skipped, allowing faster execution"""

    SKIP_CHECK_DISABLED = 1
    SKIP_CHECK_DEVICE = 2
    SKIP_CHECK_BUILD_GRADIENT = 4
    SKIP_CHECK_EXECUTION_AGENT = 8

    def is_set(self, check):
        """Check whether `check` is set on the `_SkipCheck instance

        SKIP_CHECK_DISABLED implies the check will return False
        """

        return not _SkipCheck.is_disabled(self) and check in self

    def is_disabled(self):
        """Check whether `_SkipCheck.SKIP_CHECK_DISABLED is set on the `_SkipCheck instance"""

        return _SkipCheck.SKIP_CHECK_DISABLED in self


class _RuntimeOptions:
    """Configurable runtime options for ORTModule."""

    def __init__(self, logger: Logger, config=None):
        """Constructor for _RuntimeOptions.

        Initially set all the options to their default values, then override them with the values
        from the environment variables.
        """
        self._logger = logger

        self.onnx_opset_version = ortmodule.ONNX_OPSET_VERSION
        self.conv_algo_search = "HEURISTIC"

        # Configuration for cast optimization.
        # Specify cast propagation strategy. Currently, three strategies are available:
        #  NONE, INSERT-AND-REDUCE and FLOOD-FILL
        # The default is FLOOD_FILL, expand FP16 computation regions in the graph using
        # allowed opcodes for the given level.
        self._propagate_cast_ops_strategy = C.PropagateCastOpsStrategy.FLOOD_FILL
        # Optimize by moving Cast operations if propagate_cast_ops_level is non-negative.
        # - If the _propagate_cast_ops_level is set to zero, then the transformation considers only the opcodes
        #   specified by _propagate_cast_ops_allow as "FP16 safe", to insert/(re)move cast operations before/after
        #   to perform such operations in reduced (16-bit) precision.
        # - If propagate_cast_ops_level is positive, 1 or 2, then in addition to opcode codes specified by
        #   propagate_cast_ops_allow, use onnxruntime predetermined list of opcodes considered safe to move
        #   before/after the cast operation.
        # - Onnxruntime Level 1 predetermined "FP16 safe" opcodes include only opcodes that do not perform
        #   any computation such as Transpose, Split, Reshape, etc., or the computation is actually in Float
        #   such as GeLU, etc.
        # - Whereas Level 2 predetermined "FP16 safe" opcodes include opcodes that perform computation using
        #   contrib ops, Dropout, LayerNormalization, etc.
        self._propagate_cast_ops_level = 1
        # List of opcodes to be considered safe to move before/after the cast operation if propagate_cast_ops_level
        # is zero.
        self._propagate_cast_ops_allow = []

        # default execution order is priority-based for both dynamic/static shape input for now
        # if we observe the benefit of static shape, we can expose this flag to the user
        self._use_static_shape = False

        # flag to enable symbolic shape inference for dynamic shape inputs to improve performance
        self._run_symbolic_shape_infer = True

        # PyTorch custom Autograd function support
        from ._custom_autograd_function import custom_autograd_function_enabler

        self.enable_custom_autograd_function = custom_autograd_function_enabler.state

        self._use_external_gpu_allocator = True

        # WIP feature to enable caching in Gradient accumulation scenario.
        self._enable_grad_acc_optimization = False

        # Memory-aware gradient builder.
        self._use_memory_efficient_gradient = False

        # Configuration for compute optimization.
        self.enable_compute_optimizer = True
        self.enable_sparse_optimizer = True
        self.label_sparsity_ratio = ""
        self.embed_sparsity_ratio = ""
        self._enable_embedding_sparse_optimizer = False  # TODO(pengwa): remove once validation on more models are done.

        # Configuration for memory optimization.
        self.memory_optimizer_config = ""
        self.probe_level = "1"

        # Configuration for dev tools.
        self.print_input_density = False
        self.print_memory_stat = False

        # Configuration for fallback.
        self.fallback_policy = ortmodule.ORTMODULE_FALLBACK_POLICY

        # Configuration for skip check.
        # Indicators of some logic have been executed previously and thus could be skipped for faster training
        # default is enabled, if not defined in os env
        self.skip_check = _SkipCheck(
            _SkipCheck.SKIP_CHECK_DEVICE | _SkipCheck.SKIP_CHECK_BUILD_GRADIENT | _SkipCheck.SKIP_CHECK_EXECUTION_AGENT
        )

        # Override the feature config if it exists in os env.
        self._override_from_env_vars()

    def _override_from_env_vars(self):
        self.onnx_opset_version = os.getenv("ORTMODULE_ONNX_OPSET_VERSION", self.onnx_opset_version)
        self.conv_algo_search = os.getenv("ORTMODULE_CONV_ALGO_SEARCH", self.conv_algo_search)
        if self.conv_algo_search not in ["HEURISTIC", "EXHAUSTIVE"]:
            self._logger.warning("Invalid value of env CONV_ALGO_SEARCH. Must be HEURISTIC or EXHAUSTIVE.")
            self.conv_algo_search = "HEURISTIC"

        # Configuration for compute optimization.
        compute_optimizer_reset = False
        if "ORTMODULE_ENABLE_COMPUTE_OPTIMIZER" in os.environ:
            self.enable_compute_optimizer = os.getenv("ORTMODULE_ENABLE_COMPUTE_OPTIMIZER") == 1
            compute_optimizer_reset = True

        if "ORTMODULE_ENABLE_SPARSE_OPTIMIZER" in os.environ or compute_optimizer_reset:
            if "ORTMODULE_ENABLE_SPARSE_OPTIMIZER" in os.environ:
                self.enable_sparse_optimizer = os.getenv("ORTMODULE_ENABLE_SPARSE_OPTIMIZER") == 1
            self.enable_sparse_optimizer = self.enable_compute_optimizer and self.enable_sparse_optimizer

        # TODO(pengwa): remove once validation on more models are done.
        if "ORTMODULE_ENABLE_EMBEDDING_SPARSE_OPTIMIZER" in os.environ:
            self._enable_embedding_sparse_optimizer = (
                self.enable_sparse_optimizer and os.getenv("ORTMODULE_ENABLE_EMBEDDING_SPARSE_OPTIMIZER") == 1
            )

        # Configuration for memory optimization.
        self.memory_optimizer_config = os.getenv("ORTMODULE_MEMORY_OPT_CONFIG", self.memory_optimizer_config)
        self.probe_level = os.getenv("ORTMODULE_MEMORY_OPT_PROBE_RECOMPUTE_LEVEL", self.probe_level)

        # Configuration for dev tools.
        if "ORTMODULE_PRINT_INPUT_DENSITY" in os.environ:
            self.print_input_density = os.getenv("ORTMODULE_PRINT_INPUT_DENSITY") == 1
        if "ORTMODULE_PRINT_MEMORY_STATS" in os.environ:
            self.print_memory_stat = os.getenv("ORTMODULE_PRINT_MEMORY_STATS") == 1

        # Configuration for fallback.
        if "ORTMODULE_FALLBACK_POLICY" in os.environ:
            self.fallback_policy = os.getenv("ORTMODULE_FALLBACK_POLICY")
            if isinstance(self.fallback_policy, str):
                self.fallback_policy = _FallbackPolicy[self.fallback_policy]

        # Configuration for skip check.
        if "ORTMODULE_SKIPCHECK_POLICY" in os.environ:
            self.skip_check = reduce(
                lambda x, y: x | y,
                [_SkipCheck[name] for name in parse_os_env_skip_check_flags("ORTMODULE_SKIPCHECK_POLICY")],
            )

    def get_feature_map(self) -> List[Tuple[str, bool, str]]:
        feature_map = [
            ("ATen Executor", True, "Dispatch ATen operators to ORT's ATen executor"),
            ("Cast Propagation", self._propagate_cast_ops_level > 0, f"Level {self._propagate_cast_ops_level} enabled"),
            (
                "Custom Function",
                self.enable_custom_autograd_function,
                "Support custom torch.autograd.Function export and execution",
            ),
            (
                "Memory Optimizer",
                len(self.memory_optimizer_config) > 0,
                "Enable with env ORTMODULE_MEMORY_OPT_CONFIG=<config>",
            ),
        ]

        if self.enable_compute_optimizer:
            feature_map.extend(
                [
                    (
                        "Compute Optimizer",
                        self.enable_compute_optimizer,
                        "Enable/Disable with env ORTMODULE_ENABLE_COMPUTE_OPTIMIZER=1/0",
                    ),
                    (
                        " -FLOPReduction",
                        self.enable_compute_optimizer,
                        "Reduce FLOPs by upstreaming shrinking-sized ops",
                    ),
                ]
            )

            if len(self.label_sparsity_ratio) > 0:
                feature_map.append((" -LabelSparsityOpt", True, f"Input density: {self.label_sparsity_ratio}"))

            if len(self.embed_sparsity_ratio) > 0:
                feature_map.append((" -EmbedSparsityOpt", True, f"Input density: {self.embed_sparsity_ratio}"))

        feature_map.append(
            (
                "Auto Fallback",
                self.fallback_policy is not _FallbackPolicy.FALLBACK_DISABLE,
                "Fallback to PyTorch when encountering unsupported ops",
            )
        )

        return feature_map
