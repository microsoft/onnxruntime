# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# options.py

import os
from enum import IntFlag
from functools import reduce
from logging import Logger

from packaging import version

from onnxruntime.capi import _pybind_state as C
from onnxruntime.training import ortmodule

from ._fallback import _FallbackPolicy
from ._logger import LogLevel
from ._utils import get_runtime_pytorch_version, parse_os_env_skip_check_flags


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

    @property
    def torch_exporter_filter(self):
        """Accessor for the filter export logs configuration."""
        torch_version = get_runtime_pytorch_version()
        if self.log_level > LogLevel.DEVINFO:
            if torch_version < version.parse("2.0"):
                return [
                    # WARNING: The shape inference of com.microsoft::SoftmaxCrossEntropyLossInternal type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
                    # WARNING: The shape inference of com.microsoft::PythonOp type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
                    # WARNING: The shape inference of org.pytorch.aten::ATen type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
                    # WARNING: The shape inference of prim::Constant type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
                    "type is missing, so it may result in wrong shape inference",
                    # Warning: Checker does not support models with experimental ops: ATen
                    "Checker does not support models with experimental ops:",
                    "Dropout is a training op and should not be exported in inference mode.",
                    # Warning: Shape inference does not support models with experimental operators: ATen
                    "Shape inference does not support models with experimental operators:",
                    # Warning: Unsupported operator Trilu. No schema registered for this operator.
                    # Warning: Unsupported operator ATen. No schema registered for this operator.
                    # Warning: Unsupported operator SoftmaxCrossEntropyLossInternal. No schema registered for this operator.
                    "No schema registered for this operator.",
                ]
            return [
                # [W shape_type_inference.cpp:1974] Warning: The shape inference of com.microsoft::PythonOp type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function. (function UpdateReliable)
                "type is missing, so it may result in wrong shape inference",
                #  diagnostics [WARNING] - None
                "[WARNING] - None",
            ]

        return None

    @property
    def onnxruntime_log_filter(self):
        """Accessor for the filter onnxruntime logs configuration."""
        return None


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

    def __init__(self, logger: Logger):
        """Constructor for Options.

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
        self.propagate_cast_ops_strategy = C.PropagateCastOpsStrategy.FLOOD_FILL
        # Optimize by moving Cast operations if propagate_cast_ops_level is non-negative.
        # - If the propagate_cast_ops_level is set to zero, then the transformation considers only the opcodes
        #   specified by propagate_cast_ops_allow as "FP16 safe", to insert/(re)move cast operations before/after
        #   to perform such operations in reduced (16-bit) precision.
        # - If propagate_cast_ops_level is positive, 1 or 2, then in addition to opcode codes specified by
        #   propagate_cast_ops_allow, use onnxruntime predetermined list of opcodes considered safe to move
        #   before/after the cast operation.
        # - Onnxruntime Level 1 predetermined "FP16 safe" opcodes include only opcodes that do not perform
        #   any computation such as Transpose, Split, Reshape, etc., or the computation is actually in Float
        #   such as GeLU, etc.
        # - Whereas Level 2 predetermined "FP16 safe" opcodes include opcodes that perform computation using
        #   contrib ops, Dropout, LayerNormalization, etc.
        self.propagate_cast_ops_level = 1
        # List of opcodes to be considered safe to move before/after the cast operation if propagate_cast_ops_level
        # is zero.
        self.propagate_cast_ops_allow = []

        # default execution order is priority-based for both dynamic/static shape input for now
        # if we observe the benefit of static shape, we can expose this flag to the user
        self.use_static_shape = False

        # flag to enable symbolic shape inference for dynamic shape inputs to improve performance
        self.run_symbolic_shape_infer = True

        # PyTorch custom Autograd function support
        from ._custom_autograd_function import custom_autograd_function_enabler

        self.enable_custom_autograd_function = custom_autograd_function_enabler.state

        self.use_external_gpu_allocator = True

        # WIP feature to enable caching in Gradient accumulation scenario.
        self.enable_grad_acc_optimization = False

        # Memory-aware gradient builder.
        self.use_memory_efficient_gradient = False

        # Configuration for compute optimization.
        self.enable_compute_optimizer = True
        self.enable_sparse_optimizer = True
        self.label_sparsity_ratio = ""
        self.embed_sparsity_ratio = ""
        self.enable_embedding_sparse_optimizer = False  # TODO(pengwa): remove once validation on more models are done.

        # Configuration for memory optimization.
        self.memory_optimizer_config = ""
        self.probe_level = "1"

        # Configuration for dev tools.
        self.print_input_density = False
        self.print_memory_stat_by_step = False

        # Configuration for fallback.
        self.fallback_policy = ortmodule.ORTMODULE_FALLBACK_POLICY

        # Configuration for skip check.
        # Indicators of some logic have been executed previously and thus could be skipped for faster training
        # default is enabled, if not defined in os env
        self.skip_check = _SkipCheck(
            _SkipCheck.SKIP_CHECK_DEVICE | _SkipCheck.SKIP_CHECK_BUILD_GRADIENT | _SkipCheck.SKIP_CHECK_EXECUTION_AGENT
        )

        # Triton support.
        self.enable_triton = False
        self.enable_tuning = False
        self.max_tuning_duration_ms = 0
        self.tuning_results_path = ""

        # Cache exported model
        self.ortmodule_cache_dir = ""

        # Experimental features.
        self.enable_zero_stage3_support = False  # Once enabled, cannot be disabled.

        self.do_deepcopy_before_model_export = True

        # Override the feature config if it exists in os env.
        self._override_from_env_vars()

    def _override_from_env_vars(self):
        self.onnx_opset_version = int(os.getenv("ORTMODULE_ONNX_OPSET_VERSION", self.onnx_opset_version))
        self.conv_algo_search = os.getenv("ORTMODULE_CONV_ALGO_SEARCH", self.conv_algo_search)
        if self.conv_algo_search not in ["HEURISTIC", "EXHAUSTIVE"]:
            self._logger.warning("Invalid value of env CONV_ALGO_SEARCH. Must be HEURISTIC or EXHAUSTIVE.")
            self.conv_algo_search = "HEURISTIC"

        # Configuration for compute optimization.
        compute_optimizer_reset = False
        if "ORTMODULE_ENABLE_COMPUTE_OPTIMIZER" in os.environ:
            self.enable_compute_optimizer = int(os.getenv("ORTMODULE_ENABLE_COMPUTE_OPTIMIZER")) == 1
            compute_optimizer_reset = True

        if "ORTMODULE_ENABLE_SPARSE_OPTIMIZER" in os.environ or compute_optimizer_reset:
            if "ORTMODULE_ENABLE_SPARSE_OPTIMIZER" in os.environ:
                self.enable_sparse_optimizer = int(os.getenv("ORTMODULE_ENABLE_SPARSE_OPTIMIZER")) == 1
            self.enable_sparse_optimizer = self.enable_compute_optimizer and self.enable_sparse_optimizer

        # TODO(pengwa): remove once validation on more models are done.
        if "ORTMODULE_ENABLE_EMBEDDING_SPARSE_OPTIMIZER" in os.environ:
            self.enable_embedding_sparse_optimizer = (
                self.enable_sparse_optimizer and int(os.getenv("ORTMODULE_ENABLE_EMBEDDING_SPARSE_OPTIMIZER")) == 1
            )

        # Configuration for memory optimization.
        self.memory_optimizer_config = os.getenv("ORTMODULE_MEMORY_OPT_CONFIG", self.memory_optimizer_config)
        self.probe_level = os.getenv("ORTMODULE_MEMORY_OPT_PROBE_RECOMPUTE_LEVEL", self.probe_level)

        # Configuration for dev tools.
        if "ORTMODULE_PRINT_INPUT_DENSITY" in os.environ:
            self.print_input_density = int(os.getenv("ORTMODULE_PRINT_INPUT_DENSITY")) == 1
        if "ORTMODULE_PRINT_MEMORY_STATS" in os.environ:
            self.print_memory_stat_by_step = int(os.getenv("ORTMODULE_PRINT_MEMORY_STATS")) == 1

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

        # Configuration for Triton.
        # Enable Triton op executor if Triton is installed, backend has support and environment variable is set.
        if (
            "ORTMODULE_USE_TRITON" in os.environ
            and int(os.getenv("ORTMODULE_USE_TRITON")) == 1
            and C.is_triton_enabled()
        ):
            try:
                import triton  # noqa: F401
            except ImportError:
                pass
            else:
                self.enable_triton = True

        if "ORTMODULE_ENABLE_TUNING" in os.environ and int(os.getenv("ORTMODULE_ENABLE_TUNING")) == 1:
            self.enable_tuning = True
        if "ORTMODULE_MAX_TUNING_DURATION_MS" in os.environ:
            max_tuning_duration_ms = int(os.getenv("ORTMODULE_MAX_TUNING_DURATION_MS"))
            if max_tuning_duration_ms > 0:
                self.max_tuning_duration_ms = max_tuning_duration_ms
        if "ORTMODULE_TUNING_RESULTS_PATH" in os.environ:
            self.tuning_results_path = os.getenv("ORTMODULE_TUNING_RESULTS_PATH")

        # Cache exported model
        if "ORTMODULE_CACHE_DIR" in os.environ:
            self._logger.warning("ORTModule optimization for caching exported model is ON.")
            self.ortmodule_cache_dir = os.getenv("ORTMODULE_CACHE_DIR")

        # Experimental features.
        if "ORTMODULE_ENABLE_ZERO_STAGE3" in os.environ and int(os.getenv("ORTMODULE_ENABLE_ZERO_STAGE3")) == 1:
            self.enable_zero_stage3_support = True

        if "ORTMODULE_DO_DEEPCOPY_BEFORE_MODEL_EXPORT" in os.environ:
            self.do_deepcopy_before_model_export = int(os.getenv("ORTMODULE_DO_DEEPCOPY_BEFORE_MODEL_EXPORT")) == 1
