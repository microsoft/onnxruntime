# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# runtime_options.py

import os

from onnxruntime.training import ortmodule

from ._fallback import _FallbackPolicy


class _ExporterOptions:
    """Configurable option for PyTorch ONNX exporter."""

    _path_environment_key = "ORTMODULE_SAVE_ONNX_PATH"

    def __init__(
        self,
        opset_version,
        do_constant_folding,
        export_modules_as_functions,
        export_params,
        keep_initializers_as_inputs,
        use_static_shapes,
        exporter_extra_args,
    ):
        self._opset_version, self._export_modules_as_functions, self._exporter_extra_args = self._validate(
            opset_version, export_modules_as_functions, exporter_extra_args
        )

        self._do_constant_folding = self._validate_and_extract_flag(do_constant_folding, "do_constant_folding")
        self._export_params = self._validate_and_extract_flag(export_params, "export_params")
        self._keep_initializers_as_inputs = self._validate_and_extract_flag(
            keep_initializers_as_inputs, "keep_initializers_as_inputs"
        )
        self._use_static_shapes = self._validate_and_extract_flag(use_static_shapes, "use_static_shapes")

    def _validate_and_extract_flag(self, flag, str):
        # check if flag is an instance of Boolean
        if not isinstance(flag, bool):
            raise TypeError(f"Expected {str} of type bool, got {type(flag)}.")
        return flag

    def _validate(self, opset_version, export_modules_as_functions, exporter_extra_args):
        # check if opset_version is an int
        if not isinstance(opset_version, int):
            raise TypeError(f"Expected opset_version of type int, got {type(opset_version)}.")

        if export_modules_as_functions and opset_version < 15:
            raise ValueError(
                "`export_modules_as_functions` is not supported for `opset_version` < 15."
                "This is because `opset_version` < 15 implies IR version < 8, which means "
                "no local function support. "
            )

        is_bool = isinstance(export_modules_as_functions, bool)
        is_set = isinstance(export_modules_as_functions, set) and len(export_modules_as_functions) > 0
        if not (is_bool or is_set):
            raise TypeError(
                "Expected export_modules_as_functions of type bool or set of type of nn.Module,"
                f"got {type(export_modules_as_functions)}."
            )
        if is_set:
            for v in export_modules_as_functions:
                if not isinstance(v, type):
                    raise TypeError(f"Expected export_modules_as_functions item of type of nn.Module, got {type(v)}.")

        if not isinstance(exporter_extra_args, dict):
            raise TypeError(f"Expected exporter_extra_args of type dict, got {type(exporter_extra_args)}.")

        return opset_version, export_modules_as_functions, exporter_extra_args

    @property
    def opset_version(self):
        """Accessor for ONNX opset version configuration."""

        return self._opset_version

    @property
    def do_constant_folding(self):
        """Accessor for exporter flag to enable constant folding."""

        return self._do_constant_folding

    @property
    def export_modules_as_functions(self):
        """Accessor for exporter option to export modules as functions."""

        return self._export_modules_as_functions

    @property
    def export_params(self):
        """Accessor for exporter flag to export parameters."""

        return self._export_params

    @property
    def keep_initializers_as_inputs(self):
        """Accessor for exporter flag to keep initializers as inputs."""

        return self._keep_initializers_as_inputs

    @property
    def use_static_shapes(self):
        """Accessor for exporter flag to use static shapes."""

        return self._use_static_shapes

    @property
    def exporter_extra_args(self):
        """Accessor for extra arguments to the exporter."""

        return self._exporter_extra_args


class RuntimeOptions:
    """Configurable runtime options for ORTModule.

    Args:
        save_onnx (:obj:`bool`, optional): Configure ORTModule to save onnx models. Defaults to False.
            The output directory of the onnx models by default is set to the current working directory.
            To change the output directory, the environment variable "ORTMODULE_SAVE_ONNX_PATH" can be
            set to the destination directory path.

    Raises:
        OSError: If save_onnx is True and output directory is not writable.
        TypeError: If save_onnx is True and name_prefix is not a valid string. Or if
            log_level is not an instance of LogLevel.
        ValueError: If save_onnx is True and name_prefix is an empty string.

    """

    def __init__(
        self,
        opset_version=ortmodule.ONNX_OPSET_VERSION,
        do_constant_folding=False,
        export_modules_as_functions=False,
        export_params=False,
        keep_initializers_as_inputs=True,
        use_static_shapes=False,
        export_extra_args={},
        enable_custom_autograd=False,
        disable_custom_ops=False,
        graph_optimization_level=None,
        fallback_policy=ortmodule.ORTMODULE_FALLBACK_POLICY,
        fallback_retry=ortmodule.ORTMODULE_FALLBACK_RETRY,
        skipcheck_policy=ortmodule.ORTMODULE_SKIPCHECK_POLICY,
    ):
        self._exporter_options = _ExporterOptions(
            opset_version,
            do_constant_folding,
            export_modules_as_functions,
            export_params,
            keep_initializers_as_inputs,
            use_static_shapes,
            export_extra_args,
        )

        if enable_custom_autograd:
            ortmodule._custom_autograd_function()

        self._disable_custom_ops = disable_custom_ops
        self._graph_optimization_level = graph_optimization_level
        self._fallback_policy = fallback_policy
        self._fallback_retry = fallback_retry
        self._skipcheck_policy = skipcheck_policy

    @property
    def exporter_options(self):
        """Accessor for the exporter configuration."""

        return self._exporter_options

    @property
    def disable_custom_ops(self):
        """Accessor for the ORTModule flag to disable custom ops."""

        return self._disable_custom_ops

    @property
    def graph_optimization_level(self):
        """Accessor for the ORTModule graph optimization level."""

        return self._graph_optimization_level

    @property
    def fallback_policy(self):
        """Accessor for the ORTModule fallback policy."""

        return self._fallback_policy

    @property
    def fallback_retry(self):
        """Accessor for the ORTModule option for fallback retry."""

        return self._fallback_retry

    @property
    def skipcheck_policy(self):
        """Accessor for the ORTModule skipcheck policy."""

        return self._skipcheck_policy
