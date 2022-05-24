# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# runtime_options.py

from onnxruntime import GraphOptimizationLevel
from onnxruntime.training import ortmodule


class _ExporterOptions:
    """Configurable option for PyTorch ONNX exporter."""

    _path_environment_key = "ORTMODULE_SAVE_ONNX_PATH"

    def __init__(
        self,
        onnx_opset_version,
        do_constant_folding,
        export_modules_as_functions,
        export_params,
        keep_initializers_as_inputs,
        use_static_shapes,
        exporter_extra_args,
    ):
        self._onnx_opset_version, self._export_modules_as_functions, self._exporter_extra_args = self._validate(
            onnx_opset_version, export_modules_as_functions, exporter_extra_args
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

    def _validate(self, onnx_opset_version, export_modules_as_functions, exporter_extra_args):
        # check if onnx_opset_version is an int
        if not isinstance(onnx_opset_version, int):
            raise TypeError(f"Expected onnx_opset_version of type int, got {type(onnx_opset_version)}.")

        if export_modules_as_functions and onnx_opset_version < 15:
            raise ValueError(
                "`export_modules_as_functions` is not supported for `onnx_opset_version` < 15."
                "This is because `onnx_opset_version` < 15 implies IR version < 8, which means "
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

        return onnx_opset_version, export_modules_as_functions, exporter_extra_args

    @property
    def onnx_opset_version(self):
        """Accessor for ONNX opset version configuration."""

        return self._onnx_opset_version

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
        onnx_opset_version (:obj:`int`, optional, default 14): Configure ORTModule exporter to use this ONNX opset version..
            To change the opset version, the environment variable "ORTMODULE_ONNX_OPSET_VERSION" can also be used.

        do_constant_folding (:obj:`bool`, optional): Configure ORTModule export to apply the constant-folding optimization.
            Defaults to False. Constant-folding will replace some of the ops that have all constant inputs with pre-computed
            constant nodes.

        export_modules_as_functions (:obj:`bool` or set of python:type of nn.Module, optional, default False): ORTModule exporter
            flag to enable exporting all nn.Module forward calls as local functions in ONNX. Or a set to indicate the particular
            types of modules to export as local functions in ONNX. This feature requires onnx_opset_version >= 15, otherwise the
            export will fail. This is because onnx_opset_version < 15 implies IR version < 8, which means no local function support.

        export_params (:obj:`bool`, optional, default False): ORTModule exporter flag to export all model parameters.
            If True, all parameters will be exported. Set this to False if you want to export an untrained model. In this case,
            the exported model will first take all of its parameters as arguments.

        keep_initializers_as_inputs (:obj:`bool`, optional, default True): ORTModule exporter flag to keep initializers as inputs.
            If True, all the initializers (typically corresponding to parameters) in the exported graph will also be added as inputs
            to the graph. If False, initializers are not added as inputs to the graph, and only the non-parameter inputs are added as inputs.
            This may allow for better optimizations (e.g. constant folding) by backends/runtimes.

        use_static_shapes (:obj:`bool`, optional, default False): ORTModule exporter flag to use static output shapes.
            By default the exported model will have dynamic shapes for all input and output tensors.

        export_extra_args (:obj:`dict`, optional, default Empty): Extra arguments to specify to the ORTModule Torch ONNX exporter.
            Refer to https://pytorch.org/docs/stable/onnx.html#torch.onnx.export for documentation on exporter arguments.

        enable_custom_autograd (:obj:`bool`, optional, default False): Enable custom autograd.Function support for ORTModule.
            This will enable ``autograd.Function``s to be exported as ``PythonOp`` in ONNX.

        disable_custom_ops (:obj:`bool`, optional, default False): Disable custom ops support in ORTModule.
            This flag will disable custom ATen ops and custom gradients in ORTModule.

        graph_optimization_level (:obj:`onnxruntime.GraphOptimizationLevel`, optional, default None): Choose the graph optimization
            level for ORTModule runtime session. The available graph optimization levels are:
            ``onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL``: Disable all optimizations
            ``onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC``: Constant folding and other optimizations that only use ONNX operators
            ``onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED``: Optimizations using custom operators, excluding NCHWc and NHWC layout optimizers
            ``onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL``: Enable all optimizations (default)

    Raises:
        TypeError: If an invalid type for a runtime option is specified.
        ValueError: If an invalid runtime option value is specified.

    """

    def __init__(
        self,
        onnx_opset_version=ortmodule.ONNX_OPSET_VERSION,
        do_constant_folding=False,
        export_modules_as_functions=False,
        export_params=False,
        keep_initializers_as_inputs=True,
        use_static_shapes=False,
        export_extra_args={},
        enable_custom_autograd=False,
        disable_custom_ops=False,
        graph_optimization_level=None,
    ):
        self._exporter_options = _ExporterOptions(
            onnx_opset_version,
            do_constant_folding,
            export_modules_as_functions,
            export_params,
            keep_initializers_as_inputs,
            use_static_shapes,
            export_extra_args,
        )

        if enable_custom_autograd:
            ortmodule._custom_autograd_function.enable_custom_autograd_support()

        self._disable_custom_ops = disable_custom_ops

        if not isinstance(graph_optimization_level, GraphOptimizationLevel):
            raise TypeError(
                "Expected graph_optimization_level of type onnxruntime.GraphOptimizationLevel, "
                f"got {type(graph_optimization_level)}."
            )

        self._graph_optimization_level = graph_optimization_level

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
