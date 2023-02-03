# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import copy
import inspect
import io
import os
import warnings
from abc import ABC, abstractmethod  # noqa: F401
from enum import IntFlag
from functools import reduce

import onnx
import torch
from torch.utils.cpp_extension import ROCM_HOME

import onnxruntime
from onnxruntime.capi import _pybind_state as C
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime.training import ortmodule

from . import _are_deterministic_algorithms_enabled, _io, _logger, _onnx_models, _runtime_inspector, _utils
from ._custom_autograd_function_exporter import _post_process_after_export
from ._fallback import (
    ORTModuleDeviceException,
    ORTModuleONNXModelException,
    ORTModuleTorchModelException,
    _FallbackManager,
    wrap_exception,
)
from ._gradient_accumulation_manager import GradientAccumulationManager
from ._graph_execution_interface import GraphExecutionInterface
from .debug_options import DebugOptions, LogLevel
from .torch_cpp_extensions.cpu.aten_op_executor import load_aten_op_executor_cpp_extension


class _RunStateInfo:
    def __init__(self, state, output_info):
        """
        :param state: State of partial run that contains intermediate tensors needed to resume the run later.
        :param output_info: Output info.
        """
        self.state = state
        self.output_info = output_info


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


class GraphExecutionManager(GraphExecutionInterface):
    def __init__(self, module, debug_options: DebugOptions, fallback_manager: _FallbackManager):
        """Manages construction and execution of ONNX graphs"""

        super().__init__(module._original_module)

        # IMPORTANT: Debug and Fallback must the configured first
        self._debug_options = debug_options
        self._fallback_manager = fallback_manager

        # Original and flattened (transformed) output module
        self._flattened_module = module

        # onnx models
        self._onnx_models = _onnx_models.ONNXModels()

        # Model after inference optimization or gradient building.
        self._graph_builder = None
        self._graph_info = None
        self._graph_initializer_names = None
        self._graph_initializer_names_to_train = None
        self._graph_initializers = None

        # Update constant ONNX_OPSET_VERSION with env var ORTMODULE_ONNX_OPSET_VERSION
        # if defined.
        ortmodule.ONNX_OPSET_VERSION = ortmodule._defined_from_envvar(
            "ORTMODULE_ONNX_OPSET_VERSION", ortmodule.ONNX_OPSET_VERSION, warn=True
        )

        # TrainingAgent or InferenceAgent
        self._execution_agent = None

        # indicators of some logic have been executed previously and thus could be skipped for faster training
        # default is enabled, if not defined in os env
        self._skip_check = _SkipCheck(
            _SkipCheck.SKIP_CHECK_DEVICE | _SkipCheck.SKIP_CHECK_BUILD_GRADIENT | _SkipCheck.SKIP_CHECK_EXECUTION_AGENT
        )
        if os.getenv("ORTMODULE_SKIPCHECK_POLICY") is not None:
            self._skip_check = reduce(
                lambda x, y: x | y,
                [_SkipCheck[name] for name in _utils.parse_os_env_skip_check_flags("ORTMODULE_SKIPCHECK_POLICY")],
            )
        self._first_skip_check_warning = True

        # Inspect embedding input index sparsity.
        self._rt_inspector = _runtime_inspector.RuntimeInspector()

        # Graph transformer config
        # Specify cast propagation strategy. Currently, three strategies are available, NONE, INSERT-AND-REDUCE and FLOOD-FILL
        # The default is FLOOD_FILL, expand FP16 computation regions in the graph using allowed opcodes for the given level.
        self._propagate_cast_ops_strategy = C.PropagateCastOpsStrategy.FLOOD_FILL
        # Optimize by moving Cast operations if propagate_cast_ops_level is non-negative.
        # - If the _propagate_cast_ops_level is set to zero, then the transformation considers only the opcodes specified by _propagate_cast_ops_allow
        #   as "FP16 safe", to insert/(re)move cast operations before/after to perform such operations in reduced (16-bit) precision.
        # - If propagate_cast_ops_level is positive, 1 or 2, then in addition to opcode codes specified by propagate_cast_ops_allow, use onnxruntime
        #   predetermined list of opcodes considered safe to move before/after the cast operation.
        # - Onnxruntime Level 1 predetermined "FP16 safe" opcodes include only opcodes that do not perform any computation such as Transpose, Split, Reshape, etc.,
        #   or the computation is actually in Float such as GeLU, etc.
        #   whereas Level 2 predetermined "FP16 safe" opcodes include opcodes that perform computation using contrib ops, Dropout, LayerNormalization, etc.
        self._propagate_cast_ops_level = 1
        # List of opcodes to be considered safe to move before/after the cast operation if propagate_cast_ops_level is zero.
        self._propagate_cast_ops_allow = []

        # Value can be either torch.onnx.TrainingMode.TRAINING or torch.onnx.TrainingMode.EVAL
        # To be instantiated in the concrete implementation of GraphExecutionManager
        self._export_mode = None

        # Exporter can take extra arguments for ORTModule extensions
        # It cannot overlap with required/immutable arguments (validated in runtime)
        self._export_extra_kwargs = {}

        # Related to training graph shape inference
        self._current_input_shape = None
        # default execution order is priority-based for both dynamic/static shape input for now
        # if we observe the benefit of static shape, we can expose this flag to the user
        self._use_static_shape = False

        # flag to enable symbolic shape inference for dynamic shape inputs to improve performance
        self._run_symbolic_shape_infer = True

        # PyTorch custom Autograd function support
        from ._custom_autograd_function import custom_autograd_function_enabler

        self._enable_custom_autograd_function = custom_autograd_function_enabler.state

        self._input_info = None
        self._module_output_schema = None
        self._device = _utils.get_device_from_module(module)

        self._module_parameters = list(inspect.signature(self._original_module.forward).parameters.values())

        # TODO: remove after PyTorch ONNX exporter supports VAR_KEYWORD parameters.
        for input_parameter in self._module_parameters:
            if input_parameter.kind == inspect.Parameter.VAR_KEYWORD:
                if self._debug_options.logging.log_level <= LogLevel.WARNING:
                    warnings.warn(
                        "The model's forward method has **kwargs parameter which has EXPERIMENTAL support!", UserWarning
                    )

        self.is_rocm_pytorch = bool(torch.version.hip is not None and ROCM_HOME is not None)

        self._use_external_gpu_allocator = True
        # assign self._torch_alloc and self._torch_free if self._use_external_gpu_allocator is True
        self._get_torch_gpu_allocator_function_addresses()

        # WIP feature to enable caching in Gradient accumulation scenario.
        self._enable_grad_acc_optimization = False
        self._gradient_accumulation_manager = GradientAccumulationManager()

        # Memory-aware gradient builder.
        self._use_memory_efficient_gradient = False

        # Enable compute optimizer by default. Allowed to be disabled via an environment variable for
        # convergence parity investigation.
        self._enable_compute_optimizer = (
            ortmodule._defined_from_envvar("ORTMODULE_ENABLE_COMPUTE_OPTIMIZER", 1, warn=True) == 1
        )
        self._enable_label_sparsity_optimization = (
            self._enable_compute_optimizer
            and ortmodule._defined_from_envvar("ORTMODULE_ENABLE_LABEL_SPARSITY_OPT", 0, warn=True) == 1
        )

        # Flag to re-export the model due to attribute change on the original module.
        # Re-export will be avoided if _skip_check is enabled.
        self._original_model_has_changed = False

        # Load ATen operator executor extension.
        load_aten_op_executor_cpp_extension()

        # Enable Triton op executor if Triton is installed, backend has support and environment variable is set.
        if ortmodule._defined_from_envvar("ORTMODULE_USE_TRITON", 0) != 0 and C.is_tritonop_on():
            try:
                import triton
            except ImportError:
                pass
            else:
                from onnxruntime.training.ortmodule.ort_triton import register_triton_op_executor

                register_triton_op_executor()

    def _get_torch_gpu_allocator_function_addresses(self):
        if self._use_external_gpu_allocator and torch.cuda.is_available():
            # CPP extension to get torch GPU allocator's alloc and free function addresses
            from onnxruntime.training.ortmodule.torch_cpp_extensions import torch_gpu_allocator

            self._torch_alloc = torch_gpu_allocator.gpu_caching_allocator_raw_alloc_address()
            self._torch_free = torch_gpu_allocator.gpu_caching_allocator_raw_delete_address()
            self._torch_empty_cache = torch_gpu_allocator.gpu_caching_allocator_empty_cache_address()

    def _validate_module_type(self, module):
        """Raises ORTModuleTorchModelException if the module is not a torch.nn.Module"""

        if not isinstance(module, torch.nn.Module):
            raise wrap_exception(
                ORTModuleTorchModelException,
                TypeError(f"ORTModule only supports torch.nn.Module as input. {type(module)} is not supported."),
            )

        # Hard-coded list of unsupported torch.nn.Module goes here for fallback
        if isinstance(module, torch.nn.DataParallel):
            raise wrap_exception(
                ORTModuleTorchModelException,
                TypeError(
                    "ORTModule is not compatible with torch.nn.DataParallel. "
                    "Please use torch.nn.parallel.DistributedDataParallel instead."
                ),
            )

    @staticmethod
    def execution_session_run_forward(execution_session, onnx_model, device, *inputs):
        """Runs the forward pass on `execution_session` with given `onnx_model`, `device` and `inputs`

        This is a helper that can be called by the actual `GraphExecutionManager.forward` method

        Args:
            execution_session (InferenceAgent or InferenceAgent): Agent which runs either inference or train
            onnx_model (onnx.ModelProto): ONNX model
            device (torch.device): PyTorch device
            inputs: (torch.Tensor or a container of): User input

        Returns:
            Returns a tuple (user_outputs, run_info):
            user_outputs: The model output (either torch.Tensor or a container of torch.Tensor)
            run_info: A _RunStateInfo which contains extra information about the execution of the graph
        """

        raise NotImplementedError

    @abstractmethod
    def forward(self):
        """Executes the forward method for ORTModule

        This is an abstract method and must be overridden by a concrete implementation.
        This is the only method that the user should call on a concrete instance of the ExecutionManager
        All other methods are internal"""
        pass

    def _build_graph(self):
        if self._use_static_shape:
            self._graph_builder.build(self._input_info.shape)
        else:
            self._graph_builder.build()

        self._graph_info = self._graph_builder.get_graph_info()

    def _get_session_config(self):
        """Creates and returns the session configuration to be used for the ExecutionAgent"""

        if _are_deterministic_algorithms_enabled():
            if self._debug_options.logging.log_level <= _logger.LogLevel.INFO:
                warnings.warn(
                    "ORTModule's determinism will be enabled because PyTorch's determinism is enabled.", UserWarning
                )

        providers = None
        provider_options = None
        if self._device.type == "cuda":
            # Configure the InferenceSessions to use the specific GPU on which the model is placed.
            providers = ["ROCMExecutionProvider"] if self.is_rocm_pytorch else ["CUDAExecutionProvider"]
            providers.append("CPUExecutionProvider")
            provider_option_map = {"device_id": str(self._device.index)}
            if not self.is_rocm_pytorch:
                # Set Conv algo search mode to HEURISTIC by default, which is the same as PyTorch's default setting.
                conv_algo_search = ortmodule._defined_from_envvar("ORTMODULE_CONV_ALGO_SEARCH", "HEURISTIC", warn=True)
                if conv_algo_search not in ["HEURISTIC", "EXHAUSTIVE"]:
                    warnings.warn("Invalid value of env CONV_ALGO_SEARCH. Must be HEURISTIC or EXHAUSTIVE.")
                    conv_algo_search = "HEURISTIC"
                provider_option_map["cudnn_conv_algo_search"] = conv_algo_search
                provider_option_map["cudnn_conv_use_max_workspace"] = "1"
                provider_option_map["cudnn_conv1d_pad_to_nc1d"] = "1"
            if self._use_external_gpu_allocator:
                provider_option_map["gpu_external_alloc"] = str(self._torch_alloc)
                provider_option_map["gpu_external_free"] = str(self._torch_free)
                provider_option_map["gpu_external_empty_cache"] = str(self._torch_empty_cache)
            provider_options = [provider_option_map, {}]
        elif self._device.type == "cpu":
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]
        elif self._device.type == "ort":
            provider_info = C.get_ort_device_provider_info(self._device.index)
            assert len(provider_info.keys()) == 1
            providers = list(provider_info.keys())
            provider_options = [provider_info[providers[0]]]

        session_options = onnxruntime.SessionOptions()
        session_options.enable_mem_pattern = False
        session_options.enable_mem_reuse = False
        session_options.use_deterministic_compute = _are_deterministic_algorithms_enabled()
        # default to PRIORITY_BASED execution order
        session_options.execution_order = onnxruntime.ExecutionOrder.PRIORITY_BASED
        # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.
        session_options.log_severity_level = int(self._debug_options.logging.log_level)
        # Disable memory alleviation by default. Allow user to enable it via environment variable.
        alleviation_config = ortmodule._defined_from_envvar("ORTMODULE_MEMORY_OPT_CONFIG", "", warn=True)
        probe_level = ortmodule._defined_from_envvar("ORTMODULE_MEMORY_OPT_PROBE_RECOMPUTE_LEVEL", "1", warn=True)
        session_options.add_session_config_entry("optimization.enable_memory_optimizer", alleviation_config)
        session_options.add_session_config_entry("optimization.enable_memory_probe_recompute_level", probe_level)
        # Disable weight prepacking
        session_options.add_session_config_entry("session.disable_prepacking", "1")

        if self._debug_options.save_onnx_models.save:
            session_options.optimized_model_filepath = os.path.join(
                self._debug_options.save_onnx_models.path,
                _onnx_models._get_onnx_file_name(
                    self._debug_options.save_onnx_models.name_prefix, "execution_model", self._export_mode
                ),
            )

        return session_options, providers, provider_options

    def _export_model(self, *inputs, **kwargs):
        # 1. Set the self._device from the user module
        # 2. Verify input schema matches the schema used on the previous model export
        # 3. Export the user model under self._export_training_flag mode
        # Return True if the model needs to be exported, False if no export is required.

        # Note: Model is only exported when:
        #       1. Model has never been exported before.
        #       2. Model input schema has changed (changes in inputs requiring gradient, shape, boolean inputs values change, etc)
        #       Model is not re-exported when the model parameters change. This can happen when the model is stateful,
        #       or the user explicitly changed model parameters after the onnx export.

        # Record random states here and restore later in case any of them gets changed during the export,
        # e.g., some sympy functions in symbolic_shape_infer will change Python's random state.
        random_states = _utils.get_random_states()

        schema = _io._extract_schema({"args": copy.copy(inputs), "kwargs": copy.copy(kwargs)})
        if (
            self._onnx_models.exported_model
            and schema == self._input_info.schema
            and not self._original_model_has_changed
        ):
            # All required models have already been exported previously
            return False

        self._set_device_from_module(inputs, kwargs)
        self._onnx_models.exported_model = self._get_exported_model(schema, *inputs, **kwargs)
        if self._debug_options.save_onnx_models.save:
            self._onnx_models.save_exported_model(
                self._debug_options.save_onnx_models.path,
                self._debug_options.save_onnx_models.name_prefix,
                self._export_mode,
            )

        if self._run_symbolic_shape_infer:
            self._onnx_models.exported_model = SymbolicShapeInference.infer_shapes(
                self._onnx_models.exported_model, auto_merge=True, guess_output_rank=True
            )

        # Restore the recorded random states
        _utils.set_random_states(random_states)

        return True

    def _get_exported_model(self, input_schema, *inputs, **kwargs):
        """Exports PyTorch `self._flattened_module` to ONNX for inferencing or training, using `*inputs` and `**kwargs` as input

        TODO: How to support dynamic axes? Dimensions are determined by samples
        """

        # Setup dynamic axes for onnx model
        self._input_info = _io.parse_inputs_for_onnx_export(self._module_parameters, None, input_schema, inputs, kwargs)
        (
            output_names,
            output_dynamic_axes,
            self._module_output_schema,
        ) = _io.parse_outputs_for_onnx_export_and_extract_schema(
            self._original_module, inputs, kwargs, self._debug_options.logging.log_level
        )
        self._input_info.dynamic_axes.update(output_dynamic_axes)

        # FlattenedModule needs _InputInfo to expand user input from *args to *args + **kwargs
        self._flattened_module._input_info = self._input_info

        # Export torch.nn.Module to ONNX
        f = io.BytesIO()

        # Deepcopy inputs, since input values may change after model run.
        # NOTE: Inputs may contain tensors that have attributes preventing their deepcopy (example grad_fn).
        # Therefore, deepcopy only the data component of the input tensors for export.
        sample_inputs_copy, sample_kwargs_copy = _io.deepcopy_model_input(*inputs, **kwargs)
        # NOTE: Flattening the input will change the 'input schema', resulting in a re-export
        sample_inputs_as_tuple = tuple(self._input_info.flatten(sample_inputs_copy, sample_kwargs_copy, self._device))
        # Ops behaving differently under train/eval mode need to be exported with the
        # correct training flag to reflect the expected behavior.
        # For example, the Dropout node in a model is dropped under eval mode.
        assert self._export_mode is not None, "Please use a concrete instance of ExecutionManager"

        try:
            with torch.no_grad(), _logger.suppress_os_stream_output(log_level=self._debug_options.logging.log_level):
                required_export_kwargs = {
                    "input_names": self._input_info.names,
                    "output_names": output_names,
                    "opset_version": ortmodule.ONNX_OPSET_VERSION,
                    "do_constant_folding": False,
                    "training": self._export_mode,
                    "dynamic_axes": self._input_info.dynamic_axes,
                    "verbose": self._debug_options.logging.log_level < LogLevel.WARNING,
                    "export_params": False,
                    "keep_initializers_as_inputs": True,
                }
                invalid_args = self._export_extra_kwargs.keys() & required_export_kwargs.keys()
                assert (
                    len(invalid_args) == 0
                ), f"The following PyTorch exporter arguments cannot be specified: '{invalid_args}'."
                torch.onnx.export(
                    self._flattened_module,
                    sample_inputs_as_tuple,
                    f,
                    **required_export_kwargs,
                    **self._export_extra_kwargs,
                )
        except Exception as e:
            raise wrap_exception(  # noqa: B904
                ORTModuleONNXModelException,
                RuntimeError(
                    f"There was an error while exporting the PyTorch model to ONNX: "
                    f"\n\n{_utils.get_exception_as_string(e)}"
                ),
            )
        exported_model = onnx.load_model_from_string(f.getvalue())

        exported_model = _post_process_after_export(
            exported_model, self._enable_custom_autograd_function, self._debug_options.logging.log_level
        )

        return exported_model

    def _set_device_from_module(self, inputs, kwargs):
        """Get the device from the module and save it to self._device"""

        device = _utils.get_device_from_module(self._original_module) or _utils.get_device_from_inputs(inputs, kwargs)
        if not self._device or self._device != device:
            self._device = device
            if not self._device:
                raise wrap_exception(
                    ORTModuleDeviceException, RuntimeError("A device must be specified in the model or inputs!")
                )

    def _get_graph_transformer_config(self):
        graph_transformer_config = C.TrainingGraphTransformerConfiguration()
        graph_transformer_config.propagate_cast_ops_config = C.PropagateCastOpsConfiguration()
        graph_transformer_config.propagate_cast_ops_config.level = self._propagate_cast_ops_level
        graph_transformer_config.propagate_cast_ops_config.allow = self._propagate_cast_ops_allow
        graph_transformer_config.propagate_cast_ops_config.strategy = self._propagate_cast_ops_strategy
        graph_transformer_config.enable_compute_optimizer = self._enable_compute_optimizer
        graph_transformer_config.enable_label_sparsity_optimization = self._enable_label_sparsity_optimization
        return graph_transformer_config

    def _initialize_graph_builder(self):
        """Creates a new OrtModuleGraphBuilder, initializes it and saves it to self._graph_builder"""

        # All initializer names along with user inputs are a part of the onnx graph inputs
        # since the onnx model was exported with the flag keep_initializers_as_inputs=True
        onnx_initializer_names = {p.name for p in self._onnx_models.exported_model.graph.input}

        # TODO: PyTorch exporter bug: changes the initializer order in ONNX model
        initializer_names = [
            name for name, _ in self._flattened_module.named_parameters() if name in onnx_initializer_names
        ]
        initializer_names_to_train = [
            name
            for name, param in self._flattened_module.named_parameters()
            if param.requires_grad and name in onnx_initializer_names
        ]

        # Build and optimize the full graph
        grad_builder_config = C.OrtModuleGraphBuilderConfiguration()
        grad_builder_config.initializer_names = initializer_names
        grad_builder_config.initializer_names_to_train = initializer_names_to_train
        grad_builder_config.input_names_require_grad = self._input_info.require_grad_names
        grad_builder_config.build_gradient_graph = self._export_mode == torch.onnx.TrainingMode.TRAINING
        grad_builder_config.graph_transformer_config = self._get_graph_transformer_config()
        grad_builder_config.enable_caching = self._enable_grad_acc_optimization
        grad_builder_config.loglevel = _logger.ortmodule_loglevel_to_onnxruntime_c_loglevel(
            self._debug_options.logging.log_level
        )
        grad_builder_config.use_memory_efficient_gradient = self._use_memory_efficient_gradient
        self._graph_builder = C.OrtModuleGraphBuilder()

        # It is assumed here that the order and names of the inputs and outputs are not modified by the backend in any way
        # and are kept as they appear in the exported onnx model.
        self._graph_builder.initialize(self._onnx_models.exported_model.SerializeToString(), grad_builder_config)

        # TODO: Explore ways to make self._graph_info.initializer_names and self._graph_info.initializer_names_to_train
        #       a set (unordered_set in the backend) that does not require a copy on each reference.
        self._graph_initializer_names = set(initializer_names)
        self._graph_initializer_names_to_train = set(initializer_names_to_train)

        # Initializers can be cached and used since they are expected not to be re-instantiated
        # between forward calls.
        self._graph_initializers = [
            param for name, param in self._flattened_module.named_parameters() if name in self._graph_initializer_names
        ]

    def signal_model_changed(self):
        """Signals the execution manager to re-export the model on the next forward call"""
        self._original_model_has_changed = True

    def __getstate__(self):
        state = copy.copy(self.__dict__)
        # Remove any re-contructible/pybound object from the state
        serialization_deny_list = [
            "_onnx_models",
            "_graph_builder",
            "_graph_info",
            "_execution_agent",
            "_torch_alloc",
            "_torch_free",
            "_torch_empty_cache",
        ]
        for attribute_name in serialization_deny_list:
            del state[attribute_name]

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        _utils.reinitialize_graph_execution_manager(self)
