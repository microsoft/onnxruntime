# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import copy
import inspect
import io
import logging
import os
from abc import ABC, abstractmethod  # noqa: F401
from hashlib import md5 as hash_fn
from typing import Dict, List, Optional, Tuple

import onnx
import torch
from torch.utils.cpp_extension import ROCM_HOME

import onnxruntime
from onnxruntime.capi import _pybind_state as C
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime.training.utils import ORTModelInputOutputSchemaType, PTable, onnx_dtype_to_pytorch_dtype

from . import _are_deterministic_algorithms_enabled, _io, _logger, _onnx_models, _utils, export_context
from ._fallback import (
    ORTModuleDeviceException,
    ORTModuleONNXModelException,
    ORTModuleTorchModelException,
    _FallbackManager,
    _FallbackPolicy,
    wrap_exception,
)
from ._gradient_accumulation_manager import GradientAccumulationManager
from ._graph_execution_interface import GraphExecutionInterface
from ._io import _FlattenedModule, _InputInfo
from ._logger import LogColor
from ._runtime_inspector import FlagPaddingElimination, RuntimeInspector
from ._utils import check_function_has_param, get_rank
from .options import DebugOptions, LogLevel, _MemoryOptimizationLevel, _RuntimeOptions
from .torch_cpp_extensions.cpu.aten_op_executor import load_aten_op_executor_cpp_extension


class _RunStateInfo:
    def __init__(self, state, output_info: List[Tuple[torch.Size, torch.device, torch.dtype]]):
        """
        :param state: State of partial run that contains intermediate tensors needed to resume the run later.
        :param output_info: Output info.
        """
        self.state = state
        self.output_info = output_info


class GraphExecutionManager(GraphExecutionInterface):
    def __init__(
        self,
        module: _FlattenedModule,
        debug_options: DebugOptions,
        export_mode: int,
        fallback_manager: _FallbackManager,
        logger: logging.Logger,
    ):
        """Manages construction and execution of ONNX graphs.

        Args:
            module: The flatten PyTorch module to be executed.
            debug_options: Debug options for ORTModule.
            export_mode: export mode, should be torch.onnx.TrainingMode.TRAINING or torch.onnx.TrainingMode.EVAL.
            fallback_manager: Fallback manager to handle exceptions.
            logger: Logger for ORTModule.

        """

        super().__init__(module._original_module)

        # IMPORTANT: Debug and Fallback must the configured first
        self._debug_options = debug_options
        self._fallback_manager = fallback_manager

        self._logger = logger

        # Management for ORTModule configuration.
        self._runtime_options = _RuntimeOptions(self._logger)

        # Original and flattened (transformed) output module
        self._flattened_module = module

        # onnx models
        self._onnx_models = _onnx_models.ONNXModels()

        # Model after inference optimization or gradient building.
        self._graph_builder = None
        self._graph_info = None
        self._graph_initializer_names = set()
        self._graph_initializer_names_to_train = set()
        self._graph_initializers: List[torch.nn.parameter.Parameter] = []

        # TrainingAgent or InferenceAgent
        self._execution_agent = None

        self._first_skip_check_warning = True

        # Tracker for ORTModule model export, session creation overhead.
        self.time_tracker = _logger.TimeTracker()

        # Value can be either torch.onnx.TrainingMode.TRAINING or torch.onnx.TrainingMode.EVAL
        # To be instantiated in the concrete implementation of GraphExecutionManager
        self._export_mode = export_mode

        # Exporter can take extra arguments for ORTModule extensions
        # It cannot overlap with required/immutable arguments (validated in runtime)
        self._export_extra_kwargs = {}

        # Input and output infos (including schema) for exported model.
        self._input_info: Optional[_InputInfo] = None
        self._module_output_schema: Optional[ORTModelInputOutputSchemaType] = None

        # Device where the model is placed.
        self._device: Optional[torch.device] = _utils.get_device_from_module(module)

        # Forward function input parameters of the original module.
        self._module_parameters: List[inspect.Parameter] = list(
            inspect.signature(self._original_module.forward).parameters.values()
        )

        # TODO: remove after PyTorch ONNX exporter supports VAR_KEYWORD parameters.
        for input_parameter in self._module_parameters:
            if input_parameter.kind == inspect.Parameter.VAR_KEYWORD:
                self._logger.info("The model's forward method has **kwargs parameter which has EXPERIMENTAL support!")

        self.is_rocm_pytorch = bool(torch.version.hip is not None and ROCM_HOME is not None)

        # WIP feature to enable caching in Gradient accumulation scenario.
        self._gradient_accumulation_manager = GradientAccumulationManager()

        # Flag to re-export the model due to attribute change on the original module.
        # Re-export will be avoided if _skip_check is enabled.
        self._original_model_has_changed = False

        # Inspector for runtime information, for example input data, memory usage, etc.
        self._runtime_inspector = RuntimeInspector(
            self._logger, self._original_module, self._export_mode == torch.onnx.TrainingMode.TRAINING
        )
        self._runtime_inspector.memory_ob.enable_memory_stats_by_step(self._runtime_options.print_memory_stat_by_step)

        # Load ATen operator executor extension.
        load_aten_op_executor_cpp_extension()

        # Assign self._torch_alloc and self._torch_free if self._use_external_gpu_allocator is True
        self._get_torch_gpu_allocator_function_addresses()

        if self._runtime_options.enable_triton:
            from onnxruntime.training.ort_triton import register_triton_op_executor

            register_triton_op_executor()

        self._zero_stage3_param_map = {}
        if self._runtime_options.enable_zero_stage3_support:
            # Move import to here to avoid circular dependency error
            from onnxruntime.training.utils.hooks import configure_ort_compatible_zero_stage3  # type: ignore[import]

            # Cannot toggle feature enabling/disabling after the first time enabled.

            configure_ort_compatible_zero_stage3(debug=False, stats_output_dir="ort_output", stats_overwrite=True)

        # Will be reset everytime we re-initialize the graph builder.
        # Be noted, we will never enable this feature for inference mode.
        self._mem_efficient_grad_management_is_enabled = False

    def _get_torch_gpu_allocator_function_addresses(self):
        if self._runtime_options.use_external_gpu_allocator and torch.cuda.is_available():
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

    @abstractmethod
    def forward(self):
        """Executes the forward method for ORTModule

        This is an abstract method and must be overridden by a concrete implementation.
        This is the only method that the user should call on a concrete instance of the ExecutionManager
        All other methods are internal"""

    def _build_graph(self, config):
        if self._runtime_options.use_static_shape:
            self._graph_builder.build(config, self._input_info.shape)
        else:
            self._graph_builder.build(config)

        self._graph_info = self._graph_builder.get_graph_info()

    def _get_session_config(self):
        """Creates and returns the session configuration to be used for the ExecutionAgent"""

        if _are_deterministic_algorithms_enabled():
            self._logger.info("ORTModule's determinism will be enabled because PyTorch's determinism is enabled.")

        providers = None
        provider_options = None
        if self._device.type == "cuda":
            # Configure the InferenceSessions to use the specific GPU on which the model is placed.
            providers = ["ROCMExecutionProvider"] if self.is_rocm_pytorch else ["CUDAExecutionProvider"]
            providers.append("CPUExecutionProvider")
            provider_option_map = {"device_id": str(self._device.index)}
            if not self.is_rocm_pytorch:
                # Set Conv algo search mode to HEURISTIC by default, which is the same as PyTorch's default setting.
                provider_option_map["cudnn_conv_algo_search"] = self._runtime_options.conv_algo_search
                provider_option_map["cudnn_conv_use_max_workspace"] = "1"
                provider_option_map["cudnn_conv1d_pad_to_nc1d"] = "1"
                if self._runtime_options.enable_tuning:
                    provider_option_map["tunable_op_enable"] = "1"
                    provider_option_map["tunable_op_tuning_enable"] = "1"
                    if self._runtime_options.max_tuning_duration_ms:
                        provider_option_map["tunable_op_max_tuning_duration_ms"] = str(
                            self._runtime_options.max_tuning_duration_ms
                        )
                elif self._runtime_options.tuning_results_path:
                    provider_option_map["tunable_op_enable"] = "1"
            if self._runtime_options.use_external_gpu_allocator:
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
        # DEFAULT order is reversed DFS order, while PRIORITY_BASED order is forward BFS order.
        # DEFAULT order is likely to be better than PRIORITY_BASED order on memory. However, our recompute feature
        # requires PRIORITY_BASED order to work properly. So we use PRIORITY_BASED order when recompute is enabled.
        session_options.execution_order = (
            onnxruntime.ExecutionOrder.PRIORITY_BASED
            if self._runtime_options.memory_optimizer_is_enabled()
            else onnxruntime.ExecutionOrder.DEFAULT
        )
        # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.
        session_options.log_severity_level = int(self._debug_options.logging.log_level)

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

    @_logger.TrackTime(_logger.ORTModuleInitPhase.EXPORT)
    @_logger.SuppressLogs(_logger.ORTModuleInitPhase.EXPORT, is_ort_filter=False)
    def _export_model(self, *inputs, **kwargs) -> bool:
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

        schema = _io._extract_schema({"args": copy.copy(inputs), "kwargs": copy.copy(kwargs)}, self._device)
        if (
            self._onnx_models.exported_model
            and schema == self._input_info.schema
            and not self._original_model_has_changed
        ):
            # All required models have already been exported previously
            return False
        self._set_device_from_module(inputs, kwargs)
        # TODO: move it into runtime_inspector
        embedding_hook_handles = self._add_check_embedding_sparsity_hook()

        from onnxruntime.training.utils.hooks._subscriber_manager import no_increase_global_step

        with export_context(), no_increase_global_step():
            self._onnx_models.exported_model = self._get_exported_model(schema, *inputs, **kwargs)

        for hook in embedding_hook_handles:
            hook.remove()
        if self._debug_options.save_onnx_models.save:
            self._onnx_models.save_exported_model(
                self._debug_options.save_onnx_models.path,
                self._debug_options.save_onnx_models.name_prefix,
                self._export_mode,
            )

        if self._runtime_options.run_symbolic_shape_infer:
            self._onnx_models.exported_model = SymbolicShapeInference.infer_shapes(
                self._onnx_models.exported_model, auto_merge=True, guess_output_rank=True
            )

        # Restore the recorded random states
        _utils.set_random_states(random_states)

        return True

    def _get_exported_model(self, input_schema: ORTModelInputOutputSchemaType, *inputs, **kwargs) -> onnx.ModelProto:
        """Exports PyTorch `self._flattened_module` to ONNX for inferencing or training,
          using `*inputs` and `**kwargs` as input

        TODO: How to support dynamic axes? Dimensions are determined by samples
        """

        # VERBOSE -> FULL export verbose log + FULL torch other logs from stdout and stderr (C++ backend)
        # DEVINFO -> FULL export verbose log + FULL torch other logs from stdout and stderr (C++ backend)
        # INFO -> [Rank 0] FULL export verbose log + FILTERED torch other logs from stdout and stderr (C++ backend)
        # WARNING/ERROR -> [Rank 0] NO export verbose log + FILTERED torch other logs from stdout and stderr (C++ backend)
        # Be noted: rank 0 log only is controlled by logger configured in _logger.py
        torch_exporter_verbose_log = self._debug_options.logging.log_level <= LogLevel.INFO

        # Setup dynamic axes for onnx model
        self._input_info = _io.parse_inputs_for_onnx_export(self._module_parameters, None, input_schema, inputs, kwargs)
        need_deep_copy = self._runtime_options.deepcopy_before_model_export and _io.can_module_be_deep_cloned(
            self._original_module, self._device
        )
        if not need_deep_copy:
            if self._runtime_options.deepcopy_before_model_export:
                self._logger.warning(
                    "Since the user requested not to deep copy this model, "
                    "the initial weights may not be preserved and could change slightly during the forward run. "
                    "This could cause a minor difference between the ORTModule and the PyTorch run for the "
                    "first iteration. The computation will proceed as normal, but this should be noted."
                )
            else:
                self._logger.warning(
                    "Due to the limited GPU memory execution manager does not create a deep copy of this model. "
                    "Therefore, the initial weights might be slightly altered during the forward run. "
                    "This could result in a minor discrepancy between the ORTModule and the PyTorch run for the "
                    "first iteration. The computation will continue as usual, but this should be noted."
                )
        (
            output_names,
            output_dynamic_axes,
            self._module_output_schema,
        ) = _io.parse_outputs_for_onnx_export_and_extract_schema(
            self._original_module, inputs, kwargs, self._logger, self._device, need_deep_copy
        )
        self._input_info.dynamic_axes.update(output_dynamic_axes)

        # FlattenedModule needs _InputInfo to expand user input from *args to *args + **kwargs
        self._flattened_module._input_info = self._input_info

        self._logger.info("Exporting the PyTorch model to ONNX...")

        # Leverage cached model if available
        cache_dir = self._runtime_options.ortmodule_cache_dir
        if cache_dir:
            filename = os.path.join(
                cache_dir, f"{hash_fn(str(self._flattened_module).encode()).hexdigest()}_{get_rank()}.onnx"
            )
            if os.path.exists(cache_dir) and os.path.isfile(filename):
                self._logger.warning(
                    f"Cached model detected! Cached model will be used to save export and initialization time."
                    f"If you want the model to be re-exported then DELETE {filename}."
                )
                exported_model = onnx.load(filename)
                return exported_model

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
            from ._zero_stage3_compatibility import stage3_export_context

            with torch.no_grad(), stage3_export_context(self._runtime_options.enable_zero_stage3_support, self):
                required_export_kwargs = {
                    "input_names": self._input_info.names,
                    "output_names": output_names,
                    "opset_version": self._runtime_options.onnx_opset_version,
                    "do_constant_folding": False,
                    "training": self._export_mode,
                    "dynamic_axes": self._input_info.dynamic_axes,
                    "verbose": torch_exporter_verbose_log,
                    "export_params": False,
                    "keep_initializers_as_inputs": True,
                }

                if check_function_has_param(torch.onnx.export, "autograd_inlining"):
                    # From some PyTorch version, autograd_inlining is a valid argument.
                    # We allow it to be True if custom autograd function is disabled (where autograd.Function
                    # anyway is not supported in ONNX until it can be inlined).
                    required_export_kwargs["autograd_inlining"] = (
                        not self._runtime_options.enable_custom_autograd_function
                    )

                invalid_args = self._export_extra_kwargs.keys() & required_export_kwargs.keys()

                if len(invalid_args) != 0:
                    error_msg = f"The following PyTorch exporter arguments cannot be specified: '{invalid_args}'."
                    raise RuntimeError(error_msg)

                torch.onnx.export(
                    self._flattened_module,
                    sample_inputs_as_tuple,
                    f,
                    **required_export_kwargs,
                    **self._export_extra_kwargs,
                )
        except Exception as e:
            message = _utils.get_exception_as_string(e)

            # Special handling when Huggingface transformers gradient checkpoint usage pattern found.
            # For new versions of PyTorch 2, tracing torch.utils.checkpoint.checkpoint will be failed like this:
            #   File "microsoft/phi-2/b10c3eba545ad279e7208ee3a5d644566f001670/modeling_phi.py", line 919, in forward
            #     layer_outputs = self._gradient_checkpointing_func(
            #   File "/site-packages/torch/_compile.py", line 24, in inner
            #     return torch._dynamo.disable(fn, recursive)(*args, **kwargs)
            #   File "/site-packages/torch/_dynamo/eval_frame.py", line 470, in _fn
            #     raise RuntimeError(
            #   RuntimeError: Detected that you are using FX to torch.jit.trace a dynamo-optimized function. This is not supported at the moment.
            if (
                "_gradient_checkpointing_func" in message
                and "Detected that you are using FX to torch.jit.trace a dynamo-optimized function" in message
            ):
                is_ckpt_activation_allowed = int(os.getenv("ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT", "0")) == 1
                notes = (
                    " Your model is running with gradient checkpointing, yet the PyTorch exporter\n"
                    " failed during tracing the graph. Try to enable ORTModule's\n"
                    " gradient checkpointing (a.k.a. Transformer layerwise subgraph recompute)\n"
                    " using `export ORTMODULE_MEMORY_OPT_LEVEL=1` for similar or even better memory efficiency.\n"
                )
                if is_ckpt_activation_allowed:
                    # If the user allows the gradient checkpointing export, we should inform the user to disable it,
                    # to make layerwise recompute work.
                    notes += (
                        " We also notice your setting `export ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT=1`,\n"
                        " which enables gradient checkpointing torch.autograd.Functions(s) to export.\n"
                        " To enable ORTModule's layerwise recompute, it needs to be turned OFF by\n"
                        " `export ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT=0`.\n"
                    )

                self._logger.error(
                    f"{LogColor.RED}\n"
                    "******************************** IMPORTANT NOTE *******************************\n"
                    f"{notes}"
                    "*******************************************************************************\n"
                    f"{LogColor.ENDC}\n"
                )

            raise wrap_exception(  # noqa: B904
                ORTModuleONNXModelException,
                RuntimeError(f"There was an error while exporting the PyTorch model to ONNX: \n\n{message}"),
            )
        exported_model = onnx.load_model_from_string(f.getvalue())

        if self._runtime_options.enable_custom_autograd_function:
            from ._custom_autograd_function_exporter import post_process_enabling_autograd_function

            exported_model = post_process_enabling_autograd_function(exported_model)

        if self._runtime_options.enable_zero_stage3_support:
            from ._zero_stage3_compatibility import post_processing_enable_zero_stage3_compat

            exported_model = post_processing_enable_zero_stage3_compat(
                exported_model,
                self._zero_stage3_param_map,
                [name for name, _ in self._flattened_module.named_parameters()],
            )

            # Cannot append pull weight trigger name to input names as following, otherwise, the later check (
            # https://github.com/microsoft/onnxruntime/blob/068300d97eb25e5b52324e7af54a45ed1fa6a4c3/orttraining/orttraining/python/training/ortmodule/_training_manager.py#L466C18-L466C18)
            # find input info mismatch, will re-initialize the graph builder.
            # self._input_info.require_grad_names.append(STAGE3_PULL_WEIGHT_TRIGGER_NAME)

        # Cache model for future runs
        if cache_dir:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            filename = os.path.join(
                cache_dir, f"{hash_fn(str(self._flattened_module).encode()).hexdigest()}_{get_rank()}.onnx"
            )
            self._logger.info(f"Caching model for future runs to {filename}.")
            onnx.save(exported_model, filename)

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

    def _get_graph_transformer_config(self) -> C.TrainingGraphTransformerConfiguration:
        graph_transformer_config = C.TrainingGraphTransformerConfiguration()
        graph_transformer_config.propagate_cast_ops_config = C.PropagateCastOpsConfiguration()
        graph_transformer_config.propagate_cast_ops_config.level = self._runtime_options.propagate_cast_ops_level
        graph_transformer_config.propagate_cast_ops_config.allow = self._runtime_options.propagate_cast_ops_allow
        graph_transformer_config.propagate_cast_ops_config.strategy = self._runtime_options.propagate_cast_ops_strategy
        graph_transformer_config.enable_compute_optimizer = self._runtime_options.enable_compute_optimizer

        if self._debug_options.save_onnx_models.save:
            graph_transformer_config.optimized_pre_grad_filepath = os.path.join(
                self._debug_options.save_onnx_models.path,
                _onnx_models._get_onnx_file_name(
                    self._debug_options.save_onnx_models.name_prefix, "optimized_pre_grad", self._export_mode
                ),
            )

        return graph_transformer_config

    @_logger.TrackTime(_logger.ORTModuleInitPhase.GRAPH_BUILDER_INIT)
    def _initialize_graph_builder(self):
        """Creates a new OrtModuleGraphBuilder, initializes it and saves it to self._graph_builder"""

        self._mem_efficient_grad_management_is_enabled = (
            self._export_mode != torch.onnx.TrainingMode.EVAL
            and self._runtime_options.enable_mem_efficient_grad_management
        )

        # We post process the exported model because the trainable parame might be changed, so this path is
        # re-triggered by reinitialize_graph_builder.
        exported_model = copy.deepcopy(self._onnx_models.exported_model)
        self._onnx_models.processed_exported_model = exported_model

        if self._mem_efficient_grad_management_is_enabled:
            from ._mem_efficient_grad_mgmt import post_processing_enable_mem_efficient_training

            # Override the options if model is not modified.
            (
                self._mem_efficient_grad_management_is_enabled,
                exported_model,
            ) = post_processing_enable_mem_efficient_training(exported_model, self._flattened_module.named_parameters())

            if self._runtime_options.run_symbolic_shape_infer:
                exported_model = SymbolicShapeInference.infer_shapes(
                    exported_model, auto_merge=True, guess_output_rank=True
                )

        # All initializer names along with user inputs are a part of the onnx graph inputs
        # since the onnx model was exported with the flag keep_initializers_as_inputs=True
        # We need to use the raw exported model here since the graph inputs include both user inputrs and
        # parameters.
        onnx_initializer_names = {p.name for p in exported_model.graph.input}

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

        input_names_require_grad = self._input_info.require_grad_names
        if self._runtime_options.enable_zero_stage3_support:
            from ._zero_stage3_compatibility import STAGE3_PULL_WEIGHT_TRIGGER_NAME

            # Add stage3 pull weight trigger name to require_grad_names, so that it will be included in the gradient graph.
            input_names_require_grad.append(STAGE3_PULL_WEIGHT_TRIGGER_NAME)

        if self._mem_efficient_grad_management_is_enabled:
            from ._mem_efficient_grad_mgmt import MEM_EFFICIENT_PARAM_TRIGGER_INPUT_NAME

            # Add mem efficient grad trigger name to require_grad_names, so that it will be included in the gradient graph.
            input_names_require_grad.append(MEM_EFFICIENT_PARAM_TRIGGER_INPUT_NAME)

        grad_builder_config.input_names_require_grad = input_names_require_grad
        grad_builder_config.build_gradient_graph = self._export_mode == torch.onnx.TrainingMode.TRAINING
        grad_builder_config.enable_caching = self._runtime_options.enable_grad_acc_optimization
        grad_builder_config.loglevel = _logger.ortmodule_loglevel_to_onnxruntime_c_loglevel(
            self._debug_options.logging.log_level
        )
        grad_builder_config.use_memory_efficient_gradient = self._runtime_options.use_memory_efficient_gradient
        self._graph_builder = C.OrtModuleGraphBuilder()

        # It is assumed here that the order and names of the inputs and outputs are not modified by the backend in any way
        # and are kept as they appear in the exported onnx model.
        self._graph_builder.initialize(exported_model.SerializeToString(), grad_builder_config)

        raw_onnx_initializer_names = {p.name for p in self._onnx_models.exported_model.graph.input}

        raw_initializer_names = [
            name for name, _ in self._flattened_module.named_parameters() if name in raw_onnx_initializer_names
        ]
        raw_initializer_names_to_train = [
            name
            for name, param in self._flattened_module.named_parameters()
            if param.requires_grad and name in raw_onnx_initializer_names
        ]

        # TODO: Explore ways to make self._graph_info.initializer_names and self._graph_info.initializer_names_to_train
        #       a set (unordered_set in the backend) that does not require a copy on each reference.
        self._graph_initializer_names = set(raw_initializer_names)
        self._graph_initializer_names_to_train = set(raw_initializer_names_to_train)

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

    def _add_check_embedding_sparsity_hook(self):
        """
        Add hook to check embedding sparsity and enable padding elimination if applicable.
        1. Iterate through all modules to find Embedding modules with padding_idx >= 0.
        2. Register forward hook to the Embedding module and the hook will check sparsity of the embedding input.
        3. If the sparsity is below a threshold, enable padding elimination by adding FlagPaddingElimination after the
           output. GraphTransformer of PaddingElimination will check the FlagPaddingElimination and do the actual
           padding elimination graph modification.
        4. Return the hook handles for later removal.

        """
        if (
            not self._runtime_options.enable_sparse_optimizer
            or not self._runtime_options.enable_embedding_sparse_optimizer
            or self._device.type != "cuda"
        ):
            return []

        def _embedding_hook(module, args, output):
            ebd_input = args[0]
            if ebd_input is None or not isinstance(ebd_input, torch.Tensor):
                self._logger.warning("Embedding input is not a tensor.")
                return None

            valid_token = torch.count_nonzero(ebd_input - module.padding_idx)
            total_token = ebd_input.numel()
            embed_density = float(valid_token) / float(total_token) * 100
            if embed_density < 90:
                self._logger.info("Embedding sparsity-based optimization is ON for density: %.0f%%", embed_density)
                if module not in self._runtime_inspector._embedding_module_to_padding_density_map:
                    self._logger.warning("Found Embedding module not in the map. %s", module)
                    return None
                if self._runtime_inspector._embedding_module_to_padding_density_map[module][1] != -1:
                    self._logger.warning(
                        "Found duplicate Embedding module. %s",
                        self._runtime_inspector._embedding_module_to_padding_density_map[module][0],
                    )
                self._runtime_inspector._embedding_module_to_padding_density_map[module][1] = embed_density
                return FlagPaddingElimination.apply(output)
            else:
                self._logger.info("Embedding sparsity-based optimization is OFF for density: %.0f%%", embed_density)
                return None

        embedding_hook_handles = []
        for name, sub_module in self._flattened_module.named_modules():
            if isinstance(sub_module, torch.nn.modules.sparse.Embedding):
                if sub_module.padding_idx is not None and sub_module.padding_idx >= 0:
                    self._runtime_inspector._embedding_module_to_padding_density_map[sub_module] = [name, -1]
                    embedding_hook_handles.append(sub_module.register_forward_hook(_embedding_hook))

        return embedding_hook_handles

    @_logger.TrackTime(_logger.ORTModuleInitPhase.DETECTION)
    def _enable_conditional_optimizations(
        self, graph_transformer_config: C.TrainingGraphTransformerConfiguration, inputs: Tuple, kwargs: Dict
    ):
        """
        Based on runtime inspection, enable conditional optimizations if applicable.

        Input sparsity-based optimization workflows:
        1. Input density observer is enabled if the sparse optimizer is ON or user wants to print input density.
        2. Input density observer inspects input tensors and returns sparsity results.
        3. If label or embedding input sparsity is found in sparsity results, graph transformer config is updated to
           enable sparsity-based optimization.

        """
        # Enable data sparsity inspection if sparse optimizer is ON or user wants to print input density.
        if self._runtime_options.enable_sparse_optimizer or self._runtime_options.print_input_density:
            self._runtime_inspector.enable_input_inspector(
                self._onnx_models.processed_exported_model, self._graph_builder.get_graph_info().user_input_names
            )

            if self._runtime_options.enable_sparse_optimizer:
                detected_device = _utils.get_device_from_module(self._original_module) or _utils.get_device_from_inputs(
                    inputs, kwargs
                )

                if self._runtime_options.enable_zero_stage3_support or self._mem_efficient_grad_management_is_enabled:
                    self._append_pull_weight_trigger_as_input(kwargs, detected_device)

                param_to_append_as_onnx_graph_inputs = []
                if self._mem_efficient_grad_management_is_enabled:
                    from ._mem_efficient_grad_mgmt import get_params_not_connected_to_pull_param_trigger

                    param_to_append_as_onnx_graph_inputs = get_params_not_connected_to_pull_param_trigger(
                        self._flattened_module.named_parameters(), self._onnx_models.exported_model
                    )
                else:
                    param_to_append_as_onnx_graph_inputs = self._graph_initializers

                _, _, label_sparsity_results = _io._combine_input_buffers_initializers(
                    param_to_append_as_onnx_graph_inputs,
                    self._graph_builder.get_graph_info().user_input_names,
                    self._input_info,
                    self._flattened_module.named_buffers(),
                    inputs,
                    kwargs,
                    detected_device,
                    self._runtime_inspector,
                    self._zero_stage3_param_map,
                )

                # Enable sparsity-based optimization when applicable.
                if len(label_sparsity_results) > 0:
                    graph_transformer_config.sparse_label_input_names = list(label_sparsity_results.keys())
                    self._logger.info("Label sparsity-based optimization is ON for %s", label_sparsity_results)
                    self._runtime_options.label_sparsity_ratio = ",".join(
                        [f"{k}:{v:.0f}%" for k, v in label_sparsity_results.items()]
                    )

                if self._runtime_inspector._embedding_module_to_padding_density_map:
                    self._runtime_options.embed_sparsity_ratio = ",".join(
                        [
                            f"{v[0]}:{v[1]:.0f}%"
                            for v in self._runtime_inspector._embedding_module_to_padding_density_map.values()
                        ]
                    )

            # If users don't want to print input density, disable the input density observer to avoid overhead
            # when looping through inputs during training.
            if not self._runtime_options.print_input_density:
                self._runtime_inspector.disable_input_inspector()

    def _append_pull_weight_trigger_as_input(self, kwargs: Dict, device: torch.device):
        if self._runtime_options.enable_zero_stage3_support:
            from ._zero_stage3_compatibility import (
                STAGE3_PULL_WEIGHT_TRIGGER_NAME,
                STAGE3_PULL_WEIGHT_TRIGGER_OUTPUT_DTYPE,
                STAGE3_PULL_WEIGHT_TRIGGER_OUTPUT_SHAPE,
            )

            kwargs[STAGE3_PULL_WEIGHT_TRIGGER_NAME] = torch.zeros(
                STAGE3_PULL_WEIGHT_TRIGGER_OUTPUT_SHAPE,
                dtype=onnx_dtype_to_pytorch_dtype(STAGE3_PULL_WEIGHT_TRIGGER_OUTPUT_DTYPE),
                device=device,
            ).requires_grad_()

        if self._mem_efficient_grad_management_is_enabled:
            from ._mem_efficient_grad_mgmt import (
                MEM_EFFICIENT_PARAM_TRIGGER_INPUT_NAME,
                MEM_EFFICIENT_PARAM_TRIGGER_OUTPUT_DTYPE,
                MEM_EFFICIENT_PARAM_TRIGGER_OUTPUT_SHAPE,
            )

            kwargs[MEM_EFFICIENT_PARAM_TRIGGER_INPUT_NAME] = torch.zeros(
                MEM_EFFICIENT_PARAM_TRIGGER_OUTPUT_SHAPE,
                dtype=onnx_dtype_to_pytorch_dtype(MEM_EFFICIENT_PARAM_TRIGGER_OUTPUT_DTYPE),
                device=device,
            ).requires_grad_()

    def _log_feature_stats(self):
        if get_rank() != 0:
            return

        tbl = PTable(sortable=True)

        def _add_record(tbl, columns):
            return tbl.add_row([columns[0], ":", "ON" if columns[1] else "OFF", ":", columns[2]])

        notes = []

        _add_record(tbl, ["ATen Executor", True, "Dispatch ATen operators to ORT's ATen executor"])
        _add_record(
            tbl,
            [
                "Cast Propagation",
                self._runtime_options.propagate_cast_ops_level > 0,
                f"Level {self._runtime_options.propagate_cast_ops_level} enabled",
            ],
        )
        _add_record(
            tbl,
            [
                "Custom Function",
                self._runtime_options.enable_custom_autograd_function,
                "Support custom torch.autograd.Function export and execution",
            ],
        )

        if self._runtime_options.memory_optimization_level == _MemoryOptimizationLevel.TRANSFORMER_LAYERWISE_RECOMPUTE:
            opt_config_to_display = "ALL_RECOMPUTE_FOR_EACH_LAYER"
        elif (
            self._runtime_options.memory_optimization_level
            == _MemoryOptimizationLevel.TRANSFORMER_LAYERWISE_RECOMPUTE_WITH_COMPROMISE
        ):
            opt_config_to_display = "ALL_RECOMPUTE_FOR_EACH_LAYER_WITH_COMPROMISE"
        else:
            opt_config_to_display = self._runtime_options.memory_optimizer_config

        mem_infos = ""
        if self._runtime_options.memory_optimizer_is_enabled():
            mem_infos += (
                f"Memory Optimization Level: [{_MemoryOptimizationLevel.to_string(self._runtime_options.memory_optimization_level)}], "
                f"Optimization Config: [{opt_config_to_display}]"
            )
        else:
            mem_infos = "Enable with env ORTMODULE_MEMORY_OPT_LEVEL=1/2 or ORTMODULE_MEMORY_OPT_CONFIG=<plan1 config>,<plan2 config>,..."

        mem_row = _add_record(
            tbl,
            [
                "Memory Optimizer",
                self._runtime_options.memory_optimizer_is_enabled(),
                mem_infos,
            ],
        )

        if self._runtime_inspector.memory_ob.is_enabled() and self._debug_options.logging.log_level < LogLevel.WARNING:
            mem_notes, mem_tbl = self._runtime_inspector.memory_ob.display_memory_optimization_plans(
                self._runtime_options.memory_optimizer_config,
                details=True,
            )
            if mem_tbl is not None:
                mem_row.append_annotation_table(mem_tbl)
                notes.extend([f"[{mem_row._columns[0]}] {n}" for n in mem_notes])

        compute_opt_row = _add_record(
            tbl,
            [
                "Compute Optimizer",
                self._runtime_options.enable_compute_optimizer,
                "Enable/Disable with env ORTMODULE_ENABLE_COMPUTE_OPTIMIZER=1/0",
            ],
        )

        compute_opt_annotation_tbl = PTable()
        _add_record(
            compute_opt_annotation_tbl,
            [
                " - FLOP Reduction",
                self._runtime_options.enable_compute_optimizer,
                "Reduce FLOPs by upstreaming shrinking-sized ops",
            ],
        )

        if self._runtime_options.enable_compute_optimizer:
            if len(self._runtime_options.label_sparsity_ratio) > 0:
                _add_record(
                    compute_opt_annotation_tbl,
                    [
                        " - Label Sparsity",
                        True,
                        f"[AUTO ENABLED] Input density: {self._runtime_options.label_sparsity_ratio}",
                    ],
                )

            if len(self._runtime_options.embed_sparsity_ratio) > 0:
                _add_record(
                    compute_opt_annotation_tbl,
                    [
                        " - Embed Sparsity",
                        True,
                        f"[AUTO ENABLED] Input density: {self._runtime_options.embed_sparsity_ratio}",
                    ],
                )

        compute_opt_row.append_annotation_table(compute_opt_annotation_tbl)

        # Add fallback
        _add_record(
            tbl,
            [
                "Auto Fallback",
                self._runtime_options.fallback_policy is not _FallbackPolicy.FALLBACK_DISABLE,
                "Fallback to PyTorch when encountering unsupported ops",
            ],
        )

        # Add Triton
        triton_row = _add_record(
            tbl,
            [
                "TritonOp Enabled",
                self._runtime_options.enable_triton,
                "ORT will switch to Triton for executing some ops to further accelerate training.",
            ],
        )

        triton_annotation_tbl = PTable()

        if self._runtime_options.enable_tuning:
            desc = "Enable tunning Ops online"
            if self._runtime_options.tuning_results_path:
                desc += f", save tuning results to {self._runtime_options.tuning_results_path}"
            _add_record(triton_annotation_tbl, ["Online Op Tuning", True, desc])
        elif self._runtime_options.tuning_results_path:
            _add_record(
                triton_annotation_tbl,
                [
                    "Offline Op Tuning",
                    True,
                    f"Use offline tuning results from {self._runtime_options.tuning_results_path}",
                ],
            )

        triton_row.append_annotation_table(triton_annotation_tbl)

        _add_record(
            tbl,
            [
                "ZeRO Stage3 Support",
                self._runtime_options.enable_zero_stage3_support,
                "Enable/Disable with env ORTMODULE_ENABLE_ZERO_STAGE3=1/0",
            ],
        )

        mode = "training" if self._export_mode == torch.onnx.TrainingMode.TRAINING else "inference"
        mode = f"{_logger.LogColor.UNDERLINE}{mode}{_logger.LogColor.ENDC}"
        stat = f"\n{_logger.LogColor.HEADER}***** ONNX Runtime Training (ORTModule) is accelerating your model *****{_logger.LogColor.ENDC}\n\n"
        stat += f"ORTModule is enabled with following features ON/OFF for [{mode}] mode:\n\n"
        stat += tbl.get_string() + "\n"

        # Collect ORTModule overheads for different phases.
        stat += f"\n{self.time_tracker.to_string(self._debug_options.logging.log_level < LogLevel.WARNING)}\n"
        stat += f"Versions: ONNX Runtime - {onnxruntime.__version__}, ONNX - {onnx.__version__}\n\n"

        # Add notes
        for index, note in enumerate(notes):
            stat += f"Note {index + 1}: {note}\n"

        stat += f"\n{_logger.LogColor.HEADER}************************************************************************{_logger.LogColor.ENDC}\n\n"
        self._logger.warning(stat)
