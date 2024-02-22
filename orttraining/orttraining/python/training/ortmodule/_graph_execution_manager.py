# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import copy
import logging
import os
from abc import ABC, abstractmethod  # noqa: F401
from typing import Dict, List, Optional, OrderedDict, Tuple

import onnx
import torch
from torch.utils.cpp_extension import ROCM_HOME

import onnxruntime
from onnxruntime.capi import _pybind_state as C
from onnxruntime.training.utils import PTable, onnx_dtype_to_pytorch_dtype
from onnxruntime.training.utils.hooks import configure_ort_compatible_zero_stage3

from . import _are_deterministic_algorithms_enabled, _logger, _onnx_models, _utils
from ._fallback import ORTModuleTorchModelException, _FallbackManager, _FallbackPolicy, wrap_exception
from ._gradient_accumulation_manager import GradientAccumulationManager
from ._graph_execution_interface import GraphExecutionInterface
from ._graph_transition_manager import GraphTransitionManager, PostExportProcessedModelInfo
from ._io import _FlattenedModule
from ._runtime_inspector import RuntimeInspector
from ._utils import get_rank
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
        fallback_manager: _FallbackManager,
        export_mode: int,
        logger: logging.Logger,
    ):
        """Manages construction and execution of ONNX graphs"""

        super().__init__(module._original_module)

        # IMPORTANT: Debug and Fallback must the configured first
        self._debug_options = debug_options
        self._fallback_manager = fallback_manager

        self._logger = logger

        # Management for ORTModule configuration.
        self._runtime_options = _RuntimeOptions(self._logger)

        # Original and flattened (transformed) output module
        self._flattened_module = module

        self._onnx_models = _onnx_models.ONNXModels()
        self._export_mode = export_mode
        self._graph_transition_manager: Optional[GraphTransitionManager] = None

        # Model after inference optimization and then gradient building.
        self._graph_builder = None
        self._graph_info = None

        # TrainingAgent or InferenceAgent
        self._execution_agent = None

        self._first_skip_check_warning = True

        # Inspector for runtime information, for example input data, memory usage, etc.
        self._runtime_inspector = RuntimeInspector(self._logger, self._original_module)
        self._runtime_inspector.memory_ob.enable_memory_stats_by_step(self._runtime_options.print_memory_stat_by_step)

        # Tracker for session creation overhead.
        self.time_tracker = _logger.TimeTracker()

        self.is_rocm_pytorch = bool(torch.version.hip is not None and ROCM_HOME is not None)

        # WIP feature to enable caching in Gradient accumulation scenario.
        self._gradient_accumulation_manager = GradientAccumulationManager()

        # Load ATen operator executor extension.
        load_aten_op_executor_cpp_extension()

        # Assign self._torch_alloc and self._torch_free if self._use_external_gpu_allocator is True
        self._get_torch_gpu_allocator_function_addresses()

        if self._runtime_options.enable_triton:
            from onnxruntime.training.ort_triton import register_triton_op_executor

            register_triton_op_executor()

        self._zero_stage3_param_map = {}
        if self._runtime_options.enable_zero_stage3_support:
            # Cannot toggle feature enabling/disabling after the first time enabled.

            configure_ort_compatible_zero_stage3(debug=False, stats_output_dir="ort_output", stats_overwrite=True)

        self._initialize_graph_transition_manager()

    def _get_torch_gpu_allocator_function_addresses(self):
        if self._runtime_options.use_external_gpu_allocator and torch.cuda.is_available():
            # CPP extension to get torch GPU allocator's alloc and free function addresses
            from onnxruntime.training.ortmodule.torch_cpp_extensions import torch_gpu_allocator

            self._torch_alloc = torch_gpu_allocator.gpu_caching_allocator_raw_alloc_address()
            self._torch_free = torch_gpu_allocator.gpu_caching_allocator_raw_delete_address()
            self._torch_empty_cache = torch_gpu_allocator.gpu_caching_allocator_empty_cache_address()

    def _initialize_graph_transition_manager(self):
        """Creates a new GraphTransitionManager, initializes it and saves it to self._graph_transition_manager"""
        self._graph_transition_manager = GraphTransitionManager(
            flatten_module=self._flattened_module,
            export_mode=self._export_mode,
            debug_options=self._debug_options,
            runtime_options=self._runtime_options,
            time_tracker=self.time_tracker,
            logger=self._logger,
        )

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
        pass

    def _build_graph(self, config):
        if self._runtime_options.use_static_shape:
            self._graph_builder.build(
                config, self._graph_transition_manager._model_info_for_export.onnx_graph_input_shapes
            )
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
    def _initialize_graph_builder(self, post_export_processed_model_info: PostExportProcessedModelInfo):
        """Creates a new OrtModuleGraphBuilder, initializes it and saves it to self._graph_builder"""

        # Build and optimize the full graph
        grad_builder_config = C.OrtModuleGraphBuilderConfiguration()
        grad_builder_config.initializer_names = (
            post_export_processed_model_info.onnx_graph_input_names
        )  # containing both user defined and buffers/parameters.
        grad_builder_config.initializer_names_to_train = (
            post_export_processed_model_info.onnx_graph_input_names_require_grad
        )  # containing both user defined and parameters requiring gradients.

        input_names_require_grad = post_export_processed_model_info.onnx_graph_input_names_require_grad_user_defined
        if self._runtime_options.enable_zero_stage3_support:
            from ._zero_stage3_compatibility import STAGE3_PULL_WEIGHT_TRIGGER_NAME

            # Add stage3 pull weight trigger name to require_grad_names, so that it will be included in the gradient graph.
            input_names_require_grad.append(STAGE3_PULL_WEIGHT_TRIGGER_NAME)

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
        self._graph_builder.initialize(
            post_export_processed_model_info._post_export_processed_model.SerializeToString(), grad_builder_config
        )

    def __getstate__(self):
        state = copy.copy(self.__dict__)
        # Remove any re-contructible/pybound object from the state
        serialization_deny_list = [
            "_onnx_models",
            "_graph_builder",
            "_graph_info",
            "_graph_transition_manager",  # Not pickled as it is re-constructed in __setstate__
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

        self._initialize_graph_transition_manager()

    @property
    def _device(self):
        # Graph transition manager is responsible for detecting and managing the device to use.
        return self._graph_transition_manager._device

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
                self._graph_transition_manager._exported_model_info.exported_model,
                self._graph_transition_manager._model_info_for_export.onnx_graph_input_names,
            )

            if self._runtime_options.enable_sparse_optimizer:
                detected_device = _utils.get_device_from_module_and_inputs(self._original_module, inputs, kwargs)

                if self._runtime_options.enable_zero_stage3_support or self._mem_efficient_grad_management_is_enabled:
                    self._append_pull_weight_trigger_as_input(kwargs, detected_device)

                prepared_input_map = self._graph_transition_manager._post_export_processed_model_info.construct_inputs(
                    inputs, kwargs, True, self._device
                )

                embed_sparsity_results = OrderedDict()
                label_sparsity_results = OrderedDict()

                for name, inp in prepared_input_map.items():
                    found, embedding_density, label_density = self._runtime_inspector.inspect_input(name, inp)
                    if found:
                        if embedding_density < 100:
                            embed_sparsity_results[name] = embedding_density
                        if label_density < 100:
                            label_sparsity_results[name] = label_density
                if (
                    self._runtime_inspector.memory_ob.is_enabled()
                    and not self._runtime_inspector.memory_ob.symbolic_dim_collecting_completed
                ):
                    self._runtime_inspector.memory_ob.collect_symbolic_dim_values(
                        self._graph_transition_manager._post_export_processed_model_info.onnx_graph_input_dynamic_axes_map,
                        prepared_input_map,
                    )
                    self._runtime_inspector.memory_ob.symbolic_dim_collecting_completed = True

                # Enable sparsity-based optimization when applicable.
                if len(label_sparsity_results) > 0:
                    graph_transformer_config.sparse_label_input_names = list(label_sparsity_results.keys())
                    self._logger.info("Label sparsity-based optimization is ON for %s", label_sparsity_results)
                    self._runtime_options.label_sparsity_ratio = ",".join(
                        [f"{k}:{v:.0f}%" for k, v in label_sparsity_results.items()]
                    )

                if self._runtime_options.enable_embedding_sparse_optimizer and len(embed_sparsity_results) > 0:
                    graph_transformer_config.sparse_embedding_input_names = list(embed_sparsity_results.keys())
                    self._logger.info("Embedding sparsity-based optimization is ON for %s", embed_sparsity_results)
                    self._runtime_options.embed_sparsity_ratio = ",".join(
                        [f"{k}:{v:.0f}%" for k, v in embed_sparsity_results.items()]
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
        else:
            opt_config_to_display = self._runtime_options.memory_optimizer_config

        mem_row = _add_record(
            tbl,
            [
                "Memory Optimizer",
                len(self._runtime_options.memory_optimizer_config) > 0,
                (
                    f"Memory Optimization Level: [{_MemoryOptimizationLevel.to_string(self._runtime_options.memory_optimization_level)}], "
                    f"Optimization Config: [{opt_config_to_display}]"
                    if len(self._runtime_options.memory_optimizer_config) > 0
                    else "Enable with env ORTMODULE_MEMORY_OPT_LEVEL=1 or ORTMODULE_MEMORY_OPT_CONFIG=<plan1 config>,<plan2 config>,..."
                ),
            ],
        )

        if self._runtime_inspector.memory_ob.is_enabled() and self._debug_options.logging.log_level < LogLevel.WARNING:
            mem_notes, mem_tbl = self._runtime_inspector.memory_ob.display_memory_optimization_plans(
                self._runtime_options.memory_optimizer_config,
                details=True,
            )
            if mem_tbl is not None:
                mem_row.append_annotation_table(mem_tbl)
                notes.extend(mem_notes)

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
                    [" - Label Sparsity Opt", True, f"Input density: {self._runtime_options.label_sparsity_ratio}"],
                )

            if len(self._runtime_options.embed_sparsity_ratio) > 0:
                _add_record(
                    compute_opt_annotation_tbl,
                    [" - Embed Sparsity Opt", True, f"Input density: {self._runtime_options.embed_sparsity_ratio}"],
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
