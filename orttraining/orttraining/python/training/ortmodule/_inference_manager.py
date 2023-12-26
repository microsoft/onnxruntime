# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import Logger
from typing import Tuple

import onnx
import torch

from onnxruntime.capi import _pybind_state as C

from . import _are_deterministic_algorithms_enabled, _io, _use_deterministic_algorithms, _utils
from ._execution_agent import InferenceAgent
from ._fallback import ORTModuleFallbackException, _FallbackManager, _FallbackPolicy
from ._graph_execution_manager import GraphExecutionManager, _RunStateInfo
from ._io import unflatten_user_output
from ._logger import ORTModuleInitPhase, TrackTime
from ._utils import save_tuning_results, set_tuning_results
from .options import DebugOptions, _SkipCheck


class InferenceManager(GraphExecutionManager):
    """Concrete instance of GraphExecutionManager that is able to manage the inference model

    InferenceManager is responsible for building and running the forward graph of the inference model
    """

    def __init__(self, model, debug_options: DebugOptions, fallback_manager: _FallbackManager, logger: Logger):
        super().__init__(model, debug_options, fallback_manager, logger)
        self._export_mode = torch.onnx.TrainingMode.EVAL

    @staticmethod
    def execution_session_run_forward(
        execution_session,
        onnx_model: onnx.ModelProto,
        device: torch.device,
        *inputs,
    ) -> Tuple[Tuple[torch.Tensor, ...], _RunStateInfo]:
        """Runs the forward pass on `execution_session` with given `onnx_model`, `device` and `inputs`

        Args:
            execution_session InferenceAgent: Agent which runs inference
            onnx_model (onnx.ModelProto): ONNX model
            device (torch.device): PyTorch device
            inputs: (torch.Tensor or a container of): User inputs passed from ORTModule.forward().

        Returns:
            Returns a tuple (user_outputs, run_info):
                user_outputs: The model output (either torch.Tensor or a container of torch.Tensor)
                run_info: A _RunStateInfo which contains extra information about the execution of the graph
        """
        # TODO: Try to reuse the output buffers as some of the output tensors are same sizes,
        #   especially the backward graph outputs.
        # REVIEW(codemzs): Consolidate Training Agent with InferenceAgent on C++ side to not
        # have the need for passing IOBinding.
        io_binding = execution_session.io_binding()
        run_options = C.RunOptions()

        # Use IO binding
        _utils._create_iobinding(io_binding, inputs, onnx_model, device)

        # Run and return module outputs.
        ort_output = execution_session.run_forward(io_binding, run_options)
        forward_outputs = ort_output.ortvalues
        user_outputs = _utils._ortvalues_to_torch_tensor(forward_outputs, device)  # pylint: disable=W0212
        state = None

        output_info = [(output.shape, output.device, output.dtype) for output in user_outputs]
        run_info = _RunStateInfo(state, output_info)
        # Return user outputs and forward run information
        return user_outputs, run_info

    def forward(self, *inputs, **kwargs):
        """Forward pass of the inference model

        ONNX model is exported the first time this method is executed.
        Next, we build an optimized inference graph with module_graph_builder.
        Finally, we instantiate the ONNX Runtime InferenceSession through the InferenceAgent.

        The call stack is as follows:
            ORTModule.forward(*inputs, **kwargs) ->
            ORTModule._torch_module.forward(*inputs, **kwargs) where _torch_module is a TorchModuleORT instance ->
            ORTModule._torch_module._execution_manager(is_training()).forward(*inputs, **kwargs) where:
                TorchModuleORT._execution_manager(true) is a TrainingManager instance;
                and TorchModuleORT._execution_manager(false) is an InferenceManager instance.

        """

        # Fallback to PyTorch due to failures *external* to forward(),
        #  typically from initialization
        if self._fallback_manager.is_pending():
            return self._fallback_manager.fallback(self._debug_options.logging.log_level, *inputs, **kwargs)

        try:
            # Issue at most one warning message about fast path
            if self._first_skip_check_warning is True and self._runtime_options.skip_check.is_disabled() is False:
                self._first_skip_check_warning = False
                self._logger.warning(
                    "Fast path enabled - skipping checks. rebuild gradient graph: %s, execution agent recreation: %s, "
                    "device check: %s",
                    self._runtime_options.skip_check.is_set(_SkipCheck.SKIP_CHECK_BUILD_GRADIENT),
                    self._runtime_options.skip_check.is_set(_SkipCheck.SKIP_CHECK_EXECUTION_AGENT),
                    self._runtime_options.skip_check.is_set(_SkipCheck.SKIP_CHECK_DEVICE),
                )

            # If exporting module to ONNX for the first time, this skip check will not take effect.
            # It will only take effect on subsequent forward calls.
            build_graph = False
            if (
                self._runtime_options.skip_check.is_set(_SkipCheck.SKIP_CHECK_BUILD_GRADIENT) is False
                or not self._onnx_models.exported_model
            ):
                self.time_tracker.start(ORTModuleInitPhase.EndToEnd)

                # Exporting module to ONNX for the first time
                build_graph = self._export_model(*inputs, **kwargs)
                if build_graph:
                    # If model was exported, then initialize the graph builder.
                    self._initialize_graph_builder()

                # Build the inference graph
                if build_graph:
                    graph_transformer_config = self._get_graph_transformer_config()
                    # Set the config according to input inspection.
                    self._enable_conditional_optimizations(graph_transformer_config, inputs, kwargs)

                    # Build the graph
                    self._build_graph(graph_transformer_config)

            # If creating the execution agent for the first time, this skip check will not take effect.
            # It will only take effect on subsequent forward calls.
            create_execution_session = False
            if (
                self._runtime_options.skip_check.is_set(_SkipCheck.SKIP_CHECK_EXECUTION_AGENT) is False
                or not self._execution_agent
            ):
                module_device = _utils.get_device_from_module(self._original_module)

                create_execution_session = (
                    build_graph
                    or self._device != module_device
                    or torch.are_deterministic_algorithms_enabled() is not _are_deterministic_algorithms_enabled()
                )
                _use_deterministic_algorithms(torch.are_deterministic_algorithms_enabled())

                if self._device != module_device:
                    self._device = module_device

            if create_execution_session:
                # Create execution session creates the inference_session
                self._create_execution_agent()

                self.time_tracker.end(ORTModuleInitPhase.EndToEnd)
                self._log_feature_stats()

            if self._runtime_options.skip_check.is_set(_SkipCheck.SKIP_CHECK_DEVICE) is False:
                # Assert that the input and model device match
                _utils._check_same_device(self._device, "Input argument to forward", *inputs)

            if (
                self._runtime_options.enable_zero_stage3_support
                or self._runtime_options.enable_mem_efficient_grad_management
            ):
                self._append_pull_weight_trigger_as_input(kwargs, self._device)

            param_to_append_as_onnx_graph_inputs = []
            if self._runtime_options.enable_mem_efficient_grad_management:
                from ._mem_efficient_grad_mgmt import get_params_not_connected_to_pull_param_trigger

                param_to_append_as_onnx_graph_inputs = get_params_not_connected_to_pull_param_trigger(
                    self._flattened_module.named_parameters()
                )
            else:
                param_to_append_as_onnx_graph_inputs = self._graph_initializers

            prepared_input_list, _, _ = _io._combine_input_buffers_initializers(
                param_to_append_as_onnx_graph_inputs,
                self._graph_info.user_input_names,
                self._input_info,
                self._flattened_module.named_buffers(),
                inputs,
                kwargs,
                self._device,
                self._runtime_inspector,
                self._zero_stage3_param_map,
            )

            user_outputs, _ = InferenceManager.execution_session_run_forward(
                self._execution_agent,
                self._onnx_models.optimized_model,
                self._device,
                *prepared_input_list,
            )

            if (
                create_execution_session
                and self._runtime_options.enable_tuning
                and self._runtime_options.tuning_results_path
            ):
                save_tuning_results(
                    self._execution_agent._inference_session, False, self._runtime_options.tuning_results_path
                )

            return unflatten_user_output(self._module_output_schema, user_outputs)
        except ORTModuleFallbackException as e:
            # Exceptions subject to fallback are handled here
            self._fallback_manager.handle_exception(exception=e, log_level=self._debug_options.logging.log_level)
        except Exception as e:
            # Catch-all FALLBACK_FORCE_TORCH_FORWARD fallback is handled here
            self._fallback_manager.handle_exception(
                exception=e,
                log_level=self._debug_options.logging.log_level,
                override_policy=_FallbackPolicy.FALLBACK_FORCE_TORCH_FORWARD,
            )
        # Fallback to PyTorch due to failures *during* forward(),
        #  (e.g. export, model/input post-processing, forward, output processing, etc)
        if self._fallback_manager.is_pending():
            return self._fallback_manager.fallback(self._debug_options.logging.log_level, *inputs, **kwargs)

    @TrackTime(ORTModuleInitPhase.BUILD_GRAPH)
    def _build_graph(self, graph_transformer_config):
        """Build an inference graph using the module_graph_builder"""

        super()._build_graph(graph_transformer_config)
        self._onnx_models.optimized_model = onnx.load_model_from_string(self._graph_builder.get_forward_model())
        if self._debug_options.save_onnx_models.save:
            self._onnx_models.save_optimized_model(
                self._debug_options.save_onnx_models.path,
                self._debug_options.save_onnx_models.name_prefix,
                self._export_mode,
            )

    @TrackTime(ORTModuleInitPhase.CREATE_SESSION)
    def _create_execution_agent(self):
        """Creates an InferenceAgent that can run forward graph on an inference model"""

        session_options, providers, provider_options = self._get_session_config()
        self._execution_agent = InferenceAgent(
            self._onnx_models.optimized_model.SerializeToString(), session_options, providers, provider_options
        )

        if not self._runtime_options.enable_tuning and self._runtime_options.tuning_results_path:
            set_tuning_results(
                self._execution_agent._inference_session, False, self._runtime_options.tuning_results_path
            )
