# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from . import _utils, _io, _logger
from ._graph_execution_manager import GraphExecutionManager, _RunStateInfo, _SkipCheck
from ._execution_agent import TrainingAgent
from .debug_options import DebugOptions
from ._fallback import ORTModuleFallbackException, _FallbackPolicy, _FallbackManager

from onnxruntime.capi import _pybind_state as C
from onnxruntime.capi.onnxruntime_inference_collection import get_ort_device_type

import torch
import warnings
from torch.utils.dlpack import from_dlpack, to_dlpack


class TrainingManager(GraphExecutionManager):
    """Concrete instance of GraphExecutionManager that is able to manage the training model

    TrainingManager is responsible for building and running the forward and backward graph of the training model
    """

    def __init__(self, model, debug_options: DebugOptions, fallback_manager: _FallbackManager):
        super().__init__(model, debug_options, fallback_manager)
        self._export_mode = torch.onnx.TrainingMode.TRAINING

    @staticmethod
    def execution_session_run_forward(execution_session, onnx_model, gradient_accumulation_manager, *inputs):
        """Runs the forward graph on execution_session with given model inputs and device"""

        # TODO: Try to reuse the output buffers as some of the output tensors are same sizes,
        #   especially the backward graph outputs.
        # REVIEW(codemzs): Consolidate Training Agent with InferenceAgent on C++ side to not
        # have the need for passing IOBinding.
        state = C.PartialGraphExecutionState()
        forward_inputs = C.OrtValueVector()
        forward_inputs.reserve(len(inputs))
        for input in inputs:
            forward_inputs.push_back(to_dlpack(input), input.dtype == torch.bool)

        forward_outputs = C.OrtValueVector()
        # Run and return module outputs.
        execution_session.run_forward(forward_inputs, forward_outputs, state, gradient_accumulation_manager.cache)
        user_outputs = gradient_accumulation_manager.extract_outputs_and_maybe_update_cache(forward_outputs)

        output_info = [(output.shape, output.device, output.dtype) for output in user_outputs]
        run_info = _RunStateInfo(state, output_info)
        # Return user outputs and forward run information
        return user_outputs, run_info

    def forward(self, *inputs, **kwargs):
        '''Forward pass starts here and continues at `_ORTModuleFunction.forward`

        ONNX model is exported the first time this method is executed.
        Next, we build a full training graph with module_graph_builder.
        Finally, we instantiate the ONNX Runtime InferenceSession.
        '''

        # Fallback to PyTorch due to failures *external* to forward(),
        #  typically from initialization
        if self._fallback_manager.is_pending():
            return self._fallback_manager.fallback(self._original_module, self._debug_options.logging.log_level, *inputs, **kwargs)

        try:
            if self._first_skip_check_warning == True and self._skip_check.is_disabled() == False \
                and self._debug_options.logging.log_level <= _logger.LogLevel.WARNING:
                # Only change this after the firs time a warning is issued.
                self._first_skip_check_warning = False
                warnings.warn(f"Fast path enabled - skipping checks."
                              f"rebuild gradient graph: {self._skip_check.is_set(_SkipCheck.SKIP_CHECK_BUILD_GRADIENT)},"
                              f"execution agent recreation: {self._skip_check.is_set(_SkipCheck.SKIP_CHECK_EXECUTION_AGENT)},"
                              f"device check: {self._skip_check.is_set(_SkipCheck.SKIP_CHECK_DEVICE)}", UserWarning)

            # If exporting module to ONNX for the first time, this skip check will not take effect.
            # It will only take effect on subsequent forward calls.
            build_gradient_graph = False
            if self._skip_check.is_set(_SkipCheck.SKIP_CHECK_BUILD_GRADIENT) == False or \
                not self._onnx_models.exported_model:
                build_gradient_graph = self._export_model(*inputs, **kwargs)
                if build_gradient_graph:
                    # If model was exported, then initialize the graph builder
                    self._initialize_graph_builder(training=True)

                # since the schema was just extracted while trying to export the model and it was either
                # saved to self._input_info.schema or checked for equality with the self._input_info.schema
                # it should not need to be updated again. Pass it inside parse_inputs_for_onnx_export.
                input_info = _io.parse_inputs_for_onnx_export(self._module_parameters,
                                                              self._onnx_models.exported_model,
                                                              self._input_info.schema,
                                                              inputs,
                                                              kwargs)

                # Reinitialize graph builder if the inputs or initializers requiring gradient have changed.
                # Order of or operation is important here because we always need to call
                # _reinitialize_graph_builder irrespective of the value of build_gradient_graph.
                build_gradient_graph = self._reinitialize_graph_builder(
                    input_info) or build_gradient_graph

                # Build the gradient graph
                if build_gradient_graph:
                    self._build_graph()

            # If creating the execution agent for the first time, this skip check will not take effect.
            # It will only take effect on subsequent forward calls.
            create_execution_session = False
            if self._skip_check.is_set(_SkipCheck.SKIP_CHECK_EXECUTION_AGENT) == False or \
                not self._execution_agent:
                device = _utils.get_device_from_module(self._original_module) or \
                    _utils.get_device_from_inputs(inputs, kwargs)
                # The _training_session/_inference_session should be created every time
                # the graph was built or if the device changed between calls to forward
                create_execution_session = build_gradient_graph or self._device != device
                if self._device != device:
                    self._device = device

            if create_execution_session:
                # Create execution session creates the training_session
                self._create_execution_agent()

                self._gradient_accumulation_manager.initialize(self._enable_grad_acc_optimization, self._flattened_module, self._graph_info)
        
            self._gradient_accumulation_manager.maybe_update_cache_before_run()

            class _ORTModuleFunction(torch.autograd.Function):
                '''Use a custom torch.autograd.Function to associate self.backward_graph as the
                gradient implementation for self.forward_graph.'''

                @staticmethod
                def forward(ctx, *inputs):
                    '''Performs forward pass based on user input and PyTorch initializer

                    Autograd Function's apply() doesn't support keyword arguments,
                    so `*inputs` has all the arguments - keyword arguments converted
                    to positional/keywords during `TrainingManager.forward`.

                    Module outputs are returned to the user
                    '''

                    if self._skip_check.is_set(_SkipCheck.SKIP_CHECK_DEVICE) == False:
                        # Assert that the input and model device match
                        _utils._check_same_device(self._device, "Input argument to forward", *inputs)

                    user_outputs, ctx.run_info = TrainingManager.execution_session_run_forward(self._execution_agent,
                                                                                               self._onnx_models.optimized_model,
                                                                                               self._gradient_accumulation_manager,
                                                                                               *inputs)

                    # Disable materializing grads then None object will not be
                    # converted to a tensor filled with zeros prior to calling backward.
                    # Save shape, device and type info to ctx for materializing tensor in backward if output grad is None.
                    ctx.set_materialize_grads(False)

                    # Mark the outputs tensors needed in backward computation
                    # ORT is NOT relying on save_for_backward() to actually save the tensor,
                    # as this tensor is also kept in ORT's PartialGraphState
                    # This call is to invoke pytorch's version check to detect the potential inplace corruption
                    # If ORT is caching tensors, the module_output_indices_requires_save_for_backward field
                    # might also have indices of cached tensors that are not passed over to pytorch, and they don't 
                    # need marking with save_for_backward()
                    for idx in self._graph_info.module_output_indices_requires_save_for_backward:
                        if idx < len(self._graph_info.user_output_names):
                            ctx.save_for_backward(user_outputs[idx])

                    # Mark the outputs tensors non-differentiable if requires_grad is False in _graph_info
                    # This will return torch the output tensors with correct requires_grad settings
                    for idx in self._graph_info.output_grad_indices_non_differentiable:
                        ctx.mark_non_differentiable(user_outputs[idx])

                    return user_outputs

                @staticmethod
                def backward(ctx, *grad_outputs):
                    '''Performs backward pass based on grad wrt module output'''

                    assert ctx.run_info is not None, 'forward() or __call__() methods must be called before backward()'
                    if self._skip_check.is_set(_SkipCheck.SKIP_CHECK_DEVICE) == False:
                        _utils._check_same_device(self._device, "Input argument to backward", *grad_outputs)

                    # Unpack saved_tensor to trigger version detection that catches inplace corruption
                    _ = ctx.saved_tensors

                    # Use IO binding
                    # Push user output grads to ONNX backend.
                    backward_inputs = C.OrtValueVector()
                    # Preallocate length of the vector. And then delete as required towards the end.
                    backward_inputs.reserve(len(grad_outputs))
                    for idx, grad_output in enumerate(grad_outputs):
                        if idx in self._graph_info.output_grad_indices_non_differentiable:
                            assert grad_output is None, "ORT found the {}-th module output '{}' is " \
                                                        "non-differentiable according to the onnx graph. " \
                                                        "However, the gradient value is still provided by " \
                                                        "PyTorch's autograd engine." \
                                                        .format(idx, self._graph_info.user_output_names[idx])
                            continue

                        if grad_output is None:
                            shape, device, dtype = ctx.run_info.output_info[idx]
                            if idx in self._graph_info.output_grad_indices_require_full_shape:
                                grad_output = torch.zeros(shape, device=device, dtype=dtype)
                            else:
                                grad_output = torch.tensor(0., device=device, dtype=dtype)
                        elif not grad_output.is_contiguous():
                            grad_output = grad_output.contiguous()
                        backward_inputs.push_back(to_dlpack(grad_output), grad_output.dtype == torch.bool)
                    backward_inputs.shrink_to_fit()

                    # Run and get results
                    backward_outputs = C.OrtValueVector()
                    self._execution_agent.run_backward(backward_inputs, backward_outputs, ctx.run_info.state)
                    # Destroy the state immediately (as opposed to be at the mercy of garbage collector) so it does not
                    # affect peak memory usage in a subsequent graph run.
                    del ctx.run_info.state
                    # Return input and initializer gradients
                    num_user_input_grads = len(self._input_info.require_grad_names)
                    results = []
                    require_grad_names_set = set(self._input_info.require_grad_names)
                    require_grad_names_index = 0
                    for input_name in self._graph_info.user_input_names:
                        # Append to the results the backward output for each input that required grad
                        if input_name in require_grad_names_set:
                            results.append(_utils._torch_tensor_from_dl_pack(
                                backward_outputs.dlpack_at(require_grad_names_index),
                                backward_outputs[require_grad_names_index]))
                            require_grad_names_index += 1
                        else:
                            # input_name is not found in the self._input_info.require_grad_names list
                            # Append None to results for each input that did not require grad
                            results.append(None)

                    # Append gradients of initializer to results
                    # Go over each initializer, check if it required grad and append to results accordingly
                    initializer_index = num_user_input_grads
                    for initializer_name in self._graph_info.initializer_names:
                        if initializer_name in self._graph_initializer_names_to_train:
                            results.append(_utils._torch_tensor_from_dl_pack(
                                backward_outputs.dlpack_at(initializer_index),
                                backward_outputs[initializer_index]))
                            initializer_index += 1
                        else:
                            results.append(None)

                    return tuple(results)

            return _io.unflatten_user_output(self._module_output_schema,
                                             _ORTModuleFunction.apply(
                                                 *_io._combine_input_buffers_initializers(
                                                     self._graph_initializers,
                                                     self._graph_info.user_input_names,
                                                     self._input_info,
                                                     self._flattened_module.named_buffers(),
                                                     inputs,
                                                     kwargs,
                                                     self._device)))
        except ORTModuleFallbackException as e:
            # Exceptions subject to fallback are handled here
            self._fallback_manager.handle_exception(exception=e,
                                                    log_level=self._debug_options.logging.log_level)
        except Exception as e:
            # Catch-all FALLBACK_FORCE_TORCH_FORWARD fallback is handled here
            self._fallback_manager.handle_exception(exception=e,
                                                    log_level=self._debug_options.logging.log_level,
                                                    override_policy=_FallbackPolicy.FALLBACK_FORCE_TORCH_FORWARD)

        # Fallback to PyTorch due to failures *during* forward(),
        #  (e.g. export, model/input post-processing, forward, output processing, etc)
        if self._fallback_manager.is_pending():
            return self._fallback_manager.fallback(self._original_module, self._debug_options.logging.log_level, *inputs, **kwargs)

    def _build_graph(self):
        """Build an optimized gradient graph using the module_graph_builder"""

        super()._build_graph()

        if self._debug_options.save_onnx_models.save:
            self._onnx_models.save_optimized_model(self._debug_options.save_onnx_models.path,
                                                   self._debug_options.save_onnx_models.name_prefix,
                                                   self._export_mode)

    def _create_execution_agent(self):
        """Creates a TrainingAgent that can run the forward and backward graph on the training model"""

        session_options, providers, provider_options = self._get_session_config()
        fw_feed_names = [input.name for input in self._onnx_models.optimized_model.graph.input]
        fw_outputs_device_info = [
            C.OrtDevice(get_ort_device_type(self._device.type),
                        C.OrtDevice.default_memory(),
                        _utils.get_device_index(self._device)
                        )] * (len(self._graph_info.user_output_names) + 
                              len(self._graph_info.frontier_node_arg_map))

        bw_fetches_names = [output.name for output in self._onnx_models.optimized_model.graph.output]
        bw_outputs_device_info = [
            C.OrtDevice(get_ort_device_type(self._device.type),
                        C.OrtDevice.default_memory(),
                        _utils.get_device_index(self._device)
                        )] * len(bw_fetches_names)

        self._execution_agent = TrainingAgent(self._onnx_models.optimized_model.SerializeToString(),
                                              fw_feed_names,
                                              fw_outputs_device_info,
                                              bw_fetches_names,
                                              bw_outputs_device_info,
                                              session_options,
                                              providers,
                                              provider_options)

    def _reinitialize_graph_builder(self, input_info):
        """Return true if the module graph builder was reinitialized"""

        # Model could have unused parameters which are dropped after export and so not a part of self._graph_initializer_names_to_train.
        # To see if any trainable initializers changed, compare self._graph_initializer_names_to_train
        # with initializers in module named_parameters that are known to the onnx graph.
        initializer_names_to_train_set_user_model = {name for name, param in
                                                     self._flattened_module.named_parameters()
                                                     if param.requires_grad and name in self._graph_initializer_names}

        # If inputs requiring gradient change from forward to the next, the module_gradient_graph_builder
        # needs to be reinitialized so it can compute the backward output for the new inputs that require_grad
        if input_info.require_grad_names != self._input_info.require_grad_names or \
                initializer_names_to_train_set_user_model != self._graph_initializer_names_to_train:
            self._input_info = input_info
            self._initialize_graph_builder(training=True)
            return True
        return False
