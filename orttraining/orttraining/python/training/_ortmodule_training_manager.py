# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from . import _utils, _ortmodule_utils, _ortmodule_output_transformation as _ortmodule_io
from ._ortmodule_graph_execution_manager import GraphExecutionManager, _run_forward

import onnx
import onnxruntime

import torch

class TrainingManager(GraphExecutionManager):
    """Concrete instance of GraphExecutionManager that is able to manage the training model

    TrainingManager is resposible for building and running the forward and backward graph of the training model
    """
    def __init__(self, model):
        super().__init__(model)

        self._export_mode = torch.onnx.TrainingMode.TRAINING

    def forward(self, *inputs, **kwargs):
        '''Forward pass starts here and continues at `_ORTModuleFunction.forward`

        ONNX model is exported the first time this method is executed.
        Next, we build a full training graph with module_graph_builder.
        Finally, we instantiate the ONNX Runtime InferenceSession.
        '''

        # Exporting module to ONNX for the first time
        # Flag to indicate whether the gradient graph needs to be built
        # It should be built every time the module graph builder is initialized
        build_gradient_graph = \
            self._get_exported_model_and_init_graph_builder(
                *inputs, **kwargs)

        _, _, input_names_require_grad, new_input_shape = \
            _ortmodule_io.parse_inputs_for_onnx_export(
                self._module_parameters, self._onnx_model, *inputs, **kwargs)
        # Reinitialize graph builder if the inputs or initializers requiring gradient have changed.
        build_gradient_graph = build_gradient_graph or \
            self._reinitialize_graph_builder(input_names_require_grad)

        # Build the gradient graph
        if build_gradient_graph:
            self._current_input_shape = new_input_shape
            self._build_graph()

        module_device = _utils.get_device_from_module(self._original_module)
        # The _training_session/_inference_session should be created every time
        # the graph was built or if the device changed between calls to forward
        create_execution_session = \
            build_gradient_graph or self._device != module_device
        if self._device != module_device:
            self._device = module_device

        if create_execution_session:
            # Create execution session creates the training_session
            self._create_execution_agent()

        class _ORTModuleFunction(torch.autograd.Function):
            '''Use a custom torch.autograd.Function to associate self.backward_graph as the
            gradient implementation for self.forward_graph.'''

            @staticmethod
            def forward(ctx, *inputs, **kwargs):
                '''Performs forward pass based on user input and PyTorch initializer

                Autograd Function's apply() doesn't support keyword arguments,
                so `*inputs` has all the arguments - keyword arguments converted
                to positional by the caller.

                Module outputs are returned to the user
                '''

                user_outputs, ctx.run_info = _run_forward(self._execution_agent,
                                                          self._optimized_onnx_model,
                                                          self._device,
                                                          *inputs,
                                                          **kwargs)

                # Disable materializing grads then None object will not be converted to a tensor filled with zeros prior to calling backward.
                # Also save shape, device and type info to ctx for materializing tensor in backward if output grad is None.
                ctx.set_materialize_grads(False)

                return user_outputs

            @staticmethod
            def backward(ctx, *grad_outputs):
                '''Performs backward pass based on grad wrt module output
                '''
                assert ctx.run_info is not None, 'forward() or __call__() methods must be called before backward()'

                # Assert that the grad_outputs and model device match
                _ortmodule_utils._check_same_device(
                    self._device, "Input argument to backward", *grad_outputs)

                # Use IO binding
                # Push user output grads to ONNX backend.
                contiguous_grad_outputs = []
                for idx, grad_output in enumerate(grad_outputs):
                    if idx in self._graph_info.output_grad_indices_non_differentiable:
                        assert grad_output is None, "ORT found the {}-th module output '{}' is non-differentiable according to the onnx graph. " \
                                                    "However, the gradient value is still provided by torch's autograd engine." \
                                                    .format(idx, self._graph_info.user_output_names[idx]) 
                        continue
                    
                    if grad_output is None:
                        shape, device, dtype = ctx.run_info.output_info[idx]
                        if idx in self._graph_info.output_grad_indices_require_full_shape:
                            grad_output = torch.zeros(
                                shape, device=device, dtype=dtype)
                        else:
                            grad_output = torch.tensor(
                                0., device=device, dtype=dtype)
                    elif not grad_output.is_contiguous():
                        grad_output = grad_output.contiguous()
                    contiguous_grad_outputs.append(grad_output)
                backward_grad_output_ortvalue = [_ortmodule_utils._ortvalue_from_torch_tensor(
                    grad_output) for grad_output in contiguous_grad_outputs]

                # Run and get results
                run_id = ctx.run_info.run_id
                training_io_binding = ctx.run_info.io_binding
                self._execution_agent.run_backward(backward_grad_output_ortvalue, run_id)
                backward_outputs = training_io_binding.get_outputs()

                # Return input and initializer gradients
                num_user_input_grads = len(self._input_names_require_grad)

                results = []
                for input_name in self._graph_info.user_input_names:
                    try:
                        # Append to the results the backward output for each input that required grad
                        results.append(_ortmodule_utils._ortvalue_to_torch_tensor(
                            backward_outputs[self._input_names_require_grad.index(input_name)]))
                    except ValueError:
                        # input_name is not found in the self._input_names_require_grad list
                        # Append None to results for each input that did not require grad
                        results.append(None)

                # Append gradients of initializer to results
                # Go over each initializer, check if it required grad and append to results accordingly
                initializer_names_to_train_set = set(self._graph_info.initializer_names_to_train)
                initializer_index = num_user_input_grads
                for initializer_name in self._graph_info.initializer_names:
                    if initializer_name in initializer_names_to_train_set:
                        results.append(_ortmodule_utils._ortvalue_to_torch_tensor(backward_outputs[initializer_index]))
                        initializer_index += 1
                    else:
                        results.append(None)

                # The OrtValue has a shared_ptr to the data.
                # At this point there are two shared_ptrs to the data, one through the
                # OrtValue in the output iobinding, and the other through the copy in OrtDLManagedTensor.
                # The following call clears the iobinding output, reducing the use_count to 1, so that once torch finishes computation
                # on the DLpack tensors, the memory can be freed.
                training_io_binding.clear_binding_outputs()
                return tuple(results)

        return _ortmodule_io.populate_user_output_from_schema_and_outputs(
            self._module_output_schema,
            self._graph_info.user_output_names,
            _ORTModuleFunction.apply(*self._convert_training_graph_input_to_list(*inputs, **kwargs)))

    def _build_graph(self):
        """Build an optimized gradient graph using the module_graph_builder"""

        super()._build_graph()

        if self._save_onnx:
            onnx.save(self._optimized_onnx_model,
                      self._save_onnx_prefix + '_training.onnx')

            inference_optimized_model = onnx.load_model_from_string(
                self._graph_builder.get_inference_optimized_model())

            onnx.save(inference_optimized_model,
                      self._save_onnx_prefix + '_inference_optimized.onnx')

    def _get_exported_model_and_init_graph_builder(self, *inputs, **kwargs):
        """Export the pytorch model in training mode

        Returns True if model was exported and False if it has already been previously exported
        """

        did_export = self._export_model(*inputs, **kwargs)

        if did_export:
            # If model was exported, then initialize the graph builder
            self._initialize_graph_builder(training=True)

            # Save the onnx model if the model was exported
            if self._save_onnx:
                onnx.save(self._onnx_model,
                        self._save_onnx_prefix + '_exported_training_model.onnx')

        return did_export

    def _create_execution_agent(self):
        """Creates a TrainingAgent that can run the forward and backward graph on the training model"""

        session_options, providers, provider_options = self._get_session_config()
        self._execution_agent = onnxruntime.training.TrainingAgent(self._optimized_onnx_model.SerializeToString(),
                                                                   session_options, providers, provider_options)

    def _reinitialize_graph_builder(self, input_names_require_grad):
        """Return true if the module graph builder was reinitialized"""

        initializer_names_to_train_set_user_model = {name for name, param in
            self._flattened_module.named_parameters() if param.requires_grad}
        initializer_names_to_train_set_onnx_graph = set(self._graph_info.initializer_names_to_train) \
            if self._graph_info else None

        # If inputs requiring gradient change from forward to the next, the module_gradient_graph_builder
        # needs to be reinitialized so it can compute the backward output for the new inputs that require_grad
        if input_names_require_grad != self._input_names_require_grad or \
            initializer_names_to_train_set_user_model != initializer_names_to_train_set_onnx_graph:
            self._input_names_require_grad = input_names_require_grad
            self._initialize_graph_builder(training=True)

            return True

        return False
