# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import onnx
import torch

from . import _utils, _io
from ._graph_execution_manager import GraphExecutionManager
from ._execution_agent import TrainingAgent
from ._ort_module_function_factory import ort_module_function_factory


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
        build_gradient_graph = self._export_model(*inputs, **kwargs)
        if build_gradient_graph:
            # If model was exported, then initialize the graph builder
            self._initialize_graph_builder(training=True)

        input_info = _io.parse_inputs_for_onnx_export(self._module_parameters,
                                                      self._onnx_model,
                                                      inputs,
                                                      kwargs)

        # Reinitialize graph builder if the inputs or initializers requiring gradient have changed.
        # Order of or operation is important here because we always need to call
        # _reinitialize_graph_builder irrespective of the value of build_gradient_graph.
        build_gradient_graph = self._reinitialize_graph_builder(input_info) or build_gradient_graph

        # Build the gradient graph
        if build_gradient_graph:
            self._build_graph()

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

        # Create a torch.autograd.Function
        _ORTModuleFunction = ort_module_function_factory(self._execution_agent, self._graph_info.user_input_names,
                                                         self._input_info.require_grad_names, self._graph_info.initializer_names,
                                                         self._graph_initializer_names_to_train,
                                                         self._graph_info.module_output_indices_requires_save_for_backward)

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

    def _build_graph(self):
        """Build an optimized gradient graph using the module_graph_builder"""

        super()._build_graph()

        if self._save_onnx:
            onnx.save(self._optimized_onnx_model, self._save_onnx_prefix + '_training.onnx')
            inference_optimized_model = onnx.load_model_from_string(self._graph_builder.get_inference_optimized_model())
            onnx.save(inference_optimized_model, self._save_onnx_prefix + '_inference_optimized.onnx')

    def _create_execution_agent(self):
        """Creates a TrainingAgent that can run the forward and backward graph on the training model"""

        session_options, providers, provider_options = self._get_session_config()
        self._execution_agent = TrainingAgent(self._optimized_onnx_model,
                                              self._device,
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
