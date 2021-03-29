# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from . import _utils, _ortmodule_utils, _ortmodule_output_transformation as _ortmodule_io
from ._ortmodule_graph_execution_manager import GraphExecutionManager, _run_forward

import onnx
import onnxruntime

import torch

class InferenceManager(GraphExecutionManager):
    """Concrete instance of GraphExecutionManager that is able to manage the inference model

    InferenceManager is resposible for building and running the forward graph of the inference model
    """
    def __init__(self, model):
        super().__init__(model)

        self._export_mode = torch.onnx.TrainingMode.EVAL

    def forward(self, *inputs, **kwargs):
        '''Forward pass of the inference model

        ONNX model is exported the first time this method is executed.
        Next, we build an optimized inference graph with module_graph_builder.
        Finally, we instantiate the ONNX Runtime InferenceSession through the InferenceAgent.
        '''

        # Exporting module to ONNX for the first time
        # Flag to indicate whether the graph needs to be built
        # It should be built every time the module graph builder is initialized
        build_graph = \
            self._get_exported_model_and_init_graph_builder(
                *inputs, **kwargs)

        _, _, _, new_input_shape = \
            _ortmodule_io.parse_inputs_for_onnx_export(
                self._module_parameters, self._onnx_model, *inputs, **kwargs)

        # Build the inference graph
        if build_graph:
            self._current_input_shape = new_input_shape
            self._build_graph()

        module_device = _utils.get_device_from_module(self._original_module)
        # The inference session should be created every time
        # the graph was built or if the device changed between calls to forward
        create_execution_session = build_graph or self._device != module_device
        if self._device != module_device:
            self._device = module_device

        if create_execution_session:
            # Create execution session creates the inference_session
            self._create_execution_agent()

        user_outputs, _ = _run_forward(
            self._execution_agent, self._optimized_onnx_model, self._device,
            *self._convert_training_graph_input_to_list(*inputs, **kwargs))

        return _ortmodule_io.populate_user_output_from_schema_and_outputs(
            self._module_output_schema,
            self._graph_info.user_output_names,
            user_outputs)

    def _build_graph(self):
        """Build an optimized inference graph using the module_graph_builder"""

        super()._build_graph()

        if self._save_onnx:
            onnx.save(self._optimized_onnx_model,
                      self._save_onnx_prefix + '_inference.onnx')

    def _get_exported_model_and_init_graph_builder(self, *inputs, **kwargs):
        """Export the pytorch model in inference mode

        Returns True if model was exported and False if it has already been previously exported
        """

        did_export = self._export_model(*inputs, **kwargs)

        if did_export:
            # If model was exported, then initialize the graph builder
            self._initialize_graph_builder(training=False)

            # Save the onnx model if the model was exported
            if self._save_onnx:
                onnx.save(self._onnx_model,
                        self._save_onnx_prefix + '_exported_inference_model.onnx')

        return did_export

    def _create_execution_agent(self):
        """Creates an InferenceAgent that can run forward graph on an inference model"""

        session_options, providers, provider_options = self._get_session_config()
        self._execution_agent = onnxruntime.training.InferenceAgent(
            self._optimized_onnx_model.SerializeToString(),
            session_options, providers, provider_options)
