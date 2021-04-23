# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from . import _ortmodule_utils as _utils, _ortmodule_io as _io
from ._ortmodule_graph_execution_manager import GraphExecutionManager, _run_forward

import copy
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
        build_graph = self._export_model(*inputs, **kwargs)
        if build_graph:
            # If model was exported, then initialize the graph builder
            self._initialize_graph_builder(training=False)

            # Save the onnx model if the model was exported
            if self._save_onnx:
                onnx.save(self._onnx_model, self._save_onnx_prefix + '_exported_inference_model.onnx')

        # Build the inference graph
        if build_graph:
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

        user_outputs, _ = _run_forward(self._execution_agent,
                                       self._optimized_onnx_model,
                                       self._device,
                                       self._skip_duplicate_checks,
                                       *_io._convert_input_to_list(self._flattened_module.named_parameters(),
                                                                   self._graph_info.user_input_names,
                                                                   self._flattened_module.named_buffers(),
                                                                   inputs,
                                                                   kwargs))

        return _io.populate_user_output_from_schema_and_outputs(self._module_output_schema,
                                                                self._graph_info.user_output_names,
                                                                user_outputs)

    def _build_graph(self):
        """Build an optimized inference graph using the module_graph_builder"""

        super()._build_graph()
        if self._save_onnx:
            onnx.save(self._optimized_onnx_model, self._save_onnx_prefix + '_inference.onnx')

    def _create_execution_agent(self):
        """Creates an InferenceAgent that can run forward graph on an inference model"""

        session_options, providers, provider_options = self._get_session_config()
        self._execution_agent = onnxruntime.training.InferenceAgent(self._optimized_onnx_model.SerializeToString(),
                                                                    session_options, providers, provider_options)
