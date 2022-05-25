# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# optim.py

import copy
import onnx

import onnxruntime.training.onnxblock as onnxblock
import onnxruntime.training.onnxblock.model_accessor as accessor


class AdamW(onnxblock.Model):
    """Builds AdamW optimizer onnxblock for the given training model."""

    def __init__(
        self, bias_correction=True, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0
    ):
        super(AdamW, self).__init__()
        self._bias_correction = bias_correction
        self._betas = betas
        self._eps = eps
        self._weight_decay = weight_decay
        self._max_norm_clip = 1.0

    def build(self, parameters):
        """Returns an AdamW optimizer model based on the input training model."""

        # TODO: Avoid hard coded input/output strings
        learning_rate_name = "learning_rate"
        step_name = "step"
        gradient_output_suffix = "_grad.accumulation.out"
        first_order_moment_suffix = "exp_avg"
        second_order_moment_fuffix = "exp_avg_sq"
        output_name_suffix = "out"

        trainable_parameters, _ = parameters

        graph_nodes = []
        graph_inputs = [
            onnx.helper.make_tensor_value_info(
                learning_rate_name, onnx.TensorProto.FLOAT, [1]
            ),
            onnx.helper.make_tensor_value_info(step_name, onnx.TensorProto.INT64, [1]),
        ]
        graph_outputs = []

        # Iterate over all training graph outputs that are gradient outputs
        for idx, param in enumerate(trainable_parameters):

            param_name = param.name
            grad_name = f"{param_name}{gradient_output_suffix}"
            first_order_moment_name = f"{param_name}.{first_order_moment_suffix}"
            second_order_moment_name = f"{param_name}.{second_order_moment_fuffix}"
            # prepare node (and graph) inputs and outputs
            node_input_names = [
                learning_rate_name,  # learning rate
                step_name,  # training step (used for beta correction)
                param_name,  # param to be updated
                grad_name,  # gradient of the param to be used for update
                first_order_moment_name,  # first order moment for this param
                second_order_moment_name,  # second order moment for this param
            ]

            param_tensor_value_info = onnx.helper.make_tensor_value_info(
                param_name, param.data_type, param.dims
            )
            grad_tensor_value_info = onnx.helper.make_tensor_value_info(
                grad_name, param.data_type, param.dims
            )
            first_order_moment_tensor_value_info = onnx.helper.make_tensor_value_info(
                first_order_moment_name, param.data_type, param.dims
            )
            second_order_moment_tensor_value_info = onnx.helper.make_tensor_value_info(
                second_order_moment_name, param.data_type, param.dims
            )
            node_inputs = [
                param_tensor_value_info,
                grad_tensor_value_info,
                first_order_moment_tensor_value_info,
                second_order_moment_tensor_value_info,
            ]
            graph_inputs.extend(node_inputs)

            step_output_name = f"{param_name}.{step_name}.{output_name_suffix}"
            param_output_name = f"{param_name}.{output_name_suffix}"
            first_order_moment_output_name = (
                f"{first_order_moment_name}.{output_name_suffix}"
            )
            second_order_moment_output_name = (
                f"{second_order_moment_name}.{output_name_suffix}"
            )

            param_output_tensor_value_info = onnx.helper.make_tensor_value_info(
                param_output_name, param.data_type, param.dims
            )
            first_order_moment_output_tensor_value_info = (
                onnx.helper.make_tensor_value_info(
                    first_order_moment_output_name, param.data_type, param.dims
                )
            )
            second_order_moment_output_tensor_value_info = (
                onnx.helper.make_tensor_value_info(
                    second_order_moment_output_name, param.data_type, param.dims
                )
            )

            node_output_names = [
                step_output_name,  # step out
                first_order_moment_output_name,  # first order moment output
                second_order_moment_output_name,  # second order moment output
                param_output_name,  # updated weights
            ]

            node_outputs = [
                onnx.helper.make_tensor_value_info(
                    step_output_name, onnx.TensorProto.INT64, [1]
                ),
                first_order_moment_output_tensor_value_info,
                second_order_moment_output_tensor_value_info,
                param_output_tensor_value_info,
            ]
            graph_outputs.extend(node_outputs)

            # AdamOptimizer node attributes
            node_attributes = {
                "alpha": self._betas[0],  # beta1
                "beta": self._betas[1],  # beta2
                "lambda": self._weight_decay,  # weight decay
                "epsilon": self._eps,  # epsilon
                "do_bias_correction": 1
                if self._bias_correction
                else 0,  # bias_correction
                "weight_decay_mode": 1,  # weight decay mode
                "max_norm_clip": self._max_norm_clip,  # used for gradient scaling
            }

            # make the node
            optimizer_node = onnx.helper.make_node(
                "AdamOptimizer",
                node_input_names,
                node_output_names,
                name=f"AdamOptimizer{idx}",
                domain="com.microsoft",
                **node_attributes,
            )

            graph_nodes.append(optimizer_node)

        # make the graph and the model
        graph = onnx.helper.make_graph(
            graph_nodes, "Optimizer Graph", graph_inputs, graph_outputs
        )
        model = onnx.helper.make_model(
            graph,
            producer_name=onnxblock._producer_name,
            opset_imports=[onnxblock._opset_import],
        )

        accessor.global_accessor.model = model

        return [output.name for output in graph_outputs]
