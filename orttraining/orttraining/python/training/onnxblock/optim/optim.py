# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# optim.py

import copy
import onnx

from ..graph import Graph
from onnxruntime.training import onnxblock


class AdamW(Graph):
    """Builds AdamW optimizer onnxblock for the given training model."""

    def __init__(
        self, bias_correction=True, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0
    ):
        super(AdamW, self).__init__()
        self.bias_correction = bias_correction
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_norm_clip = 1.0

    def build(self, base_model):
        """Returns an AdamW optimizer model based on the input training model."""

        learning_rate_name = "learning_rate"
        step_name = "step"
        gradient_output_suffix = "_grad.accumulation.out"
        first_order_moment_suffix = "exp_avg"
        second_order_moment_fuffix = "exp_avg_sq"
        output_name_suffix = "out"

        graph_nodes = []
        graph_inputs = [
            onnx.helper.make_tensor_value_info(
                learning_rate_name, onnx.TensorProto.FLOAT, [1]
            ),
            onnx.helper.make_tensor_value_info(step_name, onnx.TensorProto.INT64, [1]),
        ]
        graph_outputs = []

        # Iterate over all training graph outputs that are gradient outputs
        for idx, graph_output in enumerate(base_model.graph.output):
            if not graph_output.name.endswith(gradient_output_suffix):
                continue

            weight_name = graph_output.name[: -len(gradient_output_suffix)]
            grad_name = graph_output.name
            first_order_moment_name = f"{weight_name}.{first_order_moment_suffix}"
            second_order_moment_name = f"{weight_name}.{second_order_moment_fuffix}"
            # prepare node (and graph) inputs and outputs
            node_input_names = [
                learning_rate_name,  # learning rate
                step_name,  # training step (used for beta correction)
                weight_name,  # weight to be updated
                grad_name,  # gradient of the weight to be used for update
                first_order_moment_name,  # first order moment for this weight
                second_order_moment_name,  # second order moment for this weight
            ]

            weight_tensor_value_info = copy.deepcopy(graph_output)
            weight_tensor_value_info.name = weight_name
            first_order_moment_tensor_value_info = copy.deepcopy(graph_output)
            first_order_moment_tensor_value_info.name = first_order_moment_name
            second_order_moment_tensor_value_info = copy.deepcopy(graph_output)
            second_order_moment_tensor_value_info.name = second_order_moment_name
            node_inputs = [
                weight_tensor_value_info,
                copy.deepcopy(graph_output),
                first_order_moment_tensor_value_info,
                second_order_moment_tensor_value_info,
            ]
            graph_inputs.extend(node_inputs)

            step_output_name = f"{weight_name}.{step_name}.{output_name_suffix}"
            first_order_moment_output_name = (
                f"{first_order_moment_name}.{output_name_suffix}"
            )
            second_order_moment_output_name = (
                f"{second_order_moment_name}.{output_name_suffix}"
            )
            weight_output_name = f"{weight_name}.{output_name_suffix}"

            first_order_moment_output_tensor_value_info = copy.deepcopy(graph_output)
            first_order_moment_output_tensor_value_info.name = (
                first_order_moment_output_name
            )
            second_order_moment_output_tensor_value_info = copy.deepcopy(graph_output)
            second_order_moment_output_tensor_value_info.name = (
                second_order_moment_output_name
            )
            weight_output_tensor_value_info = copy.deepcopy(graph_output)
            weight_output_tensor_value_info.name = weight_output_name

            node_output_names = [
                step_output_name,  # step out
                first_order_moment_output_name,  # first order moment output
                second_order_moment_output_name,  # second order moment output
                weight_output_name,  # updated weights
            ]

            node_outputs = [
                onnx.helper.make_tensor_value_info(
                    step_output_name, onnx.TensorProto.INT64, [1]
                ),
                first_order_moment_output_tensor_value_info,
                second_order_moment_output_tensor_value_info,
                weight_output_tensor_value_info,
            ]
            graph_outputs.extend(node_outputs)

            # AdamOptimizer node attributes
            node_attributes = {
                "alpha": self.betas[0],  # beta1
                "beta": self.betas[1],  # beta2
                "lambda": self.weight_decay,  # weight decay
                "epsilon": self.eps,  # epsilon
                "do_bias_correction": 1
                if self.bias_correction
                else 0,  # bias_correction
                "weight_decay_mode": 1,  # weight decay mode
                "max_norm_clip": self.max_norm_clip,  # used for gradient scaling
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
            opset_imports=[onnx.helper.make_opsetid("com.microsoft", 1)],
        )
        return model
