# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# loss.py

import copy
import onnx

from ..graph import Graph
from onnxruntime.training import onnxblock


def _get_output_from_output_name(onnx_model, output_name):
    """Returns the graph output given the output name"""

    # Iterate over the graph outputs looking for output_name
    for output in onnx_model.graph.output:
        if output.name == output_name:
            return output

    raise LookupError("The provided output name {output_name} is not a graph output.")


class MSELoss(Graph):
    """MSELoss onnxblock for adding MSE loss to an onnx model."""

    def __init__(self, reduction="mean"):
        super(MSELoss, self).__init__()

        self._reduction = reduction

    def build(self, base_model, output_name, target_name="target"):
        """Adds an MSELoss subgraph on top of the base_model."""

        # deepcopy the base model so we don't inadvertently change the original
        # model
        onnx_model = copy.deepcopy(base_model)

        # determine the reduction type
        if self._reduction != "mean" and self._reduction != "sum":
            raise RuntimeError(f"Reduction {self.reduction} not supported.")

        graph_nodes = onnx_model.graph.node
        graph_inputs = onnx_model.graph.input

        # create a new graph input. this is the target input needed to compare
        # the graph output against to calculate loss.
        target_input = copy.deepcopy(
            _get_output_from_output_name(onnx_model, output_name)
        )
        target_input.name = target_name
        graph_inputs.append(target_input)

        # create a new graph output for loss
        graph_outputs = [
            onnx.helper.make_tensor_value_info("loss", onnx.TensorProto.FLOAT, [1, 1])
        ]

        graph_initializers = onnx_model.graph.initializer

        # loss equation
        # loss = reduce((output-target)^2)

        # create the sub node
        sub_node_input_names = [output_name, target_name]
        sub_node_output_names = ["loss_sub_output"]
        sub_node = onnx.helper.make_node(
            "Sub", sub_node_input_names, sub_node_output_names, name=f"MSELossSub"
        )
        graph_nodes.append(sub_node)

        # create the square node
        pow_node_input_names = sub_node_output_names
        pow_node_input_names.append("0_pow_exponent")
        pow_node_output_names = ["loss_pow_output"]
        pow_node = onnx.helper.make_node(
            "Pow", pow_node_input_names, pow_node_output_names, name=f"MSELossPow"
        )
        graph_nodes.append(pow_node)
        graph_initializers.append(
            onnx.helper.make_tensor(
                "0_pow_exponent", onnx.TensorProto.FLOAT, [1], [2.0]
            )
        )

        # create the reduce node
        reduce_node_input_names = pow_node_output_names
        reduce_node_output_names = ["loss"]
        reduce_node = onnx.helper.make_node(
            "ReduceMean" if self._reduction == "mean" else "ReduceSum",
            reduce_node_input_names,
            reduce_node_output_names,
            name=f"MSELossReduce",
        )
        graph_nodes.append(reduce_node)

        # generate the graph and model with above inputs, outputs, initializers
        # and nodes
        graph = onnx.helper.make_graph(
            graph_nodes,
            "GraphWithLoss",
            graph_inputs,
            graph_outputs,
            graph_initializers,
        )
        model = onnx.helper.make_model(
            graph,
            producer_name=onnxblock._producer_name,
            opset_imports=[onnx.helper.make_opsetid("com.microsoft", 1)]
            + list(base_model.opset_import),
        )

        return model


class CrossEntropyLoss(Graph):
    """CrossEntropyLoss onnxblock for adding Cross Entropy loss to an onnx model."""

    def __init__(self, weight=False, reduction="mean", ignore_index=None):
        super(CrossEntropyLoss, self).__init__()

        self._weight = weight
        self._reduction = reduction
        self._ignore_index = ignore_index

    def build(
        self, base_model, output_name, target_name="target", weight_name="loss_weight"
    ):
        """Adds a CrossEntropyLoss subgraph on top of the base_model."""

        # deepcopy the base model so we don't inadvertently change the
        # original model
        onnx_model = copy.deepcopy(base_model)

        # determine the reduction type
        if self._reduction != "mean" and self._reduction != "sum":
            raise RuntimeError(f"Reduction {self._reduction} not supported.")

        graph_nodes = onnx_model.graph.node
        graph_inputs = onnx_model.graph.input

        # create a new graph input. this is the target input needed to compare
        # the graph output against to calculate loss.
        target_input = copy.deepcopy(
            _get_output_from_output_name(onnx_model, output_name)
        )
        target_input.name = target_name
        target_input.type.tensor_type.elem_type = onnx.TensorProto.INT32
        # if the predictions are (num_examples x num_classes)
        # labels should be (num_examples x 1)
        del target_input.type.tensor_type.shape.dim[1]
        graph_inputs.append(target_input)

        if self._weight:
            weight_input = copy.deepcopy(
                _get_output_from_output_name(onnx_model, output_name)
            )
            weight_input.name = weight_name
            dim_to_keep = weight_input.type.tensor_type.shape.dim[1]
            del weight_input.type.tensor_type.shape.dim[:]
            weight_input.type.tensor_type.shape.dim.append(dim_to_keep)
            graph_inputs.append(weight_input)

        # create a new graph output for loss
        graph_outputs = [
            onnx.helper.make_tensor_value_info("loss", onnx.TensorProto.FLOAT, [])
        ]
        graph_initializers = onnx_model.graph.initializer

        # create the loss node
        loss_node_input_name = [output_name, target_name]
        if self._weight:
            loss_node_input_name.append(weight_name)
        loss_node_output_name = ["loss", "log_prob"]
        loss_node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            loss_node_input_name,
            loss_node_output_name,
            reduction=self._reduction,
            ignore_index=self._ignore_index,
            name=f"SoftmaxCrossEntropyLoss",
        )
        graph_nodes.append(loss_node)

        # generate the graph and model with above inputs, outputs,
        # initializers and nodes
        graph = onnx.helper.make_graph(
            graph_nodes,
            "GraphWithLoss",
            graph_inputs,
            graph_outputs,
            graph_initializers,
        )
        model = onnx.helper.make_model(
            graph,
            producer_name=onnxblock._producer_name,
            opset_imports=[onnx.helper.make_opsetid("com.microsoft", 1)]
            + list(base_model.opset_import),
        )

        return model


# TODO: BCEWithLogitsLoss
