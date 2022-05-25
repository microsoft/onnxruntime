# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# loss.py

import copy
import onnx

import onnxruntime.training.onnxblock as onnxblock
import onnxruntime.training.onnxblock.model_accessor as accessor
import onnxruntime.training.onnxblock._graph_utils as graph_utils
import onnxruntime.training.onnxblock._building_blocks as building_blocks


class MSELoss(onnxblock.Model):
    """MSELoss onnxblock for adding MSE loss to an onnx model.

    Parameters:
        reduction: string representing the reduction method on the loss output.
                   can be one of "mean" or "sum"
    """

    def __init__(self, reduction="mean"):
        super(MSELoss, self).__init__()

        # determine the reduction type
        if reduction != "mean" and reduction != "sum":
            raise RuntimeError(f"Reduction {reduction} not supported.")

        self._reduce = (
            building_blocks.ReduceMean()
            if reduction == "mean"
            else building_blocks.ReduceSum()
        )
        self._sub = building_blocks.Sub()
        self._square = building_blocks.Pow(2.0)

    def build(self, loss_input_name, target_name="target"):
        """Adds an MSELoss subgraph on top of the base_model.

        Creates a block that measures the mean squared error between
        loss_input_name and the target_name.

        Args:
            loss_input_name: string input representing the loss input
            target_name: string input representing the target

        Returns:
            Returns a string of the output name from the loss
        """

        # get the model to manipulate
        onnx_model = accessor.global_accessor.model

        # create a new graph input. this is the target input needed to compare
        # the graph output against to calculate loss.
        target_input = copy.deepcopy(
            graph_utils.get_output_from_output_name(onnx_model, loss_input_name)
        )
        target_input.name = target_name
        onnx_model.graph.input.append(target_input)

        # create the mse loss
        # loss = reduce(square(sub(output, target)))
        return self._reduce(self._square(self._sub(loss_input_name, target_name)))


class CrossEntropyLoss(onnxblock.Model):
    """CrossEntropyLoss onnxblock for adding Cross Entropy loss to an onnx model.

    Parameters:
        weight: boolean representing whether a manual rescaling weight given to
                each class should be added to the inputs.
        reduction: string representing the reduction method on the loss output.
                   can be one of "mean" or "sum"
        ignore_index: specifies a target value that is ignored and does not
                      contribute to the input gradient.
    """

    def __init__(self, weight=False, reduction="mean", ignore_index=None):
        super(CrossEntropyLoss, self).__init__()

        # determine the reduction type
        if reduction != "mean" and reduction != "sum":
            raise RuntimeError(f"Reduction {reduction} not supported.")

        self._weight = weight
        self._reduction = reduction
        self._ignore_index = ignore_index

    def build(self, scores_input_name, labels_name="labels", weight_name="loss_weight"):
        """Adds a CrossEntropyLoss subgraph on top of an onnx model.

        Creates a block that measures the softmax cross entropy between
        scores_input_name and the labels_name.

        Args:
            loss_input_name: string input representing the loss input
            labels_name: string input representing the labels

        Returns:
            Returns a string of the output name from the loss
        """

        # get the model to manipulate
        onnx_model = accessor.global_accessor.model

        # create a new graph input. this is the labels input needed to compare
        # the graph output against to calculate loss.
        labels_input = copy.deepcopy(
            graph_utils.get_output_from_output_name(onnx_model, scores_input_name)
        )
        labels_input.name = labels_name
        labels_input.type.tensor_type.elem_type = onnx.TensorProto.INT32
        # if the predictions are (num_examples x num_classes)
        # labels should be (num_examples x 1)
        del labels_input.type.tensor_type.shape.dim[1]
        onnx_model.graph.input.append(labels_input)

        if self._weight:
            weight_input = copy.deepcopy(
                graph_utils.get_output_from_output_name(onnx_model, scores_input_name)
            )
            weight_input.name = weight_name
            dim_to_keep = weight_input.type.tensor_type.shape.dim[1]
            del weight_input.type.tensor_type.shape.dim[:]
            weight_input.type.tensor_type.shape.dim.append(dim_to_keep)
            onnx_model.graph.input.append(weight_input)

        # create a new graph node for the loss
        loss_node_input_names = [scores_input_name, labels_name]
        if self._weight:
            loss_node_input_names.append(weight_name)
        loss_node_output_name = graph_utils.generate_random_graph_name("loss")
        loss_node_output_names = [
            loss_node_output_name,
            graph_utils.generate_random_graph_name("log_prob"),
        ]
        loss_node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            loss_node_input_names,
            loss_node_output_names,
            reduction=self._reduction,
            ignore_index=self._ignore_index,
            name=graph_utils.generate_random_graph_name("SoftmaxCrossEntropyLoss"),
        )
        onnx_model.graph.node.append(loss_node)

        # create a new graph output for the loss
        graph_outputs = [
            onnx.helper.make_tensor_value_info(
                loss_node_output_name, onnx.TensorProto.FLOAT, []
            )
        ]
        del onnx_model.graph.output[:]
        onnx_model.graph.output.extend(graph_outputs)

        return loss_node_output_name
