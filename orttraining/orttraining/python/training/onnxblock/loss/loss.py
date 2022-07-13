# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# loss.py

import copy
import typing

import onnx

import onnxruntime.training.onnxblock._graph_utils as graph_utils
import onnxruntime.training.onnxblock.building_blocks as building_blocks
import onnxruntime.training.onnxblock.model_accessor as accessor


class MSELoss(building_blocks.Block):
    """MSELoss onnxblock for adding MSE loss to an onnx model.

    Parameters:
        reduction: string representing the reduction method on the loss output.
                   can be one of "mean" or "sum"
    """

    def __init__(self, reduction: typing.Optional[str] = "mean"):
        super().__init__()

        # determine the reduction type
        if reduction != "mean" and reduction != "sum":
            raise RuntimeError(f"Reduction {reduction} not supported.")

        self._reduce = building_blocks.ReduceMean() if reduction == "mean" else building_blocks.ReduceSum()
        self._sub = building_blocks.Sub()
        self._square = building_blocks.Pow(2.0)

    def build(self, loss_input_name: str, target_name: typing.Optional[str] = "target"):
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
        # TODO: Move input creation outside of the blocks.
        target_input = copy.deepcopy(graph_utils.get_output_from_output_name(onnx_model, loss_input_name))
        target_input.name = target_name
        onnx_model.graph.input.append(target_input)

        # create the mse loss
        # loss = reduce(square(sub(output, target)))
        return self._reduce(self._square(self._sub(loss_input_name, target_name)))


class CrossEntropyLoss(building_blocks.Block):
    """CrossEntropyLoss onnxblock for adding Cross Entropy loss to an onnx model.

    Parameters:
        weight: numpy ndarray representing a manual rescaling weight given to
                each class. If not provided, rescaling will not be applied.
        reduction: string representing the reduction method on the loss output.
                   can be one of "mean" or "sum"
        ignore_index: specifies a target value that is ignored and does not
                      contribute to the input gradient.
    """

    def __init__(
        self, weight=None, reduction: typing.Optional[str] = "mean", ignore_index: typing.Optional[int] = None
    ):
        super().__init__()

        # determine the reduction type
        if reduction != "mean" and reduction != "sum":
            raise RuntimeError(f"Reduction {reduction} not supported.")

        self._weight = weight
        self._reduction = reduction
        self._ignore_index = ignore_index

    def build(self, scores_input_name: str, labels_name: str = "labels"):
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

        weight_name = graph_utils.generate_random_graph_name("celoss.weight")
        if self._weight is not None:
            onnx_model.graph.initializer.append(onnx.numpy_helper.from_array(self._weight, weight_name))

        # create a new graph input. this is the labels input needed to compare
        # the graph output against to calculate loss.
        labels_input = copy.deepcopy(graph_utils.get_output_from_output_name(onnx_model, scores_input_name))
        labels_input.name = labels_name
        labels_input.type.tensor_type.elem_type = onnx.TensorProto.INT32
        # if the predictions are (num_examples x num_classes)
        # labels should be (num_examples x 1)
        del labels_input.type.tensor_type.shape.dim[1]
        onnx_model.graph.input.append(labels_input)

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

        return loss_node_output_name


class BCEWithLogitsLoss(building_blocks.Block):
    """BCEWithLogitsLoss onnxblock for adding binary cross entropy loss to an onnx model.

    Parameters:
        weight: numpy ndarray representing a manual rescaling weight given to
                each batch. If not provided, rescaling will not be applied.
        reduction: string representing the reduction method on the loss output.
                   can be one of "mean" or "sum"
        pos_weight: numpy ndarray representing the weight of positive examples.
    """

    def __init__(self, weight=None, reduction: typing.Optional[str] = "mean", pos_weight=None):
        super().__init__()

        # determine the reduction type
        if reduction != "mean" and reduction != "sum":
            raise RuntimeError(f"Reduction {reduction} not supported.")

        self._weight = weight
        self._reduce = building_blocks.ReduceMean() if reduction == "mean" else building_blocks.ReduceSum()
        self._pos_weight = pos_weight

        self._sigmoid = building_blocks.Sigmoid()
        self._log = building_blocks.Log()
        self._sub = building_blocks.Sub()
        self._add = building_blocks.Add()
        self._mul = building_blocks.Mul()
        self._neg = building_blocks.Neg()

    def build(self, loss_input_name: str, target_name: typing.Optional[str] = "target"):
        """Adds a BCEWithLogitsLoss subgraph on top of an onnx model.

        Creates a block that measures the binary cross entropy with logits between
        loss_input_name and the target_name. This block combines Sigmoid layer
        followed by a BCELoss.

        Args:
            loss_input_name: string input representing the loss input
            target_name: string input representing the target

        Returns:
            Returns a string of the output name from the loss
        """

        # get the model to manipulate
        onnx_model = accessor.global_accessor.model

        # create the graph initializers for pos_weight, weight, and the sub operands ([1])
        pos_weight_name = graph_utils.generate_random_graph_name("bceloss.pos_weight")
        if self._pos_weight is not None:
            onnx_model.graph.initializer.append(onnx.numpy_helper.from_array(self._pos_weight, pos_weight_name))

        weight_name = graph_utils.generate_random_graph_name("bceloss.weight")
        if self._weight is not None:
            onnx_model.graph.initializer.append(onnx.numpy_helper.from_array(self._weight, weight_name))

        sub_ones_operand_name1 = graph_utils.generate_random_graph_name("bceloss.sub_ones")
        onnx_model.graph.initializer.append(
            onnx.helper.make_tensor(sub_ones_operand_name1, onnx.TensorProto.FLOAT, [1], [1.0])
        )
        sub_ones_operand_name2 = graph_utils.generate_random_graph_name("bceloss.sub_ones")
        onnx_model.graph.initializer.append(
            onnx.helper.make_tensor(sub_ones_operand_name2, onnx.TensorProto.FLOAT, [1], [1.0])
        )

        # create a new graph input. this is the target input needed to compare
        # the graph output against to calculate loss.
        target_input = copy.deepcopy(graph_utils.get_output_from_output_name(onnx_model, loss_input_name))
        target_input.name = target_name
        onnx_model.graph.input.append(target_input)

        # create the bceloss
        sigmoid_output = self._sigmoid(loss_input_name)
        add_operand1 = self._mul(self._log(sigmoid_output), target_name)
        if self._pos_weight is not None:
            add_operand1 = self._mul(add_operand1, pos_weight_name)

        add_operand2 = self._mul(
            self._log(self._sub(sub_ones_operand_name1, sigmoid_output)),
            self._sub(sub_ones_operand_name2, target_name),
        )

        loss_output = self._neg(self._add(add_operand1, add_operand2))

        if self._weight is not None:
            loss_output = self._mul(weight_name, loss_output)

        return self._reduce(loss_output)
