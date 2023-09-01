# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
from typing import Optional

import onnx

import onnxruntime.training.onnxblock._graph_utils as _graph_utils
import onnxruntime.training.onnxblock.blocks as blocks


class MSELoss(blocks.Block):
    """MSELoss onnxblock for adding MSE loss to an onnx model.

    MSE loss is calculated as:
    loss = reduce((output - target)**2)

    Args:
        reduction (str): string representing the reduction method on the loss output.
                         can be one of "mean", "sum", or "none"
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()

        if reduction not in ["mean", "sum", "none"]:
            raise RuntimeError(f"Reduction {reduction} not supported.")

        reduction_blocks = {"mean": blocks.ReduceMean, "sum": blocks.ReduceSum, "none": blocks.PassThrough}

        self._reduce = reduction_blocks[reduction]()
        self._sub = blocks.Sub()
        self._square = blocks.Pow(2.0)

    def build(self, loss_input_name: str, target_name: str = "target"):
        """Adds an MSELoss subgraph on top of the base_model.

        Args:
            loss_input_name (str): input representing the loss input
            target_name (str): input representing the target

        Returns:
            Returns a string of the output name from the loss
        """

        if not _graph_utils.node_arg_exists(self.base, target_name):
            target_name = blocks.InputLike(loss_input_name)(target_name)

        return self._reduce(self._square(self._sub(loss_input_name, target_name)))


class CrossEntropyLoss(blocks.Block):
    """CrossEntropyLoss onnxblock for adding Cross Entropy loss to an onnx model.

    Args:
        weight: numpy ndarray representing a manual rescaling weight given to
                each class. If not provided, rescaling will not be applied.
        reduction: string representing the reduction method on the loss output.
                   can be one of "mean" or "sum"
        ignore_index: specifies a target value that is ignored and does not
                      contribute to the input gradient.
    """

    def __init__(self, weight=None, reduction: str = "mean", ignore_index: Optional[int] = None):
        super().__init__()

        if reduction not in ["mean", "sum", "none"]:
            raise RuntimeError(f"Reduction {reduction} not supported.")

        self._weight = weight
        self._reduction = reduction
        self._ignore_index = ignore_index

    def build(self, scores_input_name: str, labels_name: str = "labels"):
        """Adds a CrossEntropyLoss subgraph on top of an onnx model.

        Args:
            loss_input_name (str): input representing the loss input
            labels_name (str): input representing the labels

        Returns:
            Returns a string of the output name from the loss
        """

        weight_name = _graph_utils.generate_graph_name("celoss.weight")
        if self._weight is not None:
            self.base.graph.initializer.append(onnx.numpy_helper.from_array(self._weight, weight_name))

        if not _graph_utils.node_arg_exists(self.base, labels_name):
            # Create a new graph input. This is the labels input needed to compare
            # the graph output against to calculate loss.
            labels_input = copy.deepcopy(_graph_utils.get_output_from_output_name(self.base, scores_input_name))
            labels_input.name = labels_name
            labels_input.type.tensor_type.elem_type = onnx.TensorProto.INT64
            # If the predictions are (num_examples x num_classes)
            # labels should be (num_examples,)
            del labels_input.type.tensor_type.shape.dim[1]
            self.base.graph.input.append(labels_input)

        loss_node_input_names = [scores_input_name, labels_name]
        if self._weight:
            loss_node_input_names.append(weight_name)
        loss_node_output_name = _graph_utils.generate_graph_name("loss")
        loss_node_output_names = [
            loss_node_output_name,
            _graph_utils.generate_graph_name("log_prob"),
        ]
        loss_node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            loss_node_input_names,
            loss_node_output_names,
            reduction=self._reduction,
            ignore_index=self._ignore_index,
            name=_graph_utils.generate_graph_name("SoftmaxCrossEntropyLoss"),
        )
        self.base.graph.node.append(loss_node)

        return loss_node_output_name


class BCEWithLogitsLoss(blocks.Block):
    """BCEWithLogitsLoss onnxblock for adding binary cross entropy loss to an onnx model.

    Args:
        weight: numpy ndarray representing a manual rescaling weight given to
                each batch. If not provided, rescaling will not be applied.
        reduction: string representing the reduction method on the loss output.
                   can be one of "mean" or "sum"
        pos_weight: numpy ndarray representing the weight of positive examples.
    """

    def __init__(self, weight=None, reduction: str = "mean", pos_weight=None):
        super().__init__()

        if reduction not in ["mean", "sum", "none"]:
            raise RuntimeError(f"Reduction {reduction} not supported.")

        reduction_blocks = {"mean": blocks.ReduceMean, "sum": blocks.ReduceSum, "none": blocks.PassThrough}

        self._weight = weight
        self._reduce = reduction_blocks[reduction]()
        self._pos_weight = pos_weight

        self._sigmoid = blocks.Sigmoid()
        self._log = blocks.Log()
        self._sub = blocks.Sub()
        self._add = blocks.Add()
        self._mul = blocks.Mul()
        self._neg = blocks.Neg()

    def build(self, loss_input_name: str, target_name: str = "target"):
        """Adds a BCEWithLogitsLoss subgraph on top of an onnx model.

        Creates a block that measures the binary cross entropy with logits between
        loss_input_name and the target_name. This block combines Sigmoid layer
        followed by a BCELoss.

        Args:
            loss_input_name (str): input representing the loss input
            target_name (str): input representing the target

        Returns:
            Returns a string of the output name from the loss
        """

        # create the graph initializers for pos_weight, weight, and the sub operands ([1])
        pos_weight_name = _graph_utils.generate_graph_name("bceloss.pos_weight")
        if self._pos_weight is not None:
            self.base.graph.initializer.append(onnx.numpy_helper.from_array(self._pos_weight, pos_weight_name))

        weight_name = _graph_utils.generate_graph_name("bceloss.weight")
        if self._weight is not None:
            self.base.graph.initializer.append(onnx.numpy_helper.from_array(self._weight, weight_name))

        sub_ones_operand_name1 = _graph_utils.generate_graph_name("bceloss.sub_ones")
        self.base.graph.initializer.append(
            onnx.helper.make_tensor(sub_ones_operand_name1, onnx.TensorProto.FLOAT, [1], [1.0])
        )
        sub_ones_operand_name2 = _graph_utils.generate_graph_name("bceloss.sub_ones")
        self.base.graph.initializer.append(
            onnx.helper.make_tensor(sub_ones_operand_name2, onnx.TensorProto.FLOAT, [1], [1.0])
        )

        if not _graph_utils.node_arg_exists(self.base, target_name):
            # Create a new graph input. This is the target input needed to compare
            # the graph output against to calculate loss.
            target_input = copy.deepcopy(_graph_utils.get_output_from_output_name(self.base, loss_input_name))
            target_input.name = target_name
            self.base.graph.input.append(target_input)

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


class L1Loss(blocks.Block):
    """L1Loss onnxblock for adding MSE loss to an onnx model.

    L1Loss is computed as:
    loss = reduce(abs(input - target))

    Args:
        reduction: string representing the reduction method on the loss output.
                   can be one of "mean" or "sum"
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()

        if reduction not in ["mean", "sum", "none"]:
            raise RuntimeError(f"Reduction {reduction} not supported.")

        reduction_blocks = {"mean": blocks.ReduceMean, "sum": blocks.ReduceSum, "none": blocks.PassThrough}
        self._reduce = reduction_blocks[reduction]()
        self._abs = blocks.Abs()
        self._sub = blocks.Sub()

    def build(self, loss_input_name: str, target_name: Optional[str] = "target"):
        """Adds an L1 loss subgraph on top of the base_model.

        Args:
            loss_input_name (str): input representing the loss input
            target_name (str): input representing the target

        Returns:
            Returns a string of the output name from the loss
        """

        if not _graph_utils.node_arg_exists(self.base, target_name):
            target_name = blocks.InputLike(loss_input_name)(target_name)

        return self._reduce(self._abs(self._sub(loss_input_name, target_name)))
