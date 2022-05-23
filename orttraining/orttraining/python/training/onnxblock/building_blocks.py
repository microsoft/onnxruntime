# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _building_blocks.py

from abc import ABC, abstractmethod
import onnx

import onnxruntime.training.onnxblock.model_accessor as accessor
import onnxruntime.training.onnxblock._graph_utils as graph_utils


class Block(ABC):
    """Base class for all building blocks that can be stacked on top of each other."""

    def __init__(self):
        ...

    @abstractmethod
    def build(self, *args, **kwargs):
        """Customize and build this block.

        This method is to be overriden by the user's implementation.
        """
        ...

    def __call__(self, *args, **kwargs):
        """Calls the user's build method and runs validation on top."""

        # build the user model
        output = self.build(*args, **kwargs)

        # validate and check the model
        onnx.checker.check_model(accessor.global_accessor.model, True)

        return output


class _BinaryOp(Block):
    def __init__(self):
        super(_BinaryOp, self).__init__()

    def _build(self, input_name1, input_name2, op_name):
        # get the model to manipulate
        onnx_model = accessor.global_accessor.model

        # create the graph node for sub
        node_input_names = [input_name1, input_name2]
        node_output_name = graph_utils.generate_random_graph_name(
            f"{op_name.lower()}_output"
        )
        node_output_names = [node_output_name]
        node = onnx.helper.make_node(
            op_name,
            node_input_names,
            node_output_names,
            name=graph_utils.generate_random_graph_name(op_name),
        )
        onnx_model.graph.node.append(node)

        return node_output_name


class Add(_BinaryOp):
    """Adds Add node to an onnx model."""

    def __init__(self):
        super(Add, self).__init__()

    def build(self, add_input_name1, add_input_name2):
        return super(Add, self)._build(add_input_name1, add_input_name2, "Add")


class Sub(_BinaryOp):
    """Adds Sub node to an onnx model."""

    def __init__(self):
        super(Sub, self).__init__()

    def build(self, sub_input_name1, sub_input_name2):
        return super(Sub, self)._build(sub_input_name1, sub_input_name2, "Sub")


class Mul(_BinaryOp):
    """Adds Mul node to an onnx model."""

    def __init__(self):
        super(Mul, self).__init__()

    def build(self, mul_input_name1, mul_input_name2):
        return super(Mul, self)._build(mul_input_name1, mul_input_name2, "Mul")


class Pow(Block):
    """Adds Pow node to the onnx model."""

    def __init__(self, exponent):
        super(Pow, self).__init__()

        self._exponent = exponent

    def build(self, pow_input_name):
        # get the model to manipulate
        onnx_model = accessor.global_accessor.model

        # create the graph initializer for the exponent
        pow_node_exponent_name = graph_utils.generate_random_graph_name("pow_exponent")
        onnx_model.graph.initializer.append(
            onnx.helper.make_tensor(
                pow_node_exponent_name, onnx.TensorProto.FLOAT, [1], [self._exponent]
            )
        )

        # create the graph node for pow
        pow_node_input_names = [pow_input_name, pow_node_exponent_name]
        pow_node_output_name = graph_utils.generate_random_graph_name("pow_output")
        pow_node_output_names = [pow_node_output_name]
        pow_node = onnx.helper.make_node(
            "Pow",
            pow_node_input_names,
            pow_node_output_names,
            name=graph_utils.generate_random_graph_name("Pow"),
        )
        onnx_model.graph.node.append(pow_node)

        return pow_node_output_name


class _Reduce(Block):
    """Base class for the reduce blocks."""

    def __init__(self):
        super(_Reduce, self).__init__()

    def _build(self, reduce_input_name, reduction_op):
        # get the model to manipulate
        onnx_model = accessor.global_accessor.model

        # create the graph node for reduce
        reduce_node_input_names = [reduce_input_name]
        reduce_node_output_name = graph_utils.generate_random_graph_name(
            "reduce_output"
        )
        reduce_node_output_names = [reduce_node_output_name]
        reduce_node = onnx.helper.make_node(
            reduction_op,
            reduce_node_input_names,
            reduce_node_output_names,
            name=graph_utils.generate_random_graph_name(reduction_op),
        )
        onnx_model.graph.node.append(reduce_node)

        return reduce_node_output_name


class ReduceMean(_Reduce):
    """Adds ReduceMean node to the onnx model."""

    def __init__(self):
        super(ReduceMean, self).__init__()

    def build(self, reduce_input_name):
        return super()._build(reduce_input_name, "ReduceMean")


class ReduceSum(_Reduce):
    """Adds ReduceSum node to the onnx model."""

    def __init__(self):
        super(ReduceSum, self).__init__()

    def build(self, reduce_input_name):
        return super(ReduceSum, self)._build(reduce_input_name, "ReduceSum")


class _UnaryOp(Block):
    def __init__(self):
        super(_UnaryOp, self).__init__()

    def _build(self, input_name, op_name):
        # get the model to manipulate
        onnx_model = accessor.global_accessor.model

        # create the graph node for this unary op
        node_input_names = [input_name]
        node_output_name = graph_utils.generate_random_graph_name(
            f"{op_name.lower()}_output"
        )
        node_output_names = [node_output_name]
        node = onnx.helper.make_node(
            op_name,
            node_input_names,
            node_output_names,
            graph_utils.generate_random_graph_name(op_name),
        )
        onnx_model.graph.node.append(node)

        return node_output_name


class Sigmoid(_UnaryOp):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def build(self, sigmoid_input_name):
        return super(Sigmoid, self)._build(sigmoid_input_name, "Sigmoid")


class Log(_UnaryOp):
    def __init__(self):
        super(Log, self).__init__()

    def build(self, log_input_name):
        return super(Log, self)._build(log_input_name, "Log")


class Neg(_UnaryOp):
    def __init__(self):
        super(Neg, self).__init__()

    def build(self, neg_input_name):
        return super(Neg, self)._build(neg_input_name, "Neg")
