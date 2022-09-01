# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# building_blocks.py

from abc import ABC, abstractmethod

import onnx

import onnxruntime.training.onnxblock._graph_utils as graph_utils
import onnxruntime.training.onnxblock.model_accessor as accessor


class Block(ABC):
    """Base class for all building blocks that can be stacked on top of each other."""

    def __init__(self):
        ...

    @abstractmethod
    def build(self, *args, **kwargs):
        """Customize and build this block.

        This method is to be overridden by the user's implementation.
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
    def __init__(self, op_name):
        super().__init__()
        self._op_name = op_name

    def build(self, input_name1, input_name2):
        # get the model to manipulate
        onnx_model = accessor.global_accessor.model

        # Assert that the op name is not empty
        if not self._op_name:
            raise RuntimeError("Unknown op name. Please override _op_name")

        # create the graph node for sub
        node_input_names = [input_name1, input_name2]
        node_output_name = graph_utils.generate_random_graph_name(f"{self._op_name.lower()}_output")
        node_output_names = [node_output_name]
        node = onnx.helper.make_node(
            self._op_name,
            node_input_names,
            node_output_names,
            name=graph_utils.generate_random_graph_name(self._op_name),
        )
        onnx_model.graph.node.append(node)

        return node_output_name


class Add(_BinaryOp):
    """Adds Add node to an onnx model."""

    def __init__(self):
        super().__init__("Add")


class Sub(_BinaryOp):
    """Adds Sub node to an onnx model."""

    def __init__(self):
        super().__init__("Sub")


class Mul(_BinaryOp):
    """Adds Mul node to an onnx model."""

    def __init__(self):
        super().__init__("Mul")


class Div(_BinaryOp):
    """Adds Div node to an onnx model."""

    def __init__(self):
        super().__init__("Div")


class Pow(Block):
    """Adds Pow node to the onnx model."""

    def __init__(self, exponent):
        super().__init__()

        self._exponent = exponent

    def build(self, pow_input_name):
        # get the model to manipulate
        onnx_model = accessor.global_accessor.model

        # create the graph initializer for the exponent
        pow_node_exponent_name = graph_utils.generate_random_graph_name("pow_exponent")
        onnx_model.graph.initializer.append(
            onnx.helper.make_tensor(pow_node_exponent_name, onnx.TensorProto.FLOAT, [1], [self._exponent])
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


class _UnaryOp(Block):
    """Base class for all nodes that take in a single argument."""

    def __init__(self, op_name):
        super().__init__()
        self._op_name = op_name

    def build(self, input_name):
        # get the model to manipulate
        onnx_model = accessor.global_accessor.model

        # Assert that the op name is not empty
        if not self._op_name:
            raise RuntimeError("Unknown op name. Please override _op_name")

        # create the graph node for this unary op
        node_input_names = [input_name]
        node_output_name = graph_utils.generate_random_graph_name(f"{self._op_name.lower()}_output")
        node_output_names = [node_output_name]
        node = onnx.helper.make_node(
            self._op_name,
            node_input_names,
            node_output_names,
            graph_utils.generate_random_graph_name(self._op_name),
        )
        onnx_model.graph.node.append(node)

        return node_output_name


class ReduceMean(_UnaryOp):
    """Adds ReduceMean node to the onnx model."""

    def __init__(self):
        super().__init__("ReduceMean")


class ReduceSum(_UnaryOp):
    """Adds ReduceSum node to the onnx model."""

    def __init__(self):
        super().__init__("ReduceSum")


class Sigmoid(_UnaryOp):
    """Adds Sigmoid node to the onnx model."""

    def __init__(self):
        super().__init__("Sigmoid")


class Log(_UnaryOp):
    """Adds Log node to the onnx model."""

    def __init__(self):
        super().__init__("Log")


class Neg(_UnaryOp):
    """Adds Neg node to the onnx model."""

    def __init__(self):
        super().__init__("Neg")


class Constant(Block):
    """Creates a float initializer and adds it to the onnx model."""

    # TODO: Add ability to add all sorts of initializers (not just floats).

    def __init__(self, value_float):
        super().__init__()

        self._value = value_float

    def build(self):
        # create the graph initializer for the exponent
        initializer_name = graph_utils.generate_random_graph_name("initializer")
        accessor.global_accessor.model.graph.initializer.append(
            onnx.helper.make_tensor(initializer_name, onnx.TensorProto.FLOAT, [1], [self._value])
        )
        return initializer_name


class SequenceConstruct(Block):
    """Adds SequenceConstruct node to the onnx model."""

    def __init__(self):
        super().__init__()

    def build(self, *sequence_input_names):
        # get the model to manipulate
        onnx_model = accessor.global_accessor.model

        # create the graph node for this sequence construct node
        sc_node_input_names = list(sequence_input_names)
        sc_node_output_name = graph_utils.generate_random_graph_name("sequenceconstruct_output")
        sc_node_output_names = [sc_node_output_name]
        sc_node = onnx.helper.make_node(
            "SequenceConstruct",
            sc_node_input_names,
            sc_node_output_names,
            graph_utils.generate_random_graph_name("SequenceConstruct"),
        )
        onnx_model.graph.node.append(sc_node)

        return sc_node_output_name


class ReduceAllL2(Block):
    """Adds ReduceAllL2 node to the onnx model.

    ReduceAllL2 is a part of the com.microsoft domain and might not be accessible outside this domain.
    """

    def __init__(self):
        super().__init__()

    def build(self, *reduce_node_input_names):
        # get the model to manipulate
        onnx_model = accessor.global_accessor.model

        # create the graph node for this reducealll2 node
        reduce_node_input_names = list(reduce_node_input_names)
        reduce_node_output_name = graph_utils.generate_random_graph_name("reducealll2_output")
        reduce_node_output_names = [reduce_node_output_name]
        reduce_node = onnx.helper.make_node(
            "ReduceAllL2",
            reduce_node_input_names,
            reduce_node_output_names,
            graph_utils.generate_random_graph_name("ReduceAllL2"),
            domain="com.microsoft",
        )
        onnx_model.graph.node.append(reduce_node)
        # TODO: register shape inference with onnx
        onnx_model.graph.value_info.append(
            onnx.helper.make_tensor_value_info(reduce_node_output_name, onnx.TensorProto.FLOAT, [1])
        )

        return reduce_node_output_name


class Clip(Block):
    """Adds Clip node to the onnx model."""

    def __init__(self, clip_min=None, clip_max=None):
        super().__init__()

        self._min = clip_min
        self._max = clip_max

    def build(self, clip_input_name):
        # get the model to manipulate
        onnx_model = accessor.global_accessor.model

        # create the graph initializer for the clip min
        clip_node_min_name = ""
        if self._min is not None:
            clip_node_min_name = graph_utils.generate_random_graph_name("clip_min")
            onnx_model.graph.initializer.append(
                onnx.helper.make_tensor(clip_node_min_name, onnx.TensorProto.FLOAT, [1], [self._min])
            )

        # create the graph initializer for the clip max
        clip_node_max_name = ""
        if self._max is not None:
            clip_node_max_name = graph_utils.generate_random_graph_name("clip_max")
            onnx_model.graph.initializer.append(
                onnx.helper.make_tensor(clip_node_max_name, onnx.TensorProto.FLOAT, [1], [self._max])
            )

        # create the graph node for this clip node
        clip_node_input_names = [
            clip_input_name,
            clip_node_min_name,
            clip_node_max_name,
        ]
        clip_node_output_name = graph_utils.generate_random_graph_name("clip_output")
        clip_node_output_names = [clip_node_output_name]
        clip_node = onnx.helper.make_node(
            "Clip",
            clip_node_input_names,
            clip_node_output_names,
            graph_utils.generate_random_graph_name("Clip"),
        )
        onnx_model.graph.node.append(clip_node)

        return clip_node_output_name
