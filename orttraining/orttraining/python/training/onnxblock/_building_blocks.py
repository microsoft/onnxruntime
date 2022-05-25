# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _building_blocks.py

import copy
import onnx

import onnxruntime.training.onnxblock as onnxblock
import onnxruntime.training.onnxblock.model_accessor as accessor
import onnxruntime.training.onnxblock._graph_utils as graph_utils


class Sub(onnxblock.Model):
    """Adds Sub node to an onnx model."""

    def __init__(self):
        super(Sub, self).__init__()

    def build(self, sub_input_name1, sub_input_name2):
        # get the model to manipulate
        onnx_model = accessor.global_accessor.model

        # create the graph node for sub
        sub_node_input_names = [sub_input_name1, sub_input_name2]
        sub_node_output_name = graph_utils.generate_random_graph_name("sub_output")
        sub_node_output_names = [sub_node_output_name]
        sub_node = onnx.helper.make_node(
            "Sub",
            sub_node_input_names,
            sub_node_output_names,
            name=graph_utils.generate_random_graph_name("Sub"),
        )
        onnx_model.graph.node.append(sub_node)

        # create the graph output for sub
        graph_output = copy.deepcopy(
            graph_utils.get_output_from_output_name(onnx_model, sub_input_name1)
        )
        graph_output.name = sub_node_output_name
        del onnx_model.graph.output[:]
        onnx_model.graph.output.append(graph_output)

        return sub_node_output_name


class Pow(onnxblock.Model):
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

        # create the graph output for pow
        graph_output = copy.deepcopy(
            graph_utils.get_output_from_output_name(onnx_model, pow_input_name)
        )
        graph_output.name = pow_node_output_name
        del onnx_model.graph.output[:]
        onnx_model.graph.output.append(graph_output)

        return pow_node_output_name


class _Reduce(onnxblock.Model):
    """Base class for the reduce blocks."""

    def __init__(self):
        super(_Reduce, self).__init__()

    def _reduce(self, reduce_input_name, reduction_op):
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

        # create the graph output for reduce
        reduce_input = copy.deepcopy(
            graph_utils.get_output_from_output_name(onnx_model, reduce_input_name)
        )
        output_rank = len(reduce_input.type.tensor_type.shape.dim)
        graph_outputs = [
            onnx.helper.make_tensor_value_info(
                reduce_node_output_name, onnx.TensorProto.FLOAT, [1] * output_rank
            )
        ]
        del onnx_model.graph.output[:]
        onnx_model.graph.output.extend(graph_outputs)

        return reduce_node_output_name


class ReduceMean(_Reduce):
    """Adds ReduceMean node to the onnx model."""

    def __init__(self):
        super(ReduceMean, self).__init__()

    def build(self, reduce_input_name):
        return super()._reduce(reduce_input_name, "ReduceMean")


class ReduceSum(_Reduce):
    """Adds ReduceSum node to the onnx model."""

    def __init__(self):
        super(ReduceSum, self).__init__()

    def build(self, reduce_input_name):
        return super(ReduceSum, self)._reduce(reduce_input_name, "ReduceSum")
