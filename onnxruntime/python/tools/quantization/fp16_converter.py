import itertools
import logging

import numpy as np
import onnx
from onnx import GraphProto, NodeProto, TensorProto, numpy_helper

from onnxruntime.quantization import graph_helper
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.onnx_model_converter_base import ConverterBase

logger = logging.getLogger(__name__)


class FP16Converter(ConverterBase):
    def __init__(self, model=None, allow_list=None):
        super().__init__(model, allow_list)
        self.onnx_model = ONNXModel(self.model)

    @staticmethod
    def __cast_initializer_to_fp16(initializer, new_name=None):
        if new_name is None:
            new_name = initializer.name
        if int(initializer.data_type) == TensorProto.FLOAT:
            new_tensor = np.asarray(numpy_helper.to_array(initializer), dtype=np.float16)
            return numpy_helper.from_array(new_tensor, new_name)
        return initializer

    @staticmethod
    def __cast_graph_io(graph: GraphProto, keep_io_types: bool):
        if not keep_io_types:
            for value_info in itertools.chain(graph.input, graph.output, graph.value_info):
                if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
                    value_info.type.tensor_type.elem_type = TensorProto.FLOAT16

    @staticmethod
    def __add_cast_on_node_input(graph: GraphProto, node: NodeProto, input_name: str, to=TensorProto.FLOAT16):
        new_in_tensor = "Cast_" + input_name + "_" + node.name
        cast_node_name = "Cast/" + input_name + "/" + node.name
        cast_node = onnx.helper.make_node("Cast", [input_name], [new_in_tensor], name=cast_node_name, to=to)
        graph_helper.add_node(graph, cast_node)
        ONNXModel.replace_node_input(node, input_name, new_in_tensor)

    @staticmethod
    def __add_cast_on_node_output(graph: GraphProto, node: NodeProto, output_name: str, to=TensorProto.FLOAT16):
        new_out_tensor = "Cast_" + output_name + "_" + node.name
        cast_node_name = "Cast/" + output_name + "/" + node.name
        cast_node = onnx.helper.make_node("Cast", [new_out_tensor], [output_name], name=cast_node_name, to=to)
        graph_helper.add_node(graph, cast_node)
        ONNXModel.replace_node_output(node, output_name, new_out_tensor)

    @staticmethod
    def __cast_initializers(
        graph: GraphProto,
        initializer_name_set: set,
        allowed_nodes: list,
    ):
        for node in allowed_nodes:
            for in_tensor_name in node.input:
                if in_tensor_name in initializer_name_set:
                    initializer = graph_helper.get_initializer(graph, in_tensor_name)
                    if initializer is not None and initializer.data_type == TensorProto.FLOAT:
                        new_in_tensor = initializer.name + "_fp16"
                        new_initializer = FP16Converter.__cast_initializer_to_fp16(initializer, new_in_tensor)
                        # remove the old initializer if it is not used by any other node
                        # In case when initializer is used by multiple nodes, and other node does not support fp16,
                        # we will not remove it
                        if len(ONNXModel.find_nodes_by_initializer(graph, initializer)) == 1:
                            graph_helper.remove_initializer(graph, initializer)
                            initializer_name_set.remove(in_tensor_name)
                        # Add the new initializer if it is not already present
                        # In case when initializer is used by multiple nodes, we will add it only once
                        if len(ONNXModel.find_nodes_by_initializer(graph, new_initializer)) == 0:
                            graph_helper.add_initializer(graph, new_initializer)
                            initializer_name_set.add(new_in_tensor)
                        ONNXModel.replace_node_input(node, in_tensor_name, new_in_tensor)

    @staticmethod
    def __process_nodes_input(
        graph: GraphProto, initializer_name_set: set, allowed_nodes: list, blocked_nodes, keep_io_types: bool
    ):
        for node in allowed_nodes:
            for input_name in node.input:
                if input_name in initializer_name_set:
                    continue
                if graph_helper.is_graph_input(graph, input_name) and not keep_io_types:
                    continue
                if not graph_helper.is_graph_input(graph, input_name):
                    parent = graph_helper.output_name_to_node(graph)[input_name]
                    if parent in allowed_nodes and node in allowed_nodes:
                        continue
                FP16Converter.__add_cast_on_node_input(graph, node, input_name, TensorProto.FLOAT16)
        for node in blocked_nodes:
            for input_name in node.input:
                if input_name in initializer_name_set:
                    continue
                if graph_helper.is_graph_input(graph, input_name) and keep_io_types:
                    continue
                if not graph_helper.is_graph_input(graph, input_name):
                    parent = graph_helper.output_name_to_node(graph)[input_name]
                    if parent in blocked_nodes and node in blocked_nodes:
                        continue
                FP16Converter.__add_cast_on_node_input(graph, node, input_name, TensorProto.FLOAT)

    @staticmethod
    def __process_nodes_output(graph: GraphProto, allowed_nodes: list, blocked_nodes, keep_io_types: bool):
        for node in allowed_nodes:
            for output_name in node.output:
                if graph_helper.is_graph_output(graph, output_name) and keep_io_types:
                    FP16Converter.__add_cast_on_node_output(graph, node, output_name, TensorProto.FLOAT)
        for node in blocked_nodes:
            for output_name in node.output:
                if graph_helper.is_graph_output(graph, output_name) and not keep_io_types:
                    FP16Converter.__add_cast_on_node_output(graph, node, output_name, TensorProto.FLOAT16)

    @staticmethod
    def __process_graph(graph: GraphProto, allow_list: list, keep_io_types: bool):
        initializer_name_set = graph_helper.get_initializer_name_set(graph)
        blocked_nodes = [node for node in graph.node if node.op_type not in allow_list]
        allowed_nodes = [node for node in graph.node if node.op_type in allow_list]
        FP16Converter.__cast_graph_io(graph, keep_io_types)
        FP16Converter.__cast_initializers(graph, initializer_name_set, allowed_nodes)
        FP16Converter.__process_nodes_input(
            graph,
            initializer_name_set,
            allowed_nodes,
            blocked_nodes,
            keep_io_types,
        )
        FP16Converter.__process_nodes_output(graph, allowed_nodes, blocked_nodes, keep_io_types)

    def process(self, keep_io_types: bool):
        if self.model is None:
            return False
        self.__process_graph(self.model.graph, self.allow_list, keep_io_types)
        return True

    @staticmethod
    def convert_model(model, keep_io_types, op_allow_list=None):
        FP16Converter(model, op_allow_list).process(keep_io_types)
        return

    @staticmethod
    def convert_model_file(input_path, output_path, keep_io_types, op_allow_list=None):
        converter = FP16Converter(onnx.load(input_path), op_allow_list)
        if keep_io_types == "True" or keep_io_types == "true":
            converter.process(True)
        else:
            converter.process(False)
        converter.export_model_to_path(f"{output_path}")
        print(f"Converted model saved to {output_path}")


def main():
    args = FP16Converter.parse_arguments()
    FP16Converter.convert_model_file(
        args.input, args.output, keep_io_types=args.keep_io_types, op_allow_list=args.allow_list
    )


if __name__ == "__main__":
    main()
