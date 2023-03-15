import itertools
import logging
from typing import Dict

import numpy as np
import onnx
import packaging.version as pv
from onnx import AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto, helper, numpy_helper
from onnxruntime.quantization.onnx_model import ONNXModel

from python.tools.quantization.onnx_model_converter_base import ConverterBase

logger = logging.getLogger(__name__)


class InitializerTracker:
    """Class for keeping track of initializer."""

    def __init__(self, initializer: TensorProto):
        self.initializer = initializer
        self.fp32_nodes = []
        self.fp16_nodes = []

    def add_node(self, node: NodeProto, is_node_blocked):
        if is_node_blocked:
            self.fp32_nodes.append(node)
        else:
            self.fp16_nodes.append(node)


class FP16ConverterFinal(ConverterBase):
    @staticmethod
    def __cast_initializer_to_fp16(initializer, new_name):
        if int(initializer.data_type) == TensorProto.FLOAT:
            new_tensor = np.asarray(numpy_helper.to_array(initializer), dtype=np.float16)
            return numpy_helper.from_array(new_tensor, new_name)
        return initializer

    tensor_src: Dict[str, NodeProto] = {}
    tensor_dest: Dict[str, NodeProto] = {}

    def __init__(self, model=None, allow_list=None):
        super().__init__(model, allow_list)
        self.onnx_model = ONNXModel(self.model)
        self.initializer_name_set = self.onnx_model.get_initializer_name_set()
        self.non_initializer_inputs = self.onnx_model.get_non_initializer_inputs()
        self.input_name_to_nodes = self.onnx_model.input_name_to_nodes()
        self.output_name_to_node = self.onnx_model.output_name_to_node()
        self.new_nodes = self.model.graph.node

    def process(self, keep_io_types=True):
        if self.model is None:
            return False
        self.model.graph.CopyFrom(self.__create_new_graph(keep_io_types))
        return True

    def __create_new_graph(self, keep_io_types=False):
        new_input_list = []
        new_output_list = []
        for graph_input in self.model.graph.input:
            if graph_input.name in self.initializer_name_set:
                continue
            for node in self.input_name_to_nodes[graph_input.name]:
                new_value_info = ValueInfoProto()
                if keep_io_types:
                    if node.op_type not in self.allow_list:
                        new_value_info.CopyFrom(graph_input)
                        new_input_list.append(new_value_info)
                    else:
                        new_value_info = ValueInfoProto()
                        # Add Cast 16 Node between input and its children nodes
                else:
                    if node.op_type not in self.allow_list:
                        new_value_info = ValueInfoProto()
                        # Add Cast 32 Node between input and its children nodes
                    else:
                        new_value_info = ValueInfoProto()
                        # Cast tenser to 16
                new_input_list.append(new_value_info)
        for graph_output in self.model.graph.output:
            node = self.output_name_to_node[graph_output.name]
            new_value_info = ValueInfoProto()
            if keep_io_types:
                if node.op_type not in self.allow_list:
                    new_value_info.CopyFrom(graph_output)
                    new_input_list.append(new_value_info)
                else:
                    None
                    # Add Cast 16 Node between output and its parent node
            else:
                if node.op_type not in self.allow_list:
                    None
                    # Add Cast 32 Node between output and its parent node
                else:
                    None
                    # Cast tenser to 16
            new_output_list.append(new_value_info)

        new_graph = GraphProto(
            name=self.model.graph.name,
            input=new_input_list,
            output=new_output_list,
            quantization_annotation=self.model.graph.quantization_annotation,
            sparse_initializer=self.model.graph.sparse_initializer,
            doc_string=self.model.graph.doc_string,
        )
        return new_graph

    def __convert_io(self, model: ModelProto) -> ModelProto:
        """
        Convert input/output tensor types to float16 if keep_io_types is False.
        """
        for i, graph_input in enumerate(model.graph.input):  # checking graph inputs
            output_name = "graph_input_cast_" + str(i)
            self.name_mapping[graph_input.name] = output_name
            self.graph_io_to_skip.add(graph_input.name)
            node_name = "graph_input_cast" + str(i)
            new_value_info = model.graph.value_info.add()
            new_value_info.CopyFrom(graph_input)
            new_value_info.name = output_name
            new_value_info.type.tensor_type.elem_type = TensorProto.FLOAT16
            # add Cast node (from tensor(float) to tensor(float16) after graph input
            new_node = [
                helper.make_node(
                    "Cast",
                    [graph_input.name],
                    [output_name],
                    to=TensorProto.FLOAT16,
                    name=node_name,
                )
            ]
            model.graph.node.extend(new_node)
            self.value_info_list.append(new_value_info)
            self.io_casts.add(node_name)
        for i, graph_output in enumerate(model.graph.output):
            input_name = "graph_output_cast_" + str(i)
            self.name_mapping[graph_output.name] = input_name
            self.graph_io_to_skip.add(graph_output.name)

            node_name = "graph_output_cast" + str(i)
            # add Cast node (from tensor(float16) to tensor(float) before graph output
            new_value_info = model.graph.value_info.add()
            new_value_info.CopyFrom(graph_output)
            new_value_info.name = input_name
            new_value_info.type.tensor_type.elem_type = TensorProto.FLOAT16
            new_node = [
                helper.make_node(
                    "Cast",
                    [input_name],
                    [graph_output.name],
                    to=TensorProto.FLOAT,
                    name=node_name,
                )
            ]
            model.graph.node.extend(new_node)
            self.value_info_list.append(new_value_info)
            self.io_casts.add(node_name)
        return model

    def __convert_model_float_to_float16(
        self,
        model: ModelProto,
        keep_io_types=False,
    ) -> ModelProto:
        """
        Convert tensor float type in the ONNX ModelProto input to tensor

        :param model: ONNX ModelProto object
        :param keep_io_types: If True, keep the original input/output tensor types.
        :return: converted ONNX ModelProto object

        """
        if not isinstance(model, ModelProto):
            raise ValueError("Expected model type is an ONNX ModelProto but got %s" % type(model))

        # create a queue for BFS
        queue = []
        self.value_info_list = []
        node_list = []
        # type inference on input model
        queue.append(model)
        self.name_mapping = {}
        self.graph_io_to_skip = set()
        self.io_casts = set()

        if keep_io_types:
            self.__convert_io(model)

        fp32_initializers: Dict[str, InitializerTracker] = {}
        while queue:
            q = queue.pop(0)
            # if model_ is model, push model_.graph (GraphProto)
            if isinstance(q, ModelProto):
                queue.append(q.graph)
            # if model_ is graph push model_.node.attribute (AttributeProto)
            # process graph.initializer(TensorProto), input, output and value_info (
            # ValueInfoProto)
            if isinstance(q, GraphProto):
                for node in q.node:
                    # if node is in the block list (doesn't support float16), no conversion for the node,
                    # and save the node for further processing
                    if node.name in self.io_casts:
                        continue
                    for i in range(len(node.input)):
                        if node.input[i] in self.name_mapping:
                            node.input[i] = self.name_mapping[node.input[i]]
                    for i in range(len(node.output)):
                        if node.output[i] in self.name_mapping:
                            node.output[i] = self.name_mapping[node.output[i]]
                    # don't push the attr into queue for the node in node_keep_data_type_list,
                    # so it will not be converted to float16
                    is_node_blocked = node.op_type not in self.allow_list
                    for node_input in node.input:
                        if node_input in fp32_initializers:
                            fp32_initializers[node_input].add_node(node, is_node_blocked)
                    if is_node_blocked:
                        node_list.append(node)
                    else:
                        if node.op_type == "Cast":
                            for attr in node.attribute:
                                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                                    attr.i = TensorProto.FLOAT16
                                    break
                        for attr in node.attribute:
                            queue.append(attr)

                # for all ValueInfoProto with tensor(float) type in input, output and value_info,
                # convert them to
                # tensor(float16) except map and seq(map). And save them in value_info_list for further
                # processing
                for val_info in itertools.chain(q.input, q.output, q.value_info):
                    if val_info.type.tensor_type.elem_type == TensorProto.FLOAT:
                        if val_info.name not in self.graph_io_to_skip:
                            val_info.type.tensor_type.elem_type = TensorProto.FLOAT16
                            self.value_info_list.append(val_info)
            # if model_ is model.graph.node.attribute, push model_.g and model_.graphs (GraphProto)
            # and process node.attribute.t and node.attribute.tensors (TensorProto)
            if isinstance(q, AttributeProto):
                queue.append(q.g)
                for graph in q.graphs:
                    queue.append(graph)
                q.t.CopyFrom(self._convert_tensor_float_to_float16(q.t))
                for tensor in q.tensors:
                    self._convert_tensor_float_to_float16(tensor)
        # process the nodes in block list that doesn't support tensor(float16)
        for node in node_list:
            # if input's name is in the value_info_list meaning input is tensor(float16) type,
            # insert a float16 to float Cast node before the node,
            # change current node's input name and create new value_info for the new name
            for i in range(len(node.input)):
                node_input = node.input[i]
                for value_info in self.value_info_list:
                    if node_input == value_info.name:
                        # create new value_info for current node's new input name
                        new_value_info = model.graph.value_info.add()
                        new_value_info.CopyFrom(value_info)
                        output_name = node.name + "_input_cast_" + str(i)
                        new_value_info.name = output_name
                        new_value_info.type.tensor_type.elem_type = TensorProto.FLOAT
                        # add Cast node (from tensor(float16) to tensor(float) before current node
                        node_name = node.name + "_input_cast" + str(i)
                        new_node = [
                            helper.make_node(
                                "Cast",
                                [node_input],
                                [output_name],
                                to=TensorProto.FLOAT,
                                name=node_name,
                            )
                        ]
                        model.graph.node.extend(new_node)
                        # change current node's input name
                        node.input[i] = output_name
                        break
            # if output's name is in the value_info_list meaning output is tensor(float16) type, insert a float to
            # float16 Cast node after the node, change current node's output name and create new value_info for the
            # new name
            for i in range(len(node.output)):
                output = node.output[i]
                for value_info in self.value_info_list:
                    if output == value_info.name:
                        # create new value_info for current node's new output
                        new_value_info = model.graph.value_info.add()
                        new_value_info.CopyFrom(value_info)
                        input_name = node.name + "_output_cast_" + str(i)
                        new_value_info.name = input_name
                        new_value_info.type.tensor_type.elem_type = TensorProto.FLOAT
                        # add Cast node (from tensor(float) to tensor(float16) after current node
                        node_name = node.name + "_output_cast" + str(i)
                        new_node = [
                            helper.make_node(
                                "Cast",
                                [input_name],
                                [output],
                                to=TensorProto.FLOAT16,
                                name=node_name,
                            )
                        ]
                        model.graph.node.extend(new_node)
                        # change current node's input name
                        node.output[i] = input_name
                        break

        return model

    @staticmethod
    def convert_model(model, keep_io_types=True, op_allow_list=None):
        FP16ConverterFinal(model, op_allow_list).process(keep_io_types)
        return

    @staticmethod
    def convert_model_file(input_path, output_path, keep_io_types=True, op_allow_list=None):
        converter = FP16ConverterFinal(onnx.load(input_path), op_allow_list)
        converter.process(keep_io_types)
        converter.export_model_to_path(output_path)


def main():
    args = FP16ConverterFinal.parse_arguments()
    FP16ConverterFinal.convert_model_file(args.input, args.output, args.use_external_data_format, args.allow_list)


if __name__ == "__main__":
    main()
