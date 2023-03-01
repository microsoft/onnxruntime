import itertools

import argparse
import numpy as np
import onnx
import packaging.version as pv
from onnx import (
    AttributeProto,
    ValueInfoProto,
    GraphProto,
    ModelProto,
    TensorProto,
    helper,
    numpy_helper,
)
from onnxruntime.quantization.onnx_model import ONNXModel


# from onnx import onnx_pb as onnx_pb


class FP16Converter:
    default_allow_list = ["Conv", "MatMul"]

    def __init__(self, model=None, allow_list=None):
        self.allow_list = allow_list if allow_list is not None else self.default_allow_list
        self.model = model if model is not None else None

    @staticmethod
    def __make_value_info_from_tensor(tensor: TensorProto) -> ValueInfoProto:
        if not isinstance(tensor, TensorProto):
            raise ValueError("Expected input type is an ONNX TensorProto but got %s" % type(tensor))
        shape = numpy_helper.to_array(tensor).shape
        return helper.make_tensor_value_info(tensor.name, tensor.data_type, shape)

    @staticmethod
    def __convert_np_float16_to_int(np_array: np.ndarray(shape=(), dtype=np.float16)) -> list[int]:
        """
        Convert numpy float16 to python int.

        :param np_array: numpy float16 list
        :return int_list: python int list
        """
        return [int(bin(_.view("H"))[2:].zfill(16), 2) for _ in np_array]

    @staticmethod
    def __convert_np_float_to_float16(
        np_array: np.ndarray(shape=(), dtype=np.float32),
    ) -> np.ndarray(shape=(), dtype=np.float16):
        """
        Convert float32 numpy array to float16 without changing sign or finiteness.
        Positive values less than min_positive_val are mapped to min_positive_val.
        Positive finite values greater than max_finite_val are mapped to max_finite_val.
        Similar for negative values. NaN, 0, inf, and -inf are unchanged.
        """

        min_positive_val = 5.96e-08
        max_finite_val = 65504.0

        def between(a, b, c):
            return np.logical_and(a < b, b < c)

        np_array = np.where(between(0, np_array, min_positive_val), min_positive_val, np_array)
        np_array = np.where(between(-min_positive_val, np_array, 0), -min_positive_val, np_array)
        np_array = np.where(between(max_finite_val, np_array, float("inf")), max_finite_val, np_array)
        np_array = np.where(between(float("-inf"), np_array, -max_finite_val), -max_finite_val, np_array)
        return np.float16(np_array)

    def __convert_tensor_float_to_float16(self, tensor: TensorProto) -> TensorProto:
        """Convert tensor float to float16.

        Args:
            tensor (TensorProto): the tensor to convert.
        Raises:
            ValueError: input type is not TensorProto.

        Returns:
            TensorProto: the converted tensor.
        """

        if not isinstance(tensor, TensorProto):
            raise ValueError("Expected input type is an ONNX TensorProto but got %s" % type(tensor))
        if tensor.data_type == TensorProto.FLOAT16:
            return tensor

        tensor.data_type = TensorProto.FLOAT16
        # convert float_data (float type) to float16 and write to int32_data
        if tensor.float_data:
            float16_data = self.__convert_np_float_to_float16(np.array(tensor.float_data))
            int_list = self.__convert_np_float16_to_int(float16_data)
            tensor.int32_data[:] = int_list
            tensor.float_data[:] = []
        # convert raw_data (bytes type)
        if tensor.raw_data:
            # convert n.raw_data to float
            float32_list = np.frombuffer(tensor.raw_data, dtype="float32")
            # convert float to float16
            float16_list = self.__convert_np_float_to_float16(float32_list)
            # convert float16 to bytes and write back to raw_data
            tensor.raw_data = float16_list.tobytes()
        return tensor

    def __convert_model_float_to_float16(
        self,
        model: ModelProto,
        keep_io_types=False,
        disable_shape_infer=False,
    ) -> ModelProto:
        """
        Convert tensor float type in the ONNX ModelProto input to tensor



        :param model: ONNX ModelProto object
        :param keep_io_types: If True, keep the original input/output tensor types.
        :param disable_shape_infer: Type/shape information is needed for conversion to work.
                                    Set to True only if the model already has type/shape information for all tensors.
        :return: converted ONNX ModelProto object

        """
        if not isinstance(model, ModelProto):
            raise ValueError("Expected model type is an ONNX ModelProto but got %s" % type(model))

        func_infer_shape = None
        if not disable_shape_infer and pv.Version(onnx.__version__) >= pv.Version("1.2"):
            try:
                from onnx.shape_inference import infer_shapes

                func_infer_shape = infer_shapes
            finally:
                pass

        # create a queue for BFS
        queue = []
        value_info_list = []
        node_list = []
        # type inference on input model
        if func_infer_shape is not None:
            model = func_infer_shape(model)
        queue.append(model)
        name_mapping = {}
        graph_io_to_skip = set()
        cast_operators = set()
        if keep_io_types:
            for i, graph_input in enumerate(model.graph.input):  # checking graph inputs
                if graph_input.type.tensor_type.elem_type == TensorProto.FLOAT:
                    output_name = "graph_input_cast_" + str(i)
                    name_mapping[graph_input.name] = output_name
                    graph_io_to_skip.add(graph_input.name)

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
                    value_info_list.append(new_value_info)
                    cast_operators.add(node_name)

            for i, graph_output in enumerate(model.graph.output):
                if graph_output.type.tensor_type.elem_type == TensorProto.FLOAT:
                    input_name = "graph_output_cast_" + str(i)
                    name_mapping[graph_output.name] = input_name
                    graph_io_to_skip.add(graph_output.name)

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
                    value_info_list.append(new_value_info)
                    cast_operators.add(node_name)

        while queue:
            for model_ in queue:
                # if model_ is model, push model_.graph (GraphProto)
                if isinstance(model_, ModelProto):
                    queue.append(model_.graph)
                # if model_ is model.graph, push model_.node.attribute (AttributeProto)
                if isinstance(model_, GraphProto):
                    for node in model_.node:
                        # if node is in the block list (doesn't support float16), no conversion for the node,
                        # and save the node for further processing
                        if node.name in cast_operators:
                            continue
                        for i in range(len(node.input)):
                            if node.input[i] in name_mapping:
                                node.input[i] = name_mapping[node.input[i]]
                        for i in range(len(node.output)):
                            if node.output[i] in name_mapping:
                                node.output[i] = name_mapping[node.output[i]]
                        # don't push the attr into queue for the node in node_keep_data_type_list
                        # so it will not be converted to float16
                        if node.op_type not in self.allow_list and node.op_type != "Cast":
                            node_list.append(node)
                        elif node.op_type in self.allow_list or node.op_type == "Cast":
                            if node.op_type == "Cast":
                                for attr in node.attribute:
                                    if attr.name == "to" and attr.i == TensorProto.FLOAT:
                                        attr.i = TensorProto.FLOAT16
                                        break
                            for attr in node.attribute:
                                queue.append(attr)
                # if model_ is model.graph.node.attribute, push model_.g and model_.graphs (GraphProto)
                # and process node.attribute.t and node.attribute.tensors (TensorProto)
                if isinstance(model_, AttributeProto):
                    queue.append(model_.g)
                    for graph in model_.graphs:
                        queue.append(graph)
                    model_.t.CopyFrom(self.__convert_tensor_float_to_float16(model_.t))
                    for tensor in model_.tensors:
                        self.__convert_tensor_float_to_float16(tensor)
                # if model_ is graph, process graph.initializer(TensorProto), input, output and value_info (
                # ValueInfoProto)
                if isinstance(model_, GraphProto):
                    for initializer in model_.initializer:  # TensorProto type
                        if initializer.data_type == TensorProto.FLOAT:
                            initializer = self.__convert_tensor_float_to_float16(initializer)
                            value_info_list.append(self.__make_value_info_from_tensor(initializer))
                    # for all ValueInfoProto with tensor(float) type in input, output and value_info, convert them to
                    # tensor(float16) except map and seq(map). And save them in value_info_list for further processing
                    for val_info in itertools.chain(model_.input, model_.output, model_.value_info):
                        if val_info.type.tensor_type.elem_type == TensorProto.FLOAT:
                            if val_info.name not in graph_io_to_skip:
                                val_info.type.tensor_type.elem_type = TensorProto.FLOAT16
                                value_info_list.append(val_info)
            queue.pop(0)

        # process the nodes in block list that doesn't support tensor(float16)
        for node in node_list:
            # if input's name is in the value_info_list meaning input is tensor(float16) type,
            # insert a float16 to float Cast node before the node,
            # change current node's input name and create new value_info for the new name
            for i in range(len(node.input)):
                node_input = node.input[i]
                for value_info in value_info_list:
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
                for value_info in value_info_list:
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

    def convert(self, keep_io_types=True):
        if self.model is None:
            return False
        self.model = self.__convert_model_float_to_float16(self.model, keep_io_types=keep_io_types)
        return True

    @staticmethod
    def convert_model(model, keep_io_types=True, op_allow_list=None):
        FP16Converter(model, op_allow_list).convert(keep_io_types)
        return

    @staticmethod
    def convert_model_file(input_path, output_path, keep_io_types=True, op_allow_list=None):
        converter = FP16Converter(onnx.load(input_path), op_allow_list)
        converter.convert(keep_io_types)
        converter.export_model_to_path(output_path)

    def set_allow_list(self, allow_list: list = None):
        self.allow_list = allow_list if allow_list is None else self.default_allow_list

    def import_model_from_path(self, model_path):
        self.model = onnx.load(model_path)

    def export_model_to_path(self, model_path, use_external_data_format=False):
        if self.model is not None:
            ONNXModel(self.model).save_model_to_file(model_path, use_external_data_format)

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Graph fp16 conversion tool for ONNX Runtime."
        "It convert ONNX graph from fp32 to fp16 using --allow_list."
    )
    parser.add_argument("--input", required=True, type=str, help="input onnx model path")

    parser.add_argument("--output", required=True, type=str, help="optimized onnx model path")
    parser.add_argument(
        "--allow_list",
        required=False,
        default=[],
        nargs="+",
        help="allow list which contains all supported ops that can be converted into fp16.",
    )
    parser.add_argument(
        "--use_external_data_format",
        required=False,
        action="store_true",
        default=False,
        help="use external data format to store large model (>2GB)",
    )
    parser.set_defaults(use_external_data_format=False)
    parser.add_argument(
        "--keep_io_types",
        type=bool,
        required=False,
        help="keep input and output types as float32",
        default=False,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    FP16Converter.convert_model_file(args.input, args.output, args.use_external_data_format, args.allow_list)


if __name__ == "__main__":
    main()
