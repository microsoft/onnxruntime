import argparse
import itertools

import numpy as np
import onnx
import packaging.version as pv
from onnx import helper, numpy_helper
from onnx import onnx_pb as onnx_proto
from onnxruntime.quantization.onnx_model import ONNXModel

# from onnxruntime.transformers.float16 import convert_tensor_float_to_float16

ALLOWED_OPS_LIST = ["Conv", "MatMul"]


def make_value_info_from_tensor(tensor):
    shape = numpy_helper.to_array(tensor).shape
    return helper.make_tensor_value_info(tensor.name, tensor.data_type, shape)


def _npfloat16_to_int(np_list):
    """
    Convert numpy float16 to python int.

    :param np_list: numpy float16 list
    :return int_list: python int list
    """
    return [int(bin(_.view("H"))[2:].zfill(16), 2) for _ in np_list]


def convert_np_to_float16(np_array, min_positive_val=5.96e-08, max_finite_val=65504.0):
    """
    Convert float32 numpy array to float16 without changing sign or finiteness.
    Positive values less than min_positive_val are mapped to min_positive_val.
    Positive finite values greater than max_finite_val are mapped to max_finite_val.
    Similar for negative values. NaN, 0, inf, and -inf are unchanged.
    """

    def between(a, b, c):
        return np.logical_and(a < b, b < c)

    np_array = np.where(between(0, np_array, min_positive_val), min_positive_val, np_array)
    np_array = np.where(between(-min_positive_val, np_array, 0), -min_positive_val, np_array)
    np_array = np.where(between(max_finite_val, np_array, float("inf")), max_finite_val, np_array)
    np_array = np.where(between(float("-inf"), np_array, -max_finite_val), -max_finite_val, np_array)
    return np.float16(np_array)


def convert_tensor_float_to_float16(tensor, min_positive_val=5.96e-08, max_finite_val=65504.0):
    """Convert tensor float to float16.

    Args:
        tensor (TensorProto): the tensor to convert.
        min_positive_val (float, optional): minimal positive value. Defaults to 1e-7.
        max_finite_val (float, optional): maximal finite value. Defaults to 1e4.

    Raises:
        ValueError: input type is not TensorProto.

    Returns:
        TensorProto: the converted tensor.
    """

    if not isinstance(tensor, onnx_proto.TensorProto):
        raise ValueError("Expected input type is an ONNX TensorProto but got %s" % type(tensor))

    if tensor.data_type == onnx_proto.TensorProto.FLOAT:
        tensor.data_type = onnx_proto.TensorProto.FLOAT16
        # convert float_data (float type) to float16 and write to int32_data
        if tensor.float_data:
            float16_data = convert_np_to_float16(np.array(tensor.float_data), min_positive_val, max_finite_val)
            int_list = _npfloat16_to_int(float16_data)
            tensor.int32_data[:] = int_list
            tensor.float_data[:] = []
        # convert raw_data (bytes type)
        if tensor.raw_data:
            # convert n.raw_data to float
            float32_list = np.frombuffer(tensor.raw_data, dtype="float32")
            # convert float to float16
            float16_list = convert_np_to_float16(float32_list, min_positive_val, max_finite_val)
            # convert float16 to bytes and write back to raw_data
            tensor.raw_data = float16_list.tobytes()
    return tensor


def convert_float_to_float16(
        model,
        min_positive_val=1e-7,
        max_finite_val=1e4,
        keep_io_types=False,
        disable_shape_infer=False,
        op_allow_list=None,
):
    """
    Convert tensor float type in the ONNX ModelProto input to tensor

    :param op_allow_list:
    :param keep_io_types:
    :param max_finite_val:
    :param min_positive_val:
    :param model: ONNX ModelProto object
    :param disable_shape_infer: Type/shape information is needed for conversion to work.
                                Set to True only if the model already has type/shape information for all tensors.
    :return: converted ONNX ModelProto object

    Examples:

    ::

        Example 1: Convert ONNX ModelProto object:
        from onnxmltools.utils.float16_converter import convert_float_to_float16
        new_onnx_model = convert_float_to_float16(onnx_model)

        Example 2: Convert ONNX model binary file:
        from onnxmltools.utils.float16_converter import convert_float_to_float16
        from onnxmltools.utils import load_model, save_model
        onnx_model = load_model('model.onnx')
        new_onnx_model = convert_float_to_float16(onnx_model)
        save_model(new_onnx_model, 'new_model.onnx')

    """
    func_infer_shape = None
    if not disable_shape_infer and pv.Version(onnx.__version__) >= pv.Version("1.2"):
        try:
            from onnx.shape_inference import infer_shapes

            func_infer_shape = infer_shapes
        finally:
            pass

    if not isinstance(model, onnx_proto.ModelProto):
        raise ValueError("Expected model type is an ONNX ModelProto but got %s" % type(model))

    # create op_allow_list
    if op_allow_list is None:
        op_allow_list = ALLOWED_OPS_LIST
    op_allow_list = set(op_allow_list)
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
    io_casts = set()
    if keep_io_types:
        for i, n in enumerate(model.graph.input):
            if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                output_name = "graph_input_cast_" + str(i)
                name_mapping[n.name] = output_name
                graph_io_to_skip.add(n.name)

                node_name = "graph_input_cast" + str(i)
                new_value_info = model.graph.value_info.add()
                new_value_info.CopyFrom(n)
                new_value_info.name = output_name
                new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                # add Cast node (from tensor(float) to tensor(float16) after graph input
                new_node = [helper.make_node("Cast", [n.name], [output_name], to=10, name=node_name)]
                model.graph.node.extend(new_node)
                value_info_list.append(new_value_info)
                io_casts.add(node_name)

        for i, n in enumerate(model.graph.output):
            if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                input_name = "graph_output_cast_" + str(i)
                name_mapping[n.name] = input_name
                graph_io_to_skip.add(n.name)

                node_name = "graph_output_cast" + str(i)
                # add Cast node (from tensor(float16) to tensor(float) before graph output
                new_value_info = model.graph.value_info.add()
                new_value_info.CopyFrom(n)
                new_value_info.name = input_name
                new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                new_node = [helper.make_node("Cast", [input_name], [n.name], to=1, name=node_name)]
                model.graph.node.extend(new_node)
                value_info_list.append(new_value_info)
                io_casts.add(node_name)

    while queue:
        next_level = []
        for q in queue:
            # if q is model, push q.graph (GraphProto)
            if isinstance(q, onnx_proto.ModelProto):
                next_level.append(q.graph)
            # if q is model.graph, push q.node.attribute (AttributeProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.node:
                    # if n is in the block list (doesn't support float16), no conversion for the node,
                    # and save the node for further processing
                    if n.name in io_casts:
                        continue
                    for i in range(len(n.input)):
                        if n.input[i] in name_mapping:
                            n.input[i] = name_mapping[n.input[i]]
                    for i in range(len(n.output)):
                        if n.output[i] in name_mapping:
                            n.output[i] = name_mapping[n.output[i]]
                    # don't add the attr into next_level for the node in node_keep_data_type_list
                    # so it will not be converted to float16
                    if n.op_type not in op_allow_list:
                        node_list.append(n)
                    else:
                        if n.op_type == "Cast":
                            for attr in n.attribute:
                                if attr.name == "to" and attr.i == 1:
                                    attr.i = 10
                                    break
                        for attr in n.attribute:
                            next_level.append(attr)
            # if q is model.graph.node.attribute, push q.g and q.graphs (GraphProto)
            # and process node.attribute.t and node.attribute.tensors (TensorProto)
            if isinstance(q, onnx_proto.AttributeProto):
                next_level.append(q.g)
                for n in q.graphs:
                    next_level.append(n)
                q.t.CopyFrom(convert_tensor_float_to_float16(q.t, min_positive_val, max_finite_val))
                for n in q.tensors:
                    n = convert_tensor_float_to_float16(n, min_positive_val, max_finite_val)
            # if q is graph, process graph.initializer(TensorProto), input, output and value_info (ValueInfoProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.initializer:  # TensorProto type
                    if n.data_type == onnx_proto.TensorProto.FLOAT:
                        n = convert_tensor_float_to_float16(n, min_positive_val, max_finite_val)
                        value_info_list.append(make_value_info_from_tensor(n))
                # for all ValueInfoProto with tensor(float) type in input, output and value_info, convert them to
                # tensor(float16) except map and seq(map). And save them in value_info_list for further processing
                for n in itertools.chain(q.input, q.output, q.value_info):
                    if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                        if n.name not in graph_io_to_skip:
                            n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                            value_info_list.append(n)
        queue = next_level

    # process the nodes in block list that doesn't support tensor(float16)
    for node in node_list:
        # if input's name is in the value_info_list meaning input is tensor(float16) type,
        # insert a float16 to float Cast node before the node,
        # change current node's input name and create new value_info for the new name
        for i in range(len(node.input)):
            input = node.input[i]
            for value_info in value_info_list:
                if input == value_info.name:
                    # create new value_info for current node's new input name
                    new_value_info = model.graph.value_info.add()
                    new_value_info.CopyFrom(value_info)
                    output_name = node.name + "_input_cast_" + str(i)
                    new_value_info.name = output_name
                    new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                    # add Cast node (from tensor(float16) to tensor(float) before current node
                    node_name = node.name + "_input_cast" + str(i)
                    new_node = [helper.make_node("Cast", [input], [output_name], to=1, name=node_name)]
                    model.graph.node.extend(new_node)
                    # change current node's input name
                    node.input[i] = output_name
                    break
        # if output's name is in the value_info_list meaning output is tensor(float16) type, insert a float to
        # float16 Cast node after the node, change current node's output name and create new value_info for the new name
        for i in range(len(node.output)):
            output = node.output[i]
            for value_info in value_info_list:
                if output == value_info.name:
                    # create new value_info for current node's new output
                    new_value_info = model.graph.value_info.add()
                    new_value_info.CopyFrom(value_info)
                    input_name = node.name + "_output_cast_" + str(i)
                    new_value_info.name = input_name
                    new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                    # add Cast node (from tensor(float) to tensor(float16) after current node
                    node_name = node.name + "_output_cast" + str(i)
                    new_node = [helper.make_node("Cast", [input_name], [output], to=10, name=node_name)]
                    model.graph.node.extend(new_node)
                    # change current node's input name
                    node.output[i] = input_name
                    break

    return model


class FP16Converter:
    def __init__(self):
        self.allowed_list = None
        self.model = None

    def convert(self, keep_io_types=True):
        if self.model is None:
            return False
        self.model = convert_float_to_float16(self.model, keep_io_types=keep_io_types, op_allow_list=self.allowed_list)
        return True

    def set_allowed_list(self, allowed_list: list):
        self.allowed_list = allowed_list

    def import_model_from_path(self, model_path):
        self.model = onnx.load(model_path)

    def export_model_to_path(self, model_path, use_external_data_format=False):
        if self.model is not None:
            ONNXModel(self.model).save_model_to_file(model_path, use_external_data_format)

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model


def __parse_arguments():
    parser = argparse.ArgumentParser(
        description="Graph fp16 conversion tool for ONNX Runtime."
        "It convert ONNX graph from fp32 to fp16 using --allowed_list."
    )
    parser.add_argument("--input", required=True, type=str, help="input onnx model path")

    parser.add_argument("--output", required=True, type=str, help="optimized onnx model path")
    parser.add_argument(
        "--allowed_list",
        required=False,
        default=[],
        nargs="+",
        help="Allowed list which contains all supported ops that can be converted into fp16.",
    )
    parser.set_defaults(allowed_list=ALLOWED_OPS_LIST)
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
    )
    parser.set_defaults(keep_io_types=True)

    args = parser.parse_args()
    return args


def main():
    args = __parse_arguments()
    convertor = FP16Converter()
    convertor.import_model_from_path(args.input)
    convertor.set_allowed_list(args.allowed_list)
    convertor.convert(args.keep_io_types)
    convertor.export_model_to_path(args.output, args.use_external_data_format)


if __name__ == "__main__":
    main()
