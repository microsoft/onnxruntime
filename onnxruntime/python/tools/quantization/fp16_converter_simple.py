import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as TensorType
from onnx import numpy_helper

from onnxruntime.quantization.onnx_model import ONNXModel


class FP16ConverterSimple:
    default_allow_list = ["Conv", "MatMul", "Relu"]

    def __init__(self, model=None, allow_list=None):
        self.allow_list = allow_list if allow_list is not None else self.default_allow_list
        self.model = model if model is not None else None

    @staticmethod
    def __cast_intializer_to_fp16(initializer, new_name):
        if int(initializer.data_type) == TensorType.FLOAT:
            new_tensor = np.asarray(numpy_helper.to_array(initializer), dtype=np.float16)
            return numpy_helper.from_array(new_tensor, new_name)
        return initializer

    def __convert_all_io(self, op):
        model = ONNXModel(self.model)
        conv_nodes = model.find_nodes_by_type(op)
        initializer_name_set = model.get_initializer_name_set()
        for node in conv_nodes:
            for in_tensor_name in node.input:
                if in_tensor_name not in initializer_name_set:
                    new_in_tensor = "Cast_" + in_tensor_name + "_" + node.name
                    cast_node_name = "Cast/" + in_tensor_name + "/" + node.name
                    cast_node = onnx.helper.make_node(
                        "Cast", [in_tensor_name], [new_in_tensor], name=cast_node_name, to=TensorType.FLOAT16
                    )
                    model.add_node(cast_node)
                    model.replace_node_input(node, in_tensor_name, new_in_tensor)
                else:
                    initializer = model.get_initializer(in_tensor_name)
                    new_in_tensor = initializer.name + "_fp16"
                    new_initializer = self.__cast_intializer_to_fp16(initializer, new_in_tensor)
                    # remove the old initializer if it is not used by any other node
                    if len(model.find_nodes_by_initializer(model.graph(), initializer)) == 1:
                        model.remove_initializer(initializer)
                    # add the new initializer if it is not already present
                    if len(model.find_nodes_by_initializer(model.graph(), new_initializer)) == 0:
                        model.add_initializer(new_initializer)
                model.replace_node_input(node, in_tensor_name, new_in_tensor)
            for out_tensor in node.output:
                cast_node_name = "Cast/" + node.name + "/" + out_tensor
                new_out_tensor = "Cast_" + node.name + "_" + out_tensor
                cast_node = onnx.helper.make_node(
                    "Cast", [new_out_tensor], [out_tensor], name=cast_node_name, to=TensorType.FLOAT
                )
                model.add_node(cast_node)
                model.replace_node_output(node, out_tensor, new_out_tensor)
        return True

    @staticmethod
    def convert_model_file(input_path, output_path, keep_io_types=True, op_allow_list=None):
        converter = FP16ConverterSimple(onnx.load(input_path), op_allow_list)
        converter.convert()
        converter.export_model_to_path(output_path)

    def convert_op(self, op: str):
        if op not in self.allow_list or self.model is None:
            print("Unsupported op: " + op)
            print("Supported ops: " + str(self.allow_list))
            return False
        return self.__convert_all_io(op)

    def convert(self):
        self.convert_op("Relu")
        return self.convert_op("Conv")
        # return map(self.convert_op, self.allow_list)

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
    FP16ConverterSimple.convert_model_file(args.input, args.output, args.use_external_data_format, args.allow_list)


if __name__ == "__main__":
    main()
