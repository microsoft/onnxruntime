import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as TensorType
from onnx import numpy_helper

from onnxruntime.quantization.onnx_model import ONNXModel


class FP16Converter:
    def __init__(self):
        self.model = None
        self.supported_ops = ["Conv", "MatMul"]

    @staticmethod
    def __cast_intializer_to_fp16(initializer, new_name):
        if int(initializer.data_type) == TensorType.FLOAT:
            new_tensor = np.asarray(numpy_helper.to_array(initializer), dtype=np.float16)
            return numpy_helper.from_array(new_tensor, new_name)
        return initializer

    def __conversion(self, op):
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

    def convert_op(self, op: str):
        if op not in self.supported_ops or self.model is None:
            print("Unsupported op: " + op)
            print("Supported ops: " + str(self.supported_ops))
            return False
        return self.__conversion(op)

    def convert_all(self):
        return map(self.convert_op, self.supported_ops)

    def import_model_from_path(self, model_path: Path):
        self.model = onnx.load(model_path.as_posix())

    def export_model_to_path(self, model_path: Path):
        if self.model is not None:
            onnx.save(self.model, model_path.as_posix())

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python fp16_converter.py <in_model_path> <out_model_path>")
        return
    int_model_path = Path(args[0])
    out_model_path = Path(args[1])
    convertor = FP16Converter()
    convertor.import_model_from_path(int_model_path)
    convertor.convert_all()
    convertor.export_model_to_path(out_model_path)


if __name__ == "__main__":
    main()
