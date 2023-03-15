import numpy as np
import onnx
from onnx import TensorProto as TensorType
from onnx import numpy_helper
from onnxruntime.quantization.onnx_model import ONNXModel

from python.tools.quantization.onnx_model_converter_base import ConverterBase


class FP16ConverterSimple(ConverterBase):
    def __init__(self, model=None, allow_list=None):
        super().__init__(model, allow_list)

    @staticmethod
    def __cast_initializer_to_fp16(initializer, new_name):
        if int(initializer.data_type) == TensorType.FLOAT:
            new_tensor = np.asarray(numpy_helper.to_array(initializer), dtype=np.float16)
            return numpy_helper.from_array(new_tensor, new_name)
        return initializer

    def __convert_op_io(self, op: str):
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
                    new_initializer = self.__cast_initializer_to_fp16(initializer, new_in_tensor)
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
        converter.process()
        converter.export_model_to_path(output_path)

    def convert_ops(self, ops: [str]):
        for op in ops:
            self.__convert_op_io(op)
        return

    def process(self):
        return self.convert_ops(self.allow_list)


def main():
    args = FP16ConverterSimple.parse_arguments()
    FP16ConverterSimple.convert_model_file(args.input, args.output, args.use_external_data_format, args.allow_list)


if __name__ == "__main__":
    main()
