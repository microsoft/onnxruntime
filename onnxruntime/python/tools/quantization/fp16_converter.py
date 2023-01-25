from pathlib import Path

import onnx
from onnx import TensorProto as TensorType
from onnxruntime.quantization.onnx_model import ONNXModel


class FP16Converter:
    def __init__(self):
        self.model = None
        self.supported_ops = ['Conv']

    def __conv_conversion(self):
        model = ONNXModel(self.model)
        conv_nodes = model.find_nodes_by_type('Conv')
        for node in conv_nodes:
            for in_tensor in node.input:
                cast_node_name = 'Cast/' + in_tensor + '/' + node.name
                new_in_tensor = 'Cast_' + in_tensor + '_' + node.name
                cast_node = onnx.helper.make_node(
                    "Cast",
                    [in_tensor],
                    [new_in_tensor],
                    name=cast_node_name,
                    to=TensorType.FLOAT16)
                model.add_node(cast_node)
                model.replace_node_input(node, in_tensor, new_in_tensor)
            i = 0
            children = model.get_children(node)
            for out_tensor in node.output:
                cast_node_name = 'Cast/' + node.name + '/' + out_tensor
                new_out_tensor = 'Cast_' + node.name + '_' + out_tensor
                cast_node = onnx.helper.make_node(
                    "Cast",
                    [out_tensor],
                    [new_out_tensor],
                    name=cast_node_name,
                    to=TensorType.FLOAT)
                model.add_node(cast_node)
                model.replace_node_input(children[i], out_tensor, new_out_tensor)
        return True

    def convert_op(self, op: str):
        if op not in self.supported_ops or self.model is None:
            return False
        if op is 'Conv':
            return self.__conv_conversion()

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
