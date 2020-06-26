import onnx
from onnx import helper
from onnx import TensorProto
from enum import Enum

def GenerateModel(model_name, sign):
    nodes = [  # subgraph
        helper.make_node("MatMulInteger", ["A", "B", "a_zero_point", "b_zero_point"], ["matmul_output_int32"], "MatMulInteger"),
        helper.make_node("Mul", ["a_scale", "b_scale"], ["multiplier"], "mul_right"),
        helper.make_node("Cast", ["matmul_output_int32"], ["matmul_output_float"], "cast", to=1),
        helper.make_node("Mul", ["matmul_output_float", "multiplier"], ["Y"], "mul_bottom"),
    ]

    graph = helper.make_graph(
        nodes,
        "DynamicQuantizeMatMul_fusion",  #name
        [  # inputs
            helper.make_tensor_value_info('A', TensorProto.UINT8, ['M', 'K']),
            helper.make_tensor_value_info('B', TensorProto.INT8 if sign else TensorProto.UINT8, ['K', 'N']),
            helper.make_tensor_value_info('a_scale', TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info('b_scale', TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info('a_zero_point', TensorProto.UINT8, [1]),
            helper.make_tensor_value_info('b_zero_point', TensorProto.INT8 if sign else TensorProto.UINT8, [1]),
        ],
        [  # outputs
            helper.make_tensor_value_info('Y', TensorProto.FLOAT, ['M', 'N']),
        ])

    model = helper.make_model(graph)
    onnx.save(model, model_name)

if __name__ == "__main__":
    GenerateModel('matmul_interger_extension_int8.onnx', True)
    GenerateModel('matmul_interger_extension_uint8.onnx', False)