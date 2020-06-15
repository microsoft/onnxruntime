import onnx
from onnx import helper
from onnx import TensorProto
from enum import Enum

def GenerateModel(model_name):
    nodes = [  # LayerNorm subgraph
        helper.make_node("DynamicQuantizeLinear", ["input"], ["a_quantized", "a_scale", "a_zp"], "DynamicQuantizeLinear"),
        helper.make_node("MatMulInteger", ["a_quantized", "b_quantized", "a_zp", "b_zp"], ["matmul_output_int32"], "MatMulInteger"),
        helper.make_node("Mul", ["a_scale", "b_scale"], ["multiplier"], "mul_right"),
        helper.make_node("Cast", ["matmul_output_int32"], ["matmul_output_float"], "cast", to=1),
        helper.make_node("Mul", ["matmul_output_float", "multiplier"], ["output"], "mul_bottom"),
    ]

    initializers = [  # initializers
        helper.make_tensor('b_quantized', TensorProto.UINT8, [2,3], [2, 4, 5, 6, 7, 8]),
        helper.make_tensor('b_zp', TensorProto.UINT8, [], [128]),
        helper.make_tensor('b_scale', TensorProto.FLOAT, [], [1.8]),
    ]

    graph = helper.make_graph(
        nodes,
        "DynamicQuantizeLinear_fusion",  #name
        [  # inputs
            helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 2]),
        ],
        [  # outputs
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 3]),
        ],
        initializers)

    model = helper.make_model(graph)
    onnx.save(model, model_name)

if __name__ == "__main__":
    GenerateModel('dynamic_quantize_matmul.onnx')