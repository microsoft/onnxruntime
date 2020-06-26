import onnx
from onnx import helper
from onnx import TensorProto
from enum import Enum

def GenerateModel(model_name):
    nodes = [  # LayerNorm subgraph
        helper.make_node("DynamicQuantizeLinear", ["input"], ["a_quantized", "a_scale", "a_zp"], "DynamicQuantizeLinear"),
        
        #MatMulInteger 1
        helper.make_node("MatMulInteger", ["a_quantized", "b_quantized_1", "a_zp", "b_zp_1"], ["matmul_output_int32_1"], "MatMulInteger_1"),
        helper.make_node("Mul", ["a_scale", "b_scale_1"], ["multiplier_1"], "mul_right_1"),
        helper.make_node("Cast", ["matmul_output_int32_1"], ["matmul_output_float_1"], "cast_1", to=1),
        helper.make_node("Mul", ["matmul_output_float_1", "multiplier_1"], ["output_1"], "mul_bottom_1"),

        #MatMulInteger 2
        helper.make_node("MatMulInteger", ["a_quantized", "b_quantized_2", "a_zp", "b_zp_2"], ["matmul_output_int32_2"], "MatMulInteger_2"),
        helper.make_node("Mul", ["a_scale", "b_scale_2"], ["multiplier_2"], "mul_right_2"),
        helper.make_node("Cast", ["matmul_output_int32_2"], ["matmul_output_float_2"], "cast_2", to=1),
        helper.make_node("Mul", ["matmul_output_float_2", "multiplier_2"], ["output_2"], "mul_bottom_2"),

    ]

    initializers = [  # initializers
        helper.make_tensor('b_quantized_1', TensorProto.UINT8, [2,3], [2, 4, 5, 6, 7, 8]),
        helper.make_tensor('b_quantized_2', TensorProto.UINT8, [2,3], [2, 4, 5, 6, 7, 8]),

        helper.make_tensor('b_zp_1', TensorProto.UINT8, [], [128]),
        helper.make_tensor('b_zp_2', TensorProto.UINT8, [], [128]),

        helper.make_tensor('b_scale_1', TensorProto.FLOAT, [], [1.8]),
        helper.make_tensor('b_scale_2', TensorProto.FLOAT, [], [1.8]),
    ]

    graph = helper.make_graph(
        nodes,
        "DynamicQuantizeLinear_fusion",  #name
        [  # inputs
            helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 2]),
        ],
        [  # outputs
            helper.make_tensor_value_info('output_1', TensorProto.FLOAT, [3, 3]),
            helper.make_tensor_value_info('output_2', TensorProto.FLOAT, [3, 3]),
        ],
        initializers)

    model = helper.make_model(graph)
    onnx.save(model, model_name)

if __name__ == "__main__":
    GenerateModel('matmul_integer_extension.onnx')