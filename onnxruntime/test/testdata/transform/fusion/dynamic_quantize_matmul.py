import onnx
from onnx import helper
from onnx import TensorProto
from enum import Enum

def GenerateModel(model_name, b_has_zp = True, has_bias = False, bias_ND = False):
    mul_output = "Mul_output" if has_bias else "output"
    nodes = [ # construct graph
        helper.make_node("DynamicQuantizeLinear", ["input"], ["a_quantized", "a_scale", "a_zp"], "DynamicQuantizeLinear"),
        helper.make_node(
            "MatMulInteger",
            ["a_quantized", "b_quantized", "a_zp", "b_zp"] if b_has_zp else ["a_quantized", "b_quantized", "a_zp"], 
            ["matmul_output_int32"],
            "MatMulInteger"),
        helper.make_node("Mul", ["a_scale", "b_scale"], ["multiplier"], "mul_right"),
        helper.make_node("Cast", ["matmul_output_int32"], ["matmul_output_float"], "cast", to=1),
        helper.make_node("Mul", ["matmul_output_float", "multiplier"], [mul_output], "mul_bottom"),
    ]

    if has_bias:
        nodes.extend([helper.make_node("Add", [mul_output, "bias"], ["output"], "bias_add")])

    initializers = [  # initializers
        helper.make_tensor('b_quantized', TensorProto.UINT8, [2,3], [2, 4, 5, 6, 7, 8]),
        helper.make_tensor('b_scale', TensorProto.FLOAT, [], [1.8]),
    ]

    if b_has_zp:
        initializers.extend([  # initializers
            helper.make_tensor('b_zp', TensorProto.UINT8, [], [128]),
        ])

    if has_bias:
        if bias_ND:
            initializers.extend([  # initializers
                helper.make_tensor('bias', TensorProto.FLOAT, [3, 3], [3.0, 4.0, 6.0, 3.0, 4.0, 6.0, 3.0, 4.0, 5.0]),
            ])
        else:
            initializers.extend([  # initializers
                helper.make_tensor('bias', TensorProto.FLOAT, [3], [3.0, 4.0, 5.0]),
            ])

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
    GenerateModel('dynamic_quantize_matmul_bias.onnx', True, True)
    GenerateModel('dynamic_quantize_matmul_bias_b_no_zp.onnx', False, True)
    GenerateModel('dynamic_quantize_matmul_bias_ND.onnx', False, True, True)