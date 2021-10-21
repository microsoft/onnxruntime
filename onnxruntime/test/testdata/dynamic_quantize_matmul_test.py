import onnx
from onnx import helper
from onnx import TensorProto
from enum import Enum

def GenerateModel(model_name, sign, b_zp = True, bias = False):
    nodes = [  # DynamicQuantizeMatMul subgraph
        helper.make_node("DynamicQuantizeLinear", ["A"], ["a_quantized", "a_scale", "a_zp"], "DynamicQuantizeLinear"),

        helper.make_node(
            "MatMulInteger",
            ["a_quantized", "B", "a_zp", "b_zero_point"] if b_zp else ["a_quantized", "B", "a_zp"],
            ["matmul_output_int32"],
            "MatMulInteger"),

        helper.make_node("Mul", ["a_scale", "b_scale"], ["multiplier"], "mul_right"),

        helper.make_node("Cast", ["matmul_output_int32"], ["matmul_output_float"], "cast", to=1),

        helper.make_node("Mul", ["matmul_output_float", "multiplier"], ["mul_bottom_output" if bias else "Y"], "mul_bottom"),
    ]

    inputs = [
        helper.make_tensor_value_info('A', TensorProto.FLOAT, ['M', 'K']),
        helper.make_tensor_value_info('B', TensorProto.INT8 if sign else TensorProto.UINT8, ['K', 'N']),
        helper.make_tensor_value_info('b_scale', TensorProto.FLOAT, ['C']),
        ]

    if b_zp:
        inputs.extend([helper.make_tensor_value_info('b_zero_point', TensorProto.INT8 if sign else TensorProto.UINT8, ['C'])])        

    if bias:
        nodes.extend([helper.make_node("Add", ["mul_bottom_output", "bias"], ["Y"], "add")])

        inputs.extend([helper.make_tensor_value_info('bias', TensorProto.FLOAT, ['N'])])

    graph = helper.make_graph(
        nodes,
        "DynamicQuantizeMatMul_fusion",  #name
        inputs,
        [  # outputs
            helper.make_tensor_value_info('Y', TensorProto.FLOAT, ['M', 'N']),
        ])

    model = helper.make_model(graph)
    onnx.save(model, model_name)

if __name__ == "__main__":
    GenerateModel('dynamic_quantize_matmul_int8.onnx', True)
    GenerateModel('dynamic_quantize_matmul_uint8.onnx', False)
    GenerateModel('dynamic_quantize_matmul_int8_bias.onnx', True, False, True)
    GenerateModel('dynamic_quantize_matmul_uint8_bias.onnx', False, False, True)