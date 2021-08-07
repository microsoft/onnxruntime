import onnx
from onnx import helper
from onnx import TensorProto
from onnx import OperatorSetIdProto

onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
# The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
onnxdomain.domain = ""
msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"
opsets = [onnxdomain, msdomain]


def save(model_path, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(
        nodes,
        "TransposeGemmTest",
        inputs, outputs, initializers)

    model = helper.make_model(
        graph, opset_imports=opsets, producer_name="onnxruntime-test")

    onnx.save(model, model_path)

# (A')'B' = AB'
def gen_gemm_2inputs_transposed(model_path):
    nodes = [
        helper.make_node("Transpose", ["A"], ["tp0"], "TransposeA"),
        helper.make_node("Transpose", ["B"], ["tp1"], "TransposeB"),        
        helper.make_node("Gemm", ["tp0", "tp1"], ["output"], "Gemm", alpha=3.0, transA=1)
    ]

    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ['M', 'K']),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ['N', 'K'])
    ]

    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, ['M', 'N'])
    ]

    save(model_path, nodes, inputs, outputs, [])

# (A'B)' = B'A
def gen_gemm_output_transposed(model_path):
    nodes = [
        helper.make_node("Gemm", ["A", "B"], ["out"], "Gemm", alpha=3.0, transA=1),
        helper.make_node("Transpose", ["out"], ["output"], "TransposeOut"),
    ]

    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ['K', 'M']),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ['K', 'N'])
    ]

    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, ['N', 'M'])
    ]

    save(model_path, nodes, inputs, outputs, [])

# ((A')'B')' = BA'
def gen_gemm_inputs_output_transposed(model_path):
    nodes = [
        helper.make_node("Transpose", ["A"], ["tp0"], "TransposeA"),
        helper.make_node("Transpose", ["B"], ["tp1"], "TransposeB"),        
        helper.make_node("Gemm", ["tp0", "tp1"], ["out"], "Gemm", alpha=3.0, transA=1),
        helper.make_node("Transpose", ["out"], ["output"], "TransposeOut"),
    ]

    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ['M', 'K']),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ['N', 'K'])
    ]

    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, ['N', 'M'])
    ]

    save(model_path, nodes, inputs, outputs, [])

gen_gemm_2inputs_transposed("gemm_transpose_2inputs_transposed.onnx")
gen_gemm_output_transposed("gemm_transpose_output_transposed.onnx")
gen_gemm_inputs_output_transposed("gemm_transpose_inputs_output_transposed.onnx")

