import onnx
from onnx import OperatorSetIdProto, TensorProto, helper

onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
# The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
onnxdomain.domain = ""
msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"
opsets = [onnxdomain, msdomain]


def save(model_path, nodes, inputs, outputs, initializers):
    graph = helper.make_graph(nodes, "GemmSumTest", inputs, outputs, initializers)

    model = helper.make_model(graph, opset_imports=opsets, producer_name="onnxruntime-test")

    print(model_path)
    onnx.save(model, model_path)


def gen_gemm_sum_basic(model_path):
    nodes = [
        helper.make_node(op_type="Gemm", inputs=["A", "B"], outputs=["tp0"]),
        helper.make_node(op_type="Sum", inputs=["tp0", "C"], outputs=["output"]),
    ]

    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", "K"]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ["K", "N"]),
        helper.make_tensor_value_info("C", TensorProto.FLOAT, ["M", "N"]),
    ]

    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["M", "N"])]

    save(model_path, nodes, inputs, outputs, initializers=[])


def gen_gemm_sum_attributes(model_path):
    nodes = [
        helper.make_node(
            op_type="Gemm",
            inputs=["A", "B"],
            outputs=["tp0"],
            alpha=3.5,
            beta=6.25,
            transA=False,
            transB=True,
        ),
        helper.make_node(op_type="Sum", inputs=["tp0", "C"], outputs=["output"]),
    ]

    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", "K"]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ["N", "K"]),
        helper.make_tensor_value_info("C", TensorProto.FLOAT, ["M", "N"]),
    ]

    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["M", "N"])]

    save(model_path, nodes, inputs, outputs, initializers=[])


def gen_gemm_sum_internal_nodes(model_path):
    nodes = [
        helper.make_node(op_type="Identity", inputs=["A"], outputs=["tp0"]),
        helper.make_node(op_type="Identity", inputs=["B"], outputs=["tp1"]),
        helper.make_node(op_type="Gemm", inputs=["tp0", "tp1"], outputs=["tp2"]),
        helper.make_node(op_type="Identity", inputs=["C"], outputs=["tp3"]),
        helper.make_node(op_type="Sum", inputs=["tp2", "tp3"], outputs=["tp4"]),
        helper.make_node(op_type="Identity", inputs=["tp4"], outputs=["output"]),
    ]

    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", "K"]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ["K", "N"]),
        helper.make_tensor_value_info("C", TensorProto.FLOAT, ["M", "N"]),
    ]

    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["M", "N"])]

    save(model_path, nodes, inputs, outputs, initializers=[])


def gen_gemm_sum_no_fusion_c_used(model_path):
    nodes = [
        helper.make_node(op_type="Gemm", inputs=["A", "B", "C"], outputs=["tp0"]),
        helper.make_node(op_type="Sum", inputs=["tp0", "D"], outputs=["output"]),
    ]

    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", "K"]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ["K", "N"]),
        helper.make_tensor_value_info("C", TensorProto.FLOAT, ["M", "N"]),
        helper.make_tensor_value_info("D", TensorProto.FLOAT, ["M", "N"]),
    ]

    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["M", "N"])]

    save(model_path, nodes, inputs, outputs, initializers=[])


def gen_gemm_sum_no_fusion_sum_multiple_inputs(model_path):
    nodes = [
        helper.make_node(op_type="Gemm", inputs=["A", "B"], outputs=["tp0"]),
        helper.make_node(op_type="Sum", inputs=["tp0", "C", "D"], outputs=["output"]),
    ]

    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", "K"]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ["K", "N"]),
        helper.make_tensor_value_info("C", TensorProto.FLOAT, ["M", "N"]),
        helper.make_tensor_value_info("D", TensorProto.FLOAT, ["M", "N"]),
    ]

    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["M", "N"])]

    save(model_path, nodes, inputs, outputs, initializers=[])


def gen_gemm_sum_fusion_broadcast(model_path):
    nodes = [
        helper.make_node(op_type="Gemm", inputs=["A", "B"], outputs=["tp0"]),
        helper.make_node(op_type="Sum", inputs=["tp0", "C"], outputs=["output"]),
    ]

    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", "K"]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ["K", "N"]),
        helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, "N"]),
    ]

    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["M", "N"])]

    save(model_path, nodes, inputs, outputs, initializers=[])


def gen_gemm_sum_no_fusion_broadcast_failure(model_path):
    nodes = [
        helper.make_node(op_type="Gemm", inputs=["A", "B"], outputs=["tp0"]),
        helper.make_node(op_type="Sum", inputs=["tp0", "C"], outputs=["output"]),
    ]

    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", "K"]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ["K", "N"]),
        # should work with multidirectional broadcast as second argument, but not unidirectional.
        helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, "M", "N"]),
    ]

    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, "M", "N"])]

    save(model_path, nodes, inputs, outputs, initializers=[])


def gen_gemm_sum_no_fusion_original_gemm_output_used(model_path):
    nodes = [
        helper.make_node(op_type="Gemm", inputs=["A", "B"], outputs=["tp0"]),
        helper.make_node(op_type="Sum", inputs=["tp0", "C"], outputs=["output"]),
    ]

    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", "K"]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ["K", "N"]),
        helper.make_tensor_value_info("C", TensorProto.FLOAT, ["M", "N"]),
    ]

    outputs = [
        helper.make_tensor_value_info("tp0", TensorProto.FLOAT, ["M", "N"]),
        helper.make_tensor_value_info("output", TensorProto.FLOAT, ["M", "N"]),
    ]

    save(model_path, nodes, inputs, outputs, initializers=[])


gen_gemm_sum_basic("gemm_sum_basic.onnx")
gen_gemm_sum_attributes("gemm_sum_attributes.onnx")
gen_gemm_sum_internal_nodes("gemm_sum_internal_nodes.onnx")
gen_gemm_sum_no_fusion_c_used("gemm_sum_no_fusion_c_used.onnx")
gen_gemm_sum_no_fusion_sum_multiple_inputs("gemm_sum_no_fusion_sum_multiple_inputs.onnx")
gen_gemm_sum_fusion_broadcast("gemm_sum_fusion_broadcast.onnx")
gen_gemm_sum_no_fusion_broadcast_failure("gemm_sum_no_fusion_broadcast_failure.onnx")
gen_gemm_sum_no_fusion_original_gemm_output_used("gemm_sum_no_fusion_original_gemm_output_used.onnx")
