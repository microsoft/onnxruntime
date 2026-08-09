import onnx
from onnx import TensorProto, helper

# graph with a QDQ MatMul node unit where one input is and initializer -> DQ and the other is on a path that
# contains a supported node followed by an unsupported node followed by the DQ -> MatMul.
# The DQ of the initializer is prior to the unsupported node. If the partitioning utils do not process the QDQ node
# unit together, the DQ for the initializer and the first supported node will be in the first partition, which
# incorrectly breaks up the QDQ node unit.
graph_proto = helper.make_graph(
    [
        # DQ of initializer for MatMul B input
        helper.make_node(
            "DequantizeLinear",
            inputs=["matmul_b_uint8", "scale0"],
            outputs=["dq_matmul_b"],
            name="dq_matmul_b",
        ),
        # Treat as supported
        helper.make_node(
            "Mul",
            inputs=["input:0", "scale_input"],
            outputs=["mul:0"],
            name="mul0",
        ),
        # Treat as unsupported
        helper.make_node("Cast", inputs=["mul:0"], outputs=["mul_uint8"], name="cast0", to=2),
        # DQ of MatMul A input
        helper.make_node(
            "DequantizeLinear",
            inputs=["mul_uint8", "scale1"],
            outputs=["dq_matmul_a"],
            name="dq_matmul_a",
        ),
        # MatMul
        helper.make_node(
            "MatMul",
            inputs=[
                "dq_matmul_a",
                "dq_matmul_b",
            ],
            outputs=["matmul_ab"],
            name="matmul_ab",
        ),
        # Q
        helper.make_node(
            "QuantizeLinear",
            inputs=["matmul_ab", "scale2"],
            outputs=["q_matmul_ab"],
            name="q_matmul_ab",
        ),
        # DQ for model output
        helper.make_node(
            "DequantizeLinear",
            inputs=["q_matmul_ab", "scale2"],
            outputs=["out:0"],
            name="dq_graph_output",
        ),
    ],
    "Main_graph",
    [
        helper.make_tensor_value_info("input:0", TensorProto.FLOAT, [3, 2]),
    ],
    [
        helper.make_tensor_value_info("out:0", TensorProto.FLOAT, [3, 2]),
    ],
    [
        helper.make_tensor("scale0", TensorProto.FLOAT, [1], [20.0]),
        helper.make_tensor("scale1", TensorProto.FLOAT, [1], [30.0]),
        helper.make_tensor("scale2", TensorProto.FLOAT, [1], [40.0]),
        helper.make_tensor("matmul_b_uint8", TensorProto.UINT8, [2, 2], [1, 2, 3, 4]),
        helper.make_tensor("scale_input", TensorProto.FLOAT, [2], [3.0, 4.0]),
    ],
)

model = helper.make_model(graph_proto)
onnx.checker.check_model(model, True)
onnx.save(model, "ort_github_issue_19590.onnx")
