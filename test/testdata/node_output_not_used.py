import onnx
from onnx import TensorProto, helper


def create_model_with_node_output_not_used(model_path):
    # Create graph
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])
    w = helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])

    # Dropout node (two outputs)
    dropout_node = helper.make_node(
        "Dropout",
        inputs=["X"],
        outputs=["dropout_out", "dropout_mask"],
        name="DropoutNode",
    )

    # MatMul node
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["dropout_out", "W"],
        outputs=["Y"],
        name="MatMulNode",
    )

    graph = helper.make_graph(
        nodes=[dropout_node, matmul_node],
        name="DropoutMatMulGraph",
        inputs=[x, w],
        outputs=[y],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])

    onnx.checker.check_model(model)
    onnx.save(model, model_path)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    create_model_with_node_output_not_used("node_output_not_used.onnx")
