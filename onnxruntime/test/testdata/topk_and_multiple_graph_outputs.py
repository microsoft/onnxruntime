import onnx
from onnx import TensorProto, helper


def create_model_with_topk_graph_output(model_path):
    # ======================
    # ---- Inputs ----
    # ======================
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N"])

    # ======================
    # ---- Initializers ----
    # ======================
    k = helper.make_tensor("K", TensorProto.INT64, dims=[1], vals=[300])
    zero = helper.make_tensor("zero", TensorProto.INT64, dims=[], vals=[0])
    twenty_six = helper.make_tensor("twenty_six", TensorProto.INT64, dims=[], vals=[26])

    # ======================
    # ---- Nodes ----
    # ======================
    topk_node = helper.make_node(
        "TopK",
        inputs=["input", "K"],
        outputs=["scores", "topk_indices"],
        name="TopK",
    )

    less_node = helper.make_node(
        "Less",
        inputs=["topk_indices", "zero"],
        outputs=["Less_output_0"],
        name="Less",
    )

    div_node = helper.make_node(
        "Div",
        inputs=["topk_indices", "twenty_six"],
        outputs=["Div_17_output_0"],
        name="Div",
    )

    mod_node = helper.make_node(
        "Mod",
        inputs=["topk_indices", "twenty_six"],
        outputs=["labels"],
        name="Mod",
    )

    # =========================
    # ---- Graph Outputs ----
    # =========================
    scores_out = helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["K"])
    less_out = helper.make_tensor_value_info("Less_output_0", TensorProto.BOOL, ["K"])
    div_out = helper.make_tensor_value_info("Div_17_output_0", TensorProto.INT64, ["K"])
    labels_out = helper.make_tensor_value_info("labels", TensorProto.INT64, ["K"])

    # ======================
    # ---- Graph ----
    # ======================
    graph = helper.make_graph(
        nodes=[topk_node, less_node, div_node, mod_node],
        name="TopKGraph",
        inputs=[input_tensor],
        outputs=[scores_out, less_out, div_out, labels_out],
        initializer=[k, zero, twenty_six],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])

    # Validate + Save
    onnx.checker.check_model(model)
    onnx.save(model, model_path)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    create_model_with_topk_graph_output("topk_and_multiple_graph_outputs.onnx")
