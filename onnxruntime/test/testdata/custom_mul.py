import onnx


def create_custom_mul_model():
    # === Inputs ===
    x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3, 2])
    w = onnx.helper.make_tensor_value_info("W", onnx.TensorProto.FLOAT, [3, 2])

    # === Output ===
    y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3, 2])

    # === Custom Node: Custom_Mul ===
    # Replace "Mul" with your custom op name and domain
    custom_node = onnx.helper.make_node(
        op_type="Custom_Mul",  # <-- custom op name
        inputs=["X", "W"],
        outputs=["Y"],
        domain="test",  # <-- custom domain
    )

    # === Graph ===
    graph = onnx.helper.make_graph(
        nodes=[custom_node],
        name="CustomMulGraph",
        inputs=[x, w],
        outputs=[y],
    )

    # === Model (opset version 13 or later is fine) ===
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 13),  # standard ONNX domain
            onnx.helper.make_opsetid("com.example", 1),
        ],  # your custom domain
        producer_name="custom_mul_builder",
    )

    return model


# ===== Save the Model =====
model = create_custom_mul_model()
onnx.save(model, "custom_mul.onnx")
print("Saved custom_mul.onnx")
