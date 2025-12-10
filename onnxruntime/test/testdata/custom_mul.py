import onnx
from onnx import helper, TensorProto

def create_custom_mul_model():
    # === Inputs ===
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])
    w = helper.make_tensor_value_info("W", TensorProto.FLOAT, [3, 2])

    # === Output ===
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2])

    # === Custom Node: Custom_Mul ===
    # Replace "Mul" with your custom op name and domain
    custom_node = helper.make_node(
        op_type="Custom_Mul",     # <-- custom op name
        inputs=["X", "W"],
        outputs=["Y"],
        domain="test"      # <-- custom domain
    )

    # === Graph ===
    graph = helper.make_graph(
        nodes=[custom_node],
        name="CustomMulGraph",
        inputs=[x, w],
        outputs=[y],
    )

    # === Model (opset version 13 or later is fine) ===
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13),   # standard ONNX domain
                       helper.make_opsetid("com.example", 1)],  # your custom domain
        producer_name="custom_mul_builder"
    )

    return model

# ===== Save the Model =====
model = create_custom_mul_model()
onnx.save(model, "custom_mul.onnx")
print("Saved custom_mul.onnx")
