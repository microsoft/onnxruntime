# repro_rmsnorm_scalar_and_broadcast.py
import onnx
import numpy as np
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

X = np.arange(30, dtype=np.float32).reshape(2,5,3)

def make_model(scale_array, name):
    node = helper.make_node(
        "RMSNormalization",
        inputs=["X", "Scale"],
        outputs=["Y"],
        axis=2,
        epsilon=1e-5,
    )
    X_vi    = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2,5,3])
    Y_vi    = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2,5,3])
    Scale_t = numpy_helper.from_array(scale_array, name="Scale")

    graph = helper.make_graph(
        nodes=[node],
        name=name,
        inputs=[X_vi],
        outputs=[Y_vi],
        initializer=[Scale_t],
    )
    opset = helper.make_operatorsetid("", 23)
    model = helper.make_model(graph, opset_imports=[opset])
    onnx.checker.check_model(model)
    return model

def run_case(desc, scale_arr):
    print(f"\n=== {desc} | scale.shape={scale_arr.shape} ===")
    model = make_model(scale_arr, f"model_{desc}")
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    try:
        y = sess.run(None, {"X": X})
        print("✓ Ran OK. Output shape:", y[0].shape)

    except Exception as e:
        print("✗ Failed (this is the bug we expect):")
        print(e)

print("Run start")

run_case("scalar_scale", np.array(1.5, dtype=np.float32))

run_case("broadcast_1x1x1", np.ones((1,1,1), dtype=np.float32))

run_case("vector_len3", np.array([1.5,1.5,1.5], dtype=np.float32))
