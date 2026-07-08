"""Generate minimal ONNX test models for Go bindings testing."""

import os
import onnx
from onnx import TensorProto, helper

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_add_f32():
    """Add op: A[2,3] + B[2,3] = C[2,3], all float32."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 3])

    node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph([node], "add_graph", [A, B], [C])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = os.path.join(OUT_DIR, "add_f32.onnx")
    onnx.save(model, path)
    return path


def make_matmul_dynamic():
    """MatMul with dynamic batch: A[batch,4] @ B[4,2] = C[batch,2]."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, ["batch", 4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 2])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, ["batch", 2])

    node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph([node], "matmul_graph", [A, B], [C])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = os.path.join(OUT_DIR, "matmul_dynamic.onnx")
    onnx.save(model, path)
    return path


def make_kvconcat():
    """Concat on axis=2: past[1,2,seq,4] + new_val[1,2,1,4] = out[1,2,seq+1,4].

    Tests zero-length dim path when seq=0.
    """
    past = helper.make_tensor_value_info(
        "past", TensorProto.FLOAT, [1, 2, "seq", 4]
    )
    new_val = helper.make_tensor_value_info(
        "new_val", TensorProto.FLOAT, [1, 2, 1, 4]
    )
    out = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, [1, 2, "seq_plus_1", 4]
    )

    node = helper.make_node(
        "Concat", inputs=["past", "new_val"], outputs=["out"], axis=2
    )
    graph = helper.make_graph([node], "kvconcat_graph", [past, new_val], [out])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = os.path.join(OUT_DIR, "kvconcat.onnx")
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    for gen in [make_add_f32, make_matmul_dynamic, make_kvconcat]:
        path = gen()
        size = os.path.getsize(path)
        print(f"{os.path.basename(path):25s}  {size:4d} bytes")
