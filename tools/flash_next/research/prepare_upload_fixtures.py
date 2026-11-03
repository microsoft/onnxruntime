from pathlib import Path

import onnx
from onnx import TensorProto, helper

root = Path(__file__).resolve().parent / "upload-fixtures"
root.mkdir(exist_ok=True)
for dtype, name in ((TensorProto.FLOAT16, "fp16"), (TensorProto.FLOAT, "fp32")):
    nodes = [
        helper.make_node("Add", ["x", "x"], ["twice"]),
        helper.make_node("Shape", ["twice"], ["shape"]),
        helper.make_node("Cast", ["shape"], ["width"], to=dtype),
        helper.make_node("Add", ["twice", "width"], ["y"]),
    ]
    model = helper.make_model(helper.make_graph(
        nodes, f"ordered_upload_{name}",
        [helper.make_tensor_value_info("x", dtype, ["width"])],
        [helper.make_tensor_value_info("y", dtype, ["width"])]),
        opset_imports=[helper.make_opsetid("", 22)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, root / f"{name}.onnx")
print(root)
