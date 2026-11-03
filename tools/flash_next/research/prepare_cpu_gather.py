import os
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from onnx.external_data_helper import load_external_data_for_tensor

source = Path("/home/sayanshaw/repos/qwen38-webgpu-raw")
root = Path(__file__).resolve().parent / "cpu-gather-fixture"
root.mkdir(exist_ok=True)
graph = onnx.load(source / "model.onnx", load_external_data=False).graph
node = next(node for node in graph.node if node.op_type == "GatherBlockQuantized")
tensors = {tensor.name: tensor for tensor in graph.initializer}
initializers = []
for index in (0, 2):
    tensor = TensorProto()
    tensor.CopyFrom(tensors[node.input[index]])
    if index == 0:
        for entry in tensor.external_data:
            if entry.key == "location":
                link = root / entry.value
                if not link.exists():
                    os.link(source / entry.value, link)
    else:
        load_external_data_for_tensor(tensor, str(source))
        tensor.ClearField("external_data")
        tensor.data_location = TensorProto.DEFAULT
    initializers.append(tensor)
node.input[1], node.output[0] = "indices", "y"
model = helper.make_model(helper.make_graph(
    [node], "original_cpu_ple_gather",
    [helper.make_tensor_value_info("indices", TensorProto.INT64, ["rows", 16])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT16, ["rows", 16, 160])],
    initializers), opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid("com.microsoft", 1)])
model.ir_version = 10
onnx.save(model, root / "model.onnx")
onnx.checker.check_model(str(root / "model.onnx"))
print(root / "model.onnx")
