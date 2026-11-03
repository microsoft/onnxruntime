import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from prepare_nvfp4_fixtures import decode_scale

parser = argparse.ArgumentParser()
parser.add_argument("--quantization", choices=("nvfp4", "int4"), default="nvfp4")
parser.add_argument("--rows", type=int, default=16)
args = parser.parse_args()
if args.rows < 1:
    parser.error("--rows must be positive")
is_nvfp4 = args.quantization == "nvfp4"
source = Path("/home/sayanshaw/repos") / (
    "qwen_38_flash_nvfp4_engine_indexshare" if is_nvfp4 else "qwen38-webgpu-raw"
)
out = Path(__file__).resolve().parent / f"{args.quantization}-actual-reference"
out.mkdir(exist_ok=True)
graph = onnx.load(source / "model.onnx", load_external_data=False)
node = next(node for node in graph.graph.node if node.op_type == "QMoE")
tensors = {tensor.name: tensor for tensor in graph.graph.initializer}
attrs = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
if is_nvfp4 and (attrs.get("quant_type") != b"nvfp4" or attrs.get("weights_prepacked") != 1):
    raise ValueError("Actual-weight reference expects the original K-packed fused NVFP4 contract")
if attrs["expert_weight_bits"] != 4 or attrs["swiglu_fusion"] != 1 or attrs["block_size"] != (16 if is_nvfp4 else 32):
    raise ValueError("Actual-weight reference expects block16 NVFP4 or block32 fused INT4")
originals = [tensors[name] for name in node.input[2:] if name]
for tensor in originals:
    external = {item.key: item.value for item in tensor.external_data}
    if external:
        linked = out / external["location"]
        if not linked.exists():
            os.link(source / external["location"], linked)


def array(tensor):
    external = {item.key: item.value for item in tensor.external_data}
    if not external:
        return numpy_helper.to_array(tensor)
    dtype = {TensorProto.UINT8: np.uint8, TensorProto.FLOAT8E4M3FN: np.uint8,
             TensorProto.FLOAT16: np.float16, TensorProto.FLOAT: np.float32}[tensor.data_type]
    return np.memmap(source / external["location"], mode="r", dtype=dtype,
                     offset=int(external["offset"]), shape=tuple(tensor.dims))


rows, experts, hidden, inter = args.rows, 512, 2560, 640
rng = np.random.default_rng(719)
x = rng.uniform(-.5, .5, (rows, hidden)).astype(np.float16)
router = rng.uniform(-2, 2, (rows, experts)).astype(np.float16)
weights = [
    array(tensors[node.input[2]]).reshape(experts, 2 * inter, hidden // 2),
    array(tensors[node.input[5]]).reshape(experts, hidden, inter // 2),
]
scales = [array(tensors[node.input[3]]), array(tensors[node.input[6]])]
globals_ = [array(tensors[node.input[15]]), array(tensors[node.input[16]])] if is_nvfp4 else None
biases = [array(tensors[node.input[index]]) if node.input[index] else np.zeros((experts, n), np.float16)
          for index, n in ((4, 2 * inter), (7, hidden))]
values = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]) if is_nvfp4 else np.arange(16) - 8
cache = {}


def weight(projection, expert):
    key = (projection, int(expert))
    if key not in cache:
        packed = weights[projection][expert]
        decoded = np.empty((packed.shape[0], packed.shape[1] * 2), np.uint8)
        decoded[:, ::2] = packed & 15
        decoded[:, 1::2] = packed >> 4
        scale = decode_scale(scales[projection][expert]) if is_nvfp4 else scales[projection][expert].astype(np.float64)
        cache[key] = values[decoded] * np.repeat(scale, attrs["block_size"], axis=1)
        if globals_ is not None:
            cache[key] *= globals_[projection][expert]
    return cache[key]


expected = np.zeros_like(x)
for row in range(rows):
    selected = np.argsort(-router[row].astype(np.float64), kind="stable")[:attrs["k"]]
    p = np.exp(router[row, selected].astype(np.float64) - float(router[row, selected[0]]))
    p = (p / p.sum()).astype(np.float16)
    for rank in (np.arange(len(selected)) if is_nvfp4 else np.argsort(selected)):
        expert = selected[rank]
        fc1 = (weight(0, expert) @ x[row].astype(np.float64) + biases[0][expert]).astype(np.float16)
        gate = np.minimum(fc1[::2].astype(np.float64), attrs.get("swiglu_limit", np.finfo(np.float32).max))
        linear = np.clip(fc1[1::2].astype(np.float64),
                         -attrs.get("swiglu_limit", np.finfo(np.float32).max),
                         attrs.get("swiglu_limit", np.finfo(np.float32).max))
        active = (gate / (1 + np.exp(-attrs["activation_alpha"] * gate)) *
                  (linear + attrs["activation_beta"])).astype(np.float16)
        fc2 = (weight(1, expert) @ active.astype(np.float64) + biases[1][expert]).astype(np.float16)
        expected[row] = (expected[row] + (p[rank] * fc2).astype(np.float16)).astype(np.float16)
if not np.isfinite(expected).all():
    raise ValueError("Actual-weight fixture produced nonfinite reference output")
copy = onnx.NodeProto()
copy.CopyFrom(node)
copy.input[0], copy.input[1], copy.output[0] = "x", "router", "y"
output = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [rows, hidden])
model = helper.make_model(helper.make_graph(
    [copy], f"actual_{args.quantization}_layer0", [], [output],
    [*originals, numpy_helper.from_array(x, "x"), numpy_helper.from_array(router, "router")]),
    opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid("com.microsoft", 1)])
model.ir_version = 10
path = out / f"{args.quantization}_actual_layer0_m{rows}.onnx"
onnx.save(model, path)
onnx.checker.check_model(str(path))
reference = helper.make_model(helper.make_graph(
    [helper.make_node("Identity", ["expected"], ["y"])], "reference", [], [output],
    [numpy_helper.from_array(expected, "expected")]), opset_imports=[helper.make_opsetid("", 22)])
reference.ir_version = 10
onnx.save(reference, out / f"{args.quantization}_actual_layer0_m{rows}_reference.onnx")
print(json.dumps({"model": str(path), "rows": rows, "experts": experts,
                  "topk": attrs["k"], "weights": "unchanged original source byte ranges",
                  "max_abs_expected": float(np.abs(expected).max())}), flush=True)
