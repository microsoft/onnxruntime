import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def decode(codes):
    magnitude = codes.astype(np.uint32) & 127
    exponent = magnitude >> 3
    fraction = magnitude & 7
    values = np.where(exponent == 0, fraction * 2.0**-9,
                      (1 + fraction / 8) * np.exp2(exponent.astype(np.int32) - 7))
    return np.where(codes & 128, -values, values)


def fixture(out, rows, hidden, inter, fp32, separate, fused=False, *, seed, num_experts=None):
    rng = np.random.default_rng(seed)
    dtype = np.float32 if fp32 else np.float16
    tensor_type = TensorProto.FLOAT if fp32 else TensorProto.FLOAT16
    experts, topk = (16, 10) if hidden == 2560 else (4, 2)
    if num_experts is not None:
        experts = num_experts
    x = rng.uniform(-0.5, 0.5, (rows, hidden)).astype(dtype)
    logits = rng.uniform(-2, 2, (rows, experts)).astype(dtype)
    tensors = [numpy_helper.from_array(x, "x"), numpy_helper.from_array(logits, "router")]
    weights, biases = [], []
    shapes = [(2 * inter if fused else inter, hidden), (hidden, inter), (inter, hidden)]
    for index, (n, k) in enumerate(shapes[:3 if separate else 2]):
        codes = rng.integers(0, 256, (experts, n, k), dtype=np.uint8)
        codes[(codes & 127) == 127] = 126
        scales = rng.uniform(0.0001, 0.0005, (experts, (n + 127) // 128, (k + 127) // 128)).astype(np.float32)
        bias = rng.uniform(-0.05, 0.05, (experts, n)).astype(dtype)
        tensors.extend([
            helper.make_tensor(f"w{index}", TensorProto.FLOAT8E4M3FN, codes.shape, codes.tobytes(), raw=True),
            numpy_helper.from_array(scales, f"s{index}"),
            numpy_helper.from_array(bias, f"b{index}"),
        ])
        weights.append((codes, scales))
        biases.append(bias)
    decoded = {}

    def expert_weight(index, expert):
        key = (index, int(expert))
        if key not in decoded:
            codes, scales = weights[index]
            n, k = codes.shape[1:]
            scale = scales[expert, np.arange(n) // 128, :][:, np.arange(k) // 128]
            decoded[key] = decode(codes[expert]) * scale
        return decoded[key]

    expected = np.zeros((rows, hidden), dtype=dtype)
    for row in range(rows):
        selected = np.argsort(-logits[row].astype(np.float64), kind="stable")[:topk]
        p = np.exp(logits[row, selected].astype(np.float64) - float(logits[row, selected[0]]))
        p = (p / p.sum()).astype(dtype)
        for pos, expert in enumerate(selected):
            fc1 = (expert_weight(0, expert) @ x[row].astype(np.float64) + biases[0][expert]).astype(dtype)
            gate = fc1[::2] if fused else fc1
            gate = gate.astype(np.float64)
            if fused:
                gate = np.minimum(gate, 0.25)
                linear = np.clip(fc1[1::2].astype(np.float64), -0.25, 0.25)
                activated = gate / (1 + np.exp(-1.3 * gate)) * (linear + 0.1)
            else:
                activated = gate / (1 + np.exp(-gate))
            if separate:
                fc3 = (expert_weight(2, expert) @ x[row].astype(np.float64) + biases[2][expert]).astype(dtype)
                activated *= fc3.astype(np.float64)
            activated = activated.astype(dtype)
            fc2 = (expert_weight(1, expert) @ activated.astype(np.float64) + biases[1][expert]).astype(dtype)
            expected[row] = (expected[row] + (p[pos] * fc2).astype(dtype)).astype(dtype)
    inputs = ["x", "router", "w0", "s0", "b0", "w1", "s1", "b1"]
    inputs += ["w2", "s2", "b2"] if separate else ["", "", ""]
    attributes = dict(activation_type="silu", normalize_routing_weights=1,
                      expert_weight_bits=8, block_size=128, quant_type="fp8")
    if fused:
        attributes.update(activation_type="swiglu", swiglu_fusion=1,
                          activation_alpha=1.3, activation_beta=0.1, swiglu_limit=0.25)
    node = helper.make_node("QMoE", inputs, ["y"], domain="com.microsoft", k=topk, **attributes)
    output = helper.make_tensor_value_info("y", tensor_type, [rows, hidden])
    name = f"qmoe_m{rows}_h{hidden}_i{inter}_{'f32' if fp32 else 'f16'}_fc3{int(separate)}"
    if fused:
        name += "_fused"
    if num_experts is not None:
        name += f"_e{experts}"
    model = helper.make_model(helper.make_graph([node], name, [], [output], tensors),
                              opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 10
    model_path = out / (name + ".onnx")
    if sum(len(tensor.raw_data) for tensor in tensors) >= 2_000_000_000:
        onnx.save_model(model, model_path, save_as_external_data=True,
                        all_tensors_to_one_file=True, location=name + ".data", size_threshold=65536)
        onnx.checker.check_model(str(model_path))
    else:
        onnx.checker.check_model(model)
        onnx.save(model, model_path)
    reference = helper.make_model(helper.make_graph(
        [helper.make_node("Identity", ["expected"], ["y"])], "reference", [], [output],
        [numpy_helper.from_array(expected, "expected")]), opset_imports=[helper.make_opsetid("", 21)])
    reference.ir_version = 10
    onnx.save(reference, out / (name + "_reference.onnx"))
    print(name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--full-expert-shape", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.full_expert_shape:
        fixture(out, 8, 2560, 640, False, True, seed=493, num_experts=512)
        raise SystemExit(0)
    cases = [
        (1, 129, 131, False, False), (1, 256, 128, False, True),
        (1, 257, 259, True, True), (3, 129, 131, False, True),
        (8, 256, 128, False, False), (1, 2560, 640, False, True),
        (4, 257, 259, True, True), (9, 129, 131, False, True),
        (8, 2560, 640, False, True),
        (3, 513, 515, False, True), (4, 513, 515, True, True),
        (1, 513, 515, False, False, True), (8, 513, 515, True, False, True),
    ]
    for seed, case in enumerate(cases):
        fixture(out, *case, seed=71 + seed)
