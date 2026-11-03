import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def decode_scale(codes):
    exponent = (codes.astype(np.uint32) & 127) >> 3
    fraction = codes & 7
    return np.where(
        exponent == 0, fraction * 2.0**-9,
        (1 + fraction / 8) * np.exp2(exponent.astype(np.int32) - 7),
    ) * np.where(codes & 128, -1, 1)


def fixture(out, rows, hidden, inter, fp32, k_packed, fused, bias, experts, topk, seed):
    rng = np.random.default_rng(seed)
    dtype = np.float32 if fp32 else np.float16
    tensor_type = TensorProto.FLOAT if fp32 else TensorProto.FLOAT16
    x = rng.uniform(-0.5, 0.5, (rows, hidden)).astype(dtype)
    router = rng.uniform(-2, 2, (rows, experts)).astype(dtype)
    tensors = [numpy_helper.from_array(x, "x"), numpy_helper.from_array(router, "router")]
    decoded, weights, biases = {}, [], []
    fp4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    for index, (n, k) in enumerate([(2 * inter if fused else inter, hidden), (hidden, inter)]):
        codes = rng.integers(0, 16, (experts, n, k), dtype=np.uint8)
        scales = rng.integers(16, 57, (experts, n, k // 16), dtype=np.uint8)
        globals_ = rng.uniform(.02, .05, experts).astype(np.float32)
        packed = codes if k_packed else codes.transpose(0, 2, 1)
        packed = np.ascontiguousarray(packed[..., ::2] | (packed[..., 1::2] << 4))
        tensors.extend([
            helper.make_tensor(f"w{index}", TensorProto.UINT8, [experts, k, n // 2],
                               packed.tobytes(), raw=True),
            helper.make_tensor(f"s{index}", TensorProto.FLOAT8E4M3FN, scales.shape,
                               scales.tobytes(), raw=True),
            numpy_helper.from_array(globals_, f"g{index}"),
        ])
        weights.append((codes, scales, globals_))
        b = rng.uniform(-.05, .05, (experts, n)).astype(dtype) if bias else np.zeros((experts, n), dtype)
        biases.append(b)
        if bias:
            tensors.append(numpy_helper.from_array(b, f"b{index}"))

    def weight(index, expert):
        key = (index, int(expert))
        if key not in decoded:
            codes, scales, globals_ = weights[index]
            decoded[key] = fp4[codes[expert]] * np.repeat(decode_scale(scales[expert]), 16, axis=1) * globals_[expert]
        return decoded[key]

    expected = np.zeros_like(x)
    for row in range(rows):
        selected = np.argsort(-router[row].astype(np.float64), kind="stable")[:topk]
        p = np.exp(router[row, selected].astype(np.float64) - float(router[row, selected[0]]))
        p = (p / p.sum()).astype(dtype)
        for rank, expert in enumerate(selected):
            fc1 = (weight(0, expert) @ x[row].astype(np.float64) + biases[0][expert]).astype(dtype)
            gate = (fc1[::2] if fused else fc1).astype(np.float64)
            if fused:
                gate = np.minimum(gate, .25)
                linear = np.clip(fc1[1::2].astype(np.float64), -.25, .25)
                active = gate / (1 + np.exp(-1.3 * gate)) * (linear + .1)
            else:
                active = gate / (1 + np.exp(-gate))
            active = active.astype(dtype)
            fc2 = (weight(1, expert) @ active.astype(np.float64) + biases[1][expert]).astype(dtype)
            expected[row] = (expected[row] + (p[rank] * fc2).astype(dtype)).astype(dtype)
    inputs = ["x", "router", "w0", "s0", "b0" if bias else "", "w1", "s1", "b1" if bias else ""]
    inputs += [""] * 7 + ["g0", "g1"]
    attrs = dict(activation_type="silu", normalize_routing_weights=1, expert_weight_bits=4,
                 block_size=16, quant_type="nvfp4", weights_prepacked=int(k_packed))
    if fused:
        attrs.update(activation_type="swiglu", swiglu_fusion=1,
                     activation_alpha=1.3, activation_beta=.1, swiglu_limit=.25)
    name = f"nvfp4_m{rows}_h{hidden}_i{inter}_e{experts}_f32{int(fp32)}_kp{int(k_packed)}_fused{int(fused)}_bias{int(bias)}"
    output = helper.make_tensor_value_info("y", tensor_type, [rows, hidden])
    model = helper.make_model(helper.make_graph(
        [helper.make_node("QMoE", inputs, ["y"], domain="com.microsoft", k=topk, **attrs)],
        name, [], [output], tensors),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, out / (name + ".onnx"))
    reference = helper.make_model(helper.make_graph(
        [helper.make_node("Identity", ["expected"], ["y"])], "reference", [], [output],
        [numpy_helper.from_array(expected, "expected")]), opset_imports=[helper.make_opsetid("", 21)])
    reference.ir_version = 10
    onnx.save(reference, out / (name + "_reference.onnx"))
    print(name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--full-expert-shape", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.full_expert_shape:
        fixture(args.output, 8, 2560, 640, False, True, True, True, 512, 10, 493)
    else:
        for k_packed in (False, True):
            for index, (rows, hidden, inter, fp32, fused, bias) in enumerate([
                (1, 16, 32, False, False, False),
                (1, 256, 128, True, True, True),
                (3, 528, 272, False, True, True),
                (8, 256, 128, False, True, False),
                (9, 16, 32, True, False, True),
                (16, 256, 128, False, True, True),
                (128, 256, 128, False, True, True),
                (129, 16, 32, True, False, True),
                (8, 2560, 640, False, True, True),
            ]):
                fixture(args.output, rows, hidden, inter, fp32, k_packed, fused, bias,
                        16 if hidden == 2560 else 4, 10 if hidden == 2560 else 2, 71 + index)
