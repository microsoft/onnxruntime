from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def fixture(out, rows, fp32, fused=True, separate=False, topk=3, experts=5, normalize=True, zeros=False):
    rng = np.random.default_rng(510 + rows + int(fp32))
    dtype = np.float32 if fp32 else np.float16
    tensor_type = TensorProto.FLOAT if fp32 else TensorProto.FLOAT16
    hidden, inter = 128, 64
    x = rng.uniform(-.5, .5, (rows, hidden)).astype(dtype)
    if fp32:
        # The unchanged DP4A baseline quantizes activations per 128-element block.
        # Use exactly quantizable random inputs to test both paths against the same ideal reference.
        x = (rng.integers(-127, 128, (rows, hidden)) / 256).astype(dtype)
        x[:, 0] = 127 / 256
    router = rng.uniform(-2, 2, (rows, experts)).astype(dtype)
    if rows == 3:
        router[0] = 0
    tensors = [numpy_helper.from_array(x, "x"), numpy_helper.from_array(router, "router")]
    weights, biases, input_names = [], [], ["x", "router"]
    for index, (n, k) in enumerate([(2 * inter if fused else inter, hidden), (hidden, inter)] +
                                  ([(inter, hidden)] if separate else [])):
        codes = rng.integers(0, 16, (experts, n, k), dtype=np.uint8)
        scales = rng.uniform(.005, .03, (experts, n, k // 32)).astype(dtype)
        zp = np.full((experts, n, k // 32), 7 if zeros else 8, dtype=np.uint8)
        packed = codes[..., ::2] | (codes[..., 1::2] << 4)
        tensors.extend([numpy_helper.from_array(packed, f"w{index}"),
                        numpy_helper.from_array(scales, f"s{index}")])
        bias = rng.uniform(-.05, .05, (experts, n)).astype(dtype)
        tensors.append(numpy_helper.from_array(bias, f"b{index}"))
        input_names.extend([f"w{index}", f"s{index}", f"b{index}"])
        weights.append((codes.astype(np.float64) - np.repeat(zp, 32, axis=2)) *
                       np.repeat(scales.astype(np.float64), 32, axis=2))
        biases.append(bias)
        if zeros:
            packed_zp = zp[..., ::2] | (zp[..., 1::2] << 4)
            tensors.append(numpy_helper.from_array(packed_zp, f"z{index}"))
    if zeros:
        if not separate:
            input_names.extend(["", "", ""])
        input_names.extend(["z0", "z1"] + (["z2"] if separate else [""]))
    expected = np.zeros_like(x)
    for row in range(rows):
        selected = np.argsort(-router[row].astype(np.float64), kind="stable")[:topk]
        logits = router[row, selected if normalize else np.arange(experts)].astype(np.float64)
        denominator = np.exp(logits - logits.max()).sum()
        probabilities = (np.exp(router[row, selected].astype(np.float64) - logits.max()) / denominator).astype(dtype)
        for rank in np.argsort(selected):
            expert = selected[rank]
            fc1 = (weights[0][expert] @ x[row].astype(np.float64) + biases[0][expert]).astype(dtype)
            gate = (fc1[::2] if fused else fc1).astype(np.float64)
            if fused:
                gate = np.minimum(gate, .5)
                linear = np.clip(fc1[1::2].astype(np.float64), -.5, .5)
                active = gate / (1 + np.exp(-1.3 * gate)) * (linear + .1)
            else:
                active = gate / (1 + np.exp(-gate))
                if separate:
                    fc3 = (weights[2][expert] @ x[row].astype(np.float64) + biases[2][expert]).astype(dtype)
                    active *= fc3.astype(np.float64)
            active = active.astype(dtype)
            fc2 = (weights[1][expert] @ active.astype(np.float64) + biases[1][expert]).astype(dtype)
            expected[row] = (expected[row] + (probabilities[rank] * fc2).astype(dtype)).astype(dtype)
    attrs = dict(k=topk, activation_type="silu", expert_weight_bits=4, block_size=32,
                 weights_prepacked=0, normalize_routing_weights=int(normalize))
    if fused:
        attrs.update(activation_type="swiglu", swiglu_fusion=1, activation_alpha=1.3,
                     activation_beta=.1, swiglu_limit=.5)
    name = f"int4_m{rows}_f32{int(fp32)}_fused{int(fused)}_fc3{int(separate)}_k{topk}_norm{int(normalize)}_zp{int(zeros)}"
    output = helper.make_tensor_value_info("y", tensor_type, [rows, hidden])
    model = helper.make_model(helper.make_graph(
        [helper.make_node("QMoE", input_names, ["y"], domain="com.microsoft", **attrs)],
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


out = Path(__file__).resolve().parent / "int4-fixtures"
out.mkdir(exist_ok=True)
for fp32 in (False, True):
    for rows in (1, 3, 8, 16, 17):
        fixture(out, rows, fp32)
    fixture(out, 3, fp32, fused=False, separate=True)
    fixture(out, 16, fp32, topk=16, experts=16)
    fixture(out, 16, fp32, topk=17, experts=17)
    fixture(out, 3, fp32, normalize=False)
    fixture(out, 4, fp32, zeros=True)
