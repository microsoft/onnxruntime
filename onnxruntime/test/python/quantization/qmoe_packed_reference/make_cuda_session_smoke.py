import argparse
import json
from pathlib import Path

import ml_dtypes
import numpy as np
from onnx import TensorProto, checker, helper, numpy_helper, save


def pack_rows(values, bits):
    rows, count = values.shape
    packed = np.zeros((rows, (count * bits + 7) // 8), dtype=np.uint8)
    for element in range(count):
        for bit in range(bits):
            position = element * bits + bit
            packed[:, position // 8] |= ((values[:, element] >> bit) & 1) << (position % 8)
    return packed


def make_model(path, block, first_bits, second_bits, bfloat, separate, malformed=0, sparse=False, dense=False):
    dtype = ml_dtypes.bfloat16 if bfloat else np.float16
    tensor_type = TensorProto.BFLOAT16 if bfloat else TensorProto.FLOAT16
    experts, rows, hidden, inter = 8 if sparse else 3, 5, block * 2, block * 3
    tensors = []
    projections = []

    def add(name, values):
        tensors.append(numpy_helper.from_array(values, name))

    inputs = ((np.arange(rows * hidden).reshape(rows, hidden) % 17 - 8) * 0.04).astype(dtype)
    add("input", inputs.reshape(1, rows, hidden) if dense else np.ones((1, rows, hidden), dtype=dtype))
    add("router", np.zeros((rows, experts), dtype=dtype))
    names = [
        "input",
        "router",
        "first",
        "first_scale",
        "first_bias",
        "second",
        "second_scale",
        "second_bias",
        "",
        "",
        "",
    ]
    for prefix, outputs, features, bits in [
        ("first", 2 * inter, hidden, first_bits),
        ("second", hidden, inter, second_bits),
    ]:
        codes = np.full((experts * outputs, features), 1 << (bits - 1), dtype=np.uint8)
        if dense:
            codes = ((np.arange(experts * outputs)[:, None] * 11 + np.arange(features) * 7) % (1 << bits)).astype(
                np.uint8
            )
        packed = pack_rows(codes, bits)
        add(prefix, packed.reshape(experts, outputs, -1))
        scales = np.full((experts, outputs, features // block), 0.015625, dtype=dtype)
        if malformed == 1 and prefix == "first":
            scales.flat[0] = -1
        if malformed == 2 and prefix == "second":
            scales.flat[0] = np.inf
        add(prefix + "_scale", scales)
        bias = np.ones((experts, outputs), dtype=dtype)
        if prefix == "second":
            bias[:] = 2 if sparse else np.array([1, 3, 9], dtype=dtype)[:, None]
        if dense:
            bias[:] = ((np.arange(experts * outputs).reshape(experts, outputs) % 7 - 3) * 0.003).astype(dtype)
        projections.append(
            (
                ((codes.astype(np.float32) - (1 << (bits - 1))) * 0.015625)
                .astype(dtype)
                .astype(np.float32)
                .reshape(experts, outputs, features),
                bias.astype(np.float32),
            )
        )
        add(prefix + "_bias", bias)
        zeros = pack_rows(np.full((experts * outputs, features // block), 1 << (bits - 1), dtype=np.uint8), bits)
        if malformed == 3 and prefix == "first":
            zeros.flat[0] ^= 1
        if malformed == 4 and prefix == "first":
            zeros.flat[0] |= 0x80
        add(prefix + "_zeros", zeros.reshape(experts, outputs, -1))
        names.append(prefix + "_zeros")
    names.append("")
    if separate:
        add("mixing", np.tile(np.array([0.25, 0.5, 0.75], dtype=dtype), (rows, 1)))
        names.append("mixing")
    node = helper.make_node(
        "QMoE",
        names,
        ["output"],
        domain="com.microsoft",
        k=2,
        activation_type="swiglu",
        swiglu_fusion=1,
        normalize_routing_weights=1,
        expert_weight_bits=first_bits,
        fc2_expert_weight_bits=second_bits,
        block_size=block,
        weights_prepacked=0,
        use_sparse_mixer=int(sparse),
    )
    graph = helper.make_graph(
        [node],
        "int3-cuda-session",
        [],
        [helper.make_tensor_value_info("output", tensor_type, [1, rows, hidden])],
        tensors,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    checker.check_model(model)
    save(model, path)
    if dense:
        first_weight, first_bias = projections[0]
        second_weight, second_bias = projections[1]
        expected = np.zeros((rows, hidden), dtype=np.float32)
        for expert in (0, 1):
            projected = inputs.astype(np.float32) @ first_weight[expert].T + first_bias[expert]
            gate, up = projected[:, 0::2], projected[:, 1::2]
            activation = (gate / (1 + np.exp(-gate)) * up).astype(dtype).astype(np.float32)
            result = (activation @ second_weight[expert].T + second_bias[expert]).astype(dtype).astype(np.float32)
            expected += result * (0.25 * (expert + 1) if separate else 0.5)
        expected.astype(dtype).astype(np.float32).tofile(str(path) + ".expected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for block in (32, 64, 128):
        for first, second in ((3, 3), (3, 4), (4, 3)):
            for bfloat in (False, True):
                for separate in (False, True):
                    name = f"block{block}_{first}_{second}_{'bf16' if bfloat else 'fp16'}_mix{int(separate)}.onnx"
                    make_model(args.output / name, block, first, second, bfloat, separate)
                    cases.append([name, "1.75" if separate else "2.0"])
                    name = "dense_" + name
                    make_model(args.output / name, block, first, second, bfloat, separate, dense=True)
                    cases.append([name, "@" + str(args.output / name) + ".expected"])
    for bfloat in (False, True):
        for malformed in (1, 2, 3, 4):
            name = f"invalid{malformed}_bf{int(bfloat)}.onnx"
            make_model(args.output / name, 32, 3, 3, bfloat, False, malformed)
            cases.append([name, "invalid"])
        name = f"sparse_bf{int(bfloat)}.onnx"
        make_model(args.output / name, 32, 3, 3, bfloat, False, sparse=True)
        cases.append([name, str(2 * (1 / 8 + 1 / 7))])
    (args.output / "manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Generated {len(cases)} CUDA-only Session smoke cases")


if __name__ == "__main__":
    main()
