import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main(destination=None, include_large=False):
    destination = destination or Path(__file__).parent / "qmoe-subgroup-fixtures"
    destination.mkdir(exist_ok=True)
    rng = np.random.default_rng(17391)
    cases = [(512, 10, 1, np.float16, True, pattern) for pattern in
             ("random", "equal", "plateau", "large", "minimum", "negative_infinity")]
    cases += [(experts, k, rows, dtype, normalized, "random") for experts, k, rows, dtype, normalized in
              [(256, 10, 1, np.float16, True), (511, 10, 1, np.float16, True),
               (512, 9, 1, np.float16, True), (512, 11, 1, np.float16, True),
               (512, 10, 2, np.float16, True), (512, 10, 1, np.float32, True),
               (512, 10, 1, np.float16, False)]]
    cases.append((512, 10, 1, np.float16, True, "signed_zero"))
    if include_large:
        cases.append((512, 10, 1, np.float16, True, "large_bindings"))
    manifest = []
    for index, (experts, k, rows, dtype, normalized, pattern) in enumerate(cases):
        hidden = intermediate = 64 if pattern == "large_bindings" else 32
        logits = rng.normal(0, 4, (rows, experts)).astype(dtype)
        if pattern == "equal":
            logits.fill(3)
        elif pattern == "plateau":
            logits.fill(-2)
            logits[:, 23:39] = 8
        elif pattern == "large":
            logits = np.linspace(-1000, 1000, experts, dtype=dtype)[None]
        elif pattern == "minimum":
            logits.fill(-65504)
        elif pattern == "negative_infinity":
            logits.fill(-np.inf)
            logits[:, 107] = 0
        elif pattern == "signed_zero":
            logits.fill(0)
            logits[:, ::2] = -0.0
        bias = (rng.integers(-256, 256, (experts, hidden)) / 256).astype(dtype)
        expected = np.zeros((rows, hidden), dtype=dtype)
        selected = []
        for row in range(rows):
            scores = logits[row].astype(np.float64)
            ids = np.lexsort((np.arange(experts), -scores))[:k]
            denominator_scores = scores[ids] if normalized else scores
            maximum = scores[ids[0]]
            weights = (np.exp(scores[ids] - maximum) /
                       np.exp(denominator_scores - maximum).sum()).astype(dtype)
            expected[row] = np.sum(weights.astype(np.float64)[:, None] *
                                   bias[ids].astype(np.float64), axis=0).astype(dtype)
            selected.append(ids.tolist())
        arrays = {
            "X": np.zeros((rows, hidden), dtype=dtype), "router": logits,
            "W1": np.full((experts, 2 * intermediate, hidden // 2), 0x88, dtype=np.uint8),
            "S1": np.ones((experts, 2 * intermediate, hidden // 32), dtype=dtype),
            "B1": np.ones((experts, 2 * intermediate), dtype=dtype),
            "W2": np.full((experts, hidden, intermediate // 2), 0x88, dtype=np.uint8),
            "S2": np.ones((experts, hidden, intermediate // 32), dtype=dtype),
            "B2": bias, "expected": expected,
        }
        name = f"case{index}_{pattern}"
        element_type = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
        node = helper.make_node(
            "QMoE", ["X", "router", "W1", "S1", "B1", "W2", "S2", "B2"], ["Y"],
            domain="com.microsoft", k=k, activation_type="swiglu", swiglu_fusion=1,
            expert_weight_bits=4, block_size=32, normalize_routing_weights=int(normalized))
        graph = helper.make_graph(
            [node], name, [], [helper.make_tensor_value_info(n, element_type, expected.shape)
                              for n in ("Y", "expected")],
            [numpy_helper.from_array(value, key) for key, value in arrays.items()])
        model = helper.make_model(graph, opset_imports=[
            helper.make_opsetid("", 14), helper.make_opsetid("com.microsoft", 1)], ir_version=9)
        onnx.checker.check_model(model)
        onnx.save(model, destination / (name + ".onnx"))
        manifest.append({"name": name, "experts": experts, "k": k, "rows": rows,
                         "dtype": np.dtype(dtype).name, "normalized": normalized, "selected": selected,
                         "eligible": experts == 512 and k == 10 and rows == 1 and
                         dtype == np.float16 and normalized})
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created {len(manifest)} reference fixtures; {sum(f['eligible'] for f in manifest)} eligible")


if __name__ == "__main__":
    main()
