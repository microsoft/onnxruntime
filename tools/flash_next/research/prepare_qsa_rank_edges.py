import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    cases = [
        (2048, "ties"),
        (33024, "ties"),
        (65535, "ties"),
        (65538, "ties"),
        (65539, "ties"),
        (2048, "holes"),
        (2048, "empty"),
        (2048, "first_nan"),
        (2048, "late_nan_positive_inf"),
        (2048, "late_nan_negative_inf"),
    ]
    for half in (False, True):
        dtype = np.float16 if half else np.float32
        tensor_type = TensorProto.FLOAT16 if half else TensorProto.FLOAT
        for past, mode in cases:
            total, width, heads, ratio, budget = past + 1, 4, 2, 4, 12
            query = np.ones((1, 1, heads * width), dtype=dtype)
            keys = np.zeros((1, total, width), dtype=dtype)
            keys[:, :, 0] = 1
            mask = np.ones((1, total), dtype=np.int64)
            scale = 0.0
            if mode == "holes":
                mask[:, [0, 53, total - 3]] = 0
            elif mode == "empty":
                mask.fill(0)
            elif mode == "first_nan":
                scale = float("nan")
            elif mode.startswith("late_nan"):
                scale = float("inf") if mode.endswith("positive_inf") else -float("inf")
                keys[:, ratio:2 * ratio, 0] = -1

            visible = np.flatnonzero(mask[0])
            blocks = len(visible) // ratio
            selected_blocks = min(budget // ratio, blocks)
            if mode == "first_nan":
                order = [0] if blocks else []
            else:
                valid = [i for i in range(blocks) if not (mode.startswith("late_nan") and i == 1)]
                order = valid[:selected_blocks]
            expected = np.full((1, 1, budget + ratio - 1), -1, dtype=np.int32)
            for rank, block in enumerate(order):
                expected[0, 0, rank * ratio:(rank + 1) * ratio] = visible[block * ratio:(block + 1) * ratio]
            tail = visible[blocks * ratio:]
            tail_start = selected_blocks * ratio
            expected[0, 0, tail_start:tail_start + len(tail)] = tail

            values = {
                "query": query,
                "key": keys[:, past:],
                "query_norm": np.ones(width, dtype=dtype),
                "key_norm": np.ones(width, dtype=dtype),
                "cos": np.ones((total, width), dtype=dtype),
                "sin": np.zeros((total, width), dtype=dtype),
                "mask": mask,
                "past_key": keys[:, :past],
            }
            outputs = [
                helper.make_tensor_value_info("indices", TensorProto.INT32, list(expected.shape)),
                helper.make_tensor_value_info("present", tensor_type, list(keys.shape)),
            ]
            name = f"qsa_rank_edge_past{past}_half{int(half)}_{mode}"
            node = helper.make_node(
                "SparseAttentionIndexer",
                list(values),
                ["indices", "present"],
                domain="com.microsoft",
                policy_mode="qsa",
                compress_ratio=ratio,
                token_budget=budget,
                scale=scale,
            )
            model = helper.make_model(
                helper.make_graph([node], name, [], outputs, [
                    numpy_helper.from_array(value, key) for key, value in values.items()
                ]),
                opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid("com.microsoft", 1)],
            )
            reference = helper.make_model(
                helper.make_graph([
                    helper.make_node("Identity", ["expected_indices"], ["indices"]),
                    helper.make_node("Identity", ["expected_present"], ["present"]),
                ], name + "_reference", [], outputs, [
                    numpy_helper.from_array(expected, "expected_indices"),
                    numpy_helper.from_array(keys, "expected_present"),
                ]),
                opset_imports=[helper.make_opsetid("", 22)],
            )
            for graph, suffix in ((model, ""), (reference, "_reference")):
                graph.ir_version = 10
                onnx.checker.check_model(graph)
                onnx.save(graph, output / f"{name}{suffix}.onnx")
    print(f"Prepared {2 * len(cases)} independent boundary/nonfinite fixtures in {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
