import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    output = Path(__file__).parent / "matmul-n96-fixtures"
    output.mkdir(exist_ok=True)
    rng = np.random.default_rng(8675309)
    fixtures = []
    cases = [
        (2560, 96, (2560,), False, np.float16),
        (2560, 96, (1, 2560), False, np.float16),
        (2560, 96, (1, 1, 2560), False, np.float16),
        (2560, 96, (1, 1, 1, 2560), True, np.float16),
        (2560, 96, (1, 1, 2560), False, np.float32),
        (2560, 96, (2, 1, 2560), False, np.float16),
        (2559, 96, (1, 1, 2559), False, np.float16),
        (2561, 96, (1, 1, 2561), False, np.float16),
        (2048, 96, (1, 1, 2048), False, np.float16),
        (8192, 96, (1, 1, 8192), False, np.float16),
        (2560, 64, (1, 1, 2560), False, np.float16),
        (2560, 95, (1, 1, 2560), False, np.float16),
        (2560, 100, (1, 1, 2560), False, np.float16),
    ]
    for index, (k, n, shape, batched, dtype) in enumerate(cases):
        eligible = bool(k == 2560 and n == 96 and np.prod(shape[:-1]) == 1 and dtype == np.float16)
        patterns = ("random", "zeros", "cancellation", "dyadic") if eligible else ("dyadic",)
        for pattern in patterns:
            a = np.full(shape, 0.125, dtype=dtype)
            b = np.broadcast_to(np.where(np.arange(k)[:, None] % 2 == 0, 0.125, -0.125), (k, n)).astype(dtype).copy()
            b[0] += 0.0625
            if pattern == "random":
                a = rng.normal(0, 0.125, shape).astype(dtype)
                b = rng.normal(0, 0.125, (k, n)).astype(dtype)
            elif pattern == "zeros":
                a.fill(0)
            elif pattern == "cancellation":
                a.fill(1)
                b[:k // 2] = 512
                b[k // 2:] = -512
                b[0] = 513
            if batched:
                b = b[None]
            expected = (a.astype(np.float64) @ b.astype(np.float64)).astype(dtype)
            kind = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
            name = f"case{index}_{pattern}"
            graph = helper.make_graph(
                [helper.make_node("MatMul", ["A", "B"], ["Y"])], name, [],
                [helper.make_tensor_value_info("Y", kind, expected.shape),
                 helper.make_tensor_value_info("expected", kind, expected.shape)],
                [numpy_helper.from_array(a, "A"), numpy_helper.from_array(b, "B"),
                 numpy_helper.from_array(expected, "expected")],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=9)
            helper.set_model_props(model, {"reference_check": "candidate_close" if pattern in ("random", "cancellation") else "exact"})
            onnx.checker.check_model(model)
            onnx.save(model, output / (name + ".onnx"))
            fixtures.append({"name": name, "a_shape": a.shape, "b_shape": b.shape,
                             "eligible": eligible, "pattern": pattern})
    (output / "manifest.json").write_text(json.dumps(fixtures, indent=2) + "\n")
    print(f"Created {len(fixtures)} checked fixtures with independent float64 references")


if __name__ == "__main__":
    main()
