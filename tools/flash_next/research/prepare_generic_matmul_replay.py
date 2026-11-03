from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SHAPES = ((2560, 1), (2560, 16384), (2560, 7168), (2560, 6144),
          (2560, 10240), (2560, 2560), (6144, 2560), (640, 2560), (10240, 320))


def main():
    root = Path("/home/sayanshaw/repos/qwen38-int4-generic-matmul-replay/fixtures-v1")
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261029)
    for k, n in SHAPES:
        weights = rng.integers(0, 256, size=(n, k // 32, 16), dtype=np.uint8)
        scales = rng.uniform(0.001, 0.01, size=(n, k // 32)).astype(np.float16)
        for rows in (1, 2, 16):
            name = f"matmul_k{k}_n{n}_rows{rows}"
            model = helper.make_model(helper.make_graph(
                [helper.make_node("MatMulNBits", ["x", "weights", "scales"], ["y"],
                                  domain="com.microsoft", K=k, N=n, bits=4, block_size=32)],
                name, [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [rows, 1, k])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [rows, 1, n])],
                [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(scales, "scales")]),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                ir_version=10)
            onnx.checker.check_model(model)
            onnx.save(model, root / (name + ".onnx"))
    print(f"Prepared {len(SHAPES) * 3} changing-input fixtures")


if __name__ == "__main__":
    main()
