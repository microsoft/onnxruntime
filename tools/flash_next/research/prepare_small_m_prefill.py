from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from prepare_generic_matmul_replay import SHAPES as ORIGINAL_SHAPES


SHAPES = (*ORIGINAL_SHAPES, (10240, 324), (2560, 512), (2560, 640), (320, 10240))
ROWS = (1, 2, 16, 17)
DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-small-m-prefill/fixtures-v1")


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261030)
    for k, n in SHAPES:
        weights = rng.integers(0, 256, size=(n, k // 32, 16), dtype=np.uint8)
        scales = rng.uniform(0.001, 0.01, size=(n, k // 32)).astype(np.float16)
        for rows in ROWS:
            name = f"matmul_k{k}_n{n}_m{rows}"
            model = helper.make_model(helper.make_graph(
                [helper.make_node("MatMulNBits", ["x", "weights", "scales"], ["y"],
                                  domain="com.microsoft", K=k, N=n, bits=4, block_size=32)],
                name, [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, rows, k])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, rows, n])],
                [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(scales, "scales")]),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                ir_version=10)
            onnx.checker.check_model(model)
            onnx.save(model, DIRECTORY / (name + ".onnx"))
    print(f"Prepared {len(SHAPES) * len(ROWS)} true-M changing-input fixtures")


if __name__ == "__main__":
    main()
