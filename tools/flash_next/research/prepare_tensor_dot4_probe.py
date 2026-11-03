import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from build_qmoe_node_replay import sha


def main(root=Path("/home/sayanshaw/repos/qwen38-int4-tensor-dot4-probe/fixtures-v1"), terms=4):
    if type(terms) is not int or terms not in (1, 4):
        raise ValueError("Tensor arithmetic fixtures require one or four nonzero weights")
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261031)
    rows = []
    for offset in (0, 4, 12):
        packed = np.full((2560, 80, 16), 0x88, dtype=np.uint8)
        values = rng.integers(0, 16, (2560, 4), dtype=np.uint8)
        for index in range(terms):
            k = offset + index
            shift = (k % 2) * 4
            mask = np.uint8(0xf0 if shift == 0 else 0x0f)
            packed[:, 0, k // 2] = (packed[:, 0, k // 2] & mask) | (values[:, index] << shift)
        scales = rng.uniform(0.001, 0.01, (2560, 80)).astype(np.float16)
        name = f"dot4_offset{offset}"
        model = helper.make_model(helper.make_graph(
            [helper.make_node("MatMulNBits", ["x", "weights", "scales"], ["y"],
                              domain="com.microsoft", K=2560, N=2560, bits=4, block_size=32)],
            name, [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 1, 2560])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 1, 2560])],
            [numpy_helper.from_array(packed, "weights"), numpy_helper.from_array(scales, "scales")]),
            opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            ir_version=10)
        onnx.checker.check_model(model)
        path = root / (name + ".onnx")
        onnx.save(model, path)
        rows.append(dict(fixture=path.name, sha256=sha(path), nonzero_k_positions=list(range(offset, offset + terms))))
    with (root / "manifest.json").open("x") as stream:
        stream.write(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
