import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from build_qmoe_node_replay import sha


ROOT = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-matrix-probe/fixtures-v1")


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in range(8):
        random = np.random.default_rng(seed)
        a = random.uniform(-2, 2, (16, 4)).astype(np.float16)
        a[0] = np.float16(-0.0)
        a[1] = np.float16(0.0)
        codes = random.integers(0, 16, (32, 32), dtype=np.uint8)
        codes[0, :4] = 8
        codes[16:] = codes[:16]
        b = (codes[:, 0::2] | (codes[:, 1::2] << 4)).reshape(32, 1, 16)
        scales = random.uniform(0.001, 0.2, (32, 1)).astype(np.float16)
        scales[16:] = scales[:16]
        graph = helper.make_graph(
            [helper.make_node("MatMulNBits", ["a", "b", "scales"], ["y"],
                              domain="com.microsoft", K=4, N=32, bits=4, block_size=32)],
            "dot4-matrix-probe", [], [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [16, 32])],
            [numpy_helper.from_array(a, "a"), numpy_helper.from_array(b, "b"),
             numpy_helper.from_array(scales, "scales")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21),
                                                       helper.make_opsetid("com.microsoft", 1)])
        model.ir_version = 10
        onnx.checker.check_model(model)
        path = ROOT / f"seed{seed}.onnx"
        onnx.save(model, path)
        records.append(dict(path=str(path), sha256=sha(path), seed=seed))
    (ROOT / "manifest.json").write_text(json.dumps(dict(fixtures=records, source_sha256=sha(Path(__file__))),
                                                       indent=2) + "\n")


if __name__ == "__main__":
    main()
