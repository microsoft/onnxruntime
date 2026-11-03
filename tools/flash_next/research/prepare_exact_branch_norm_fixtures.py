import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    destination = Path("/home/sayanshaw/repos/qwen38-int4-exact-branch-norm/fixtures")
    destination.mkdir(exist_ok=False)
    rng = np.random.default_rng(2026102901)
    manifest = []
    for hidden in (1, 31, 32, 33, 255, 256, 257, 640, 2560, 2561):
        for scale_kind in ("none", "shared", "branch"):
            for rows in (1, 3):
                shape = (rows, 4, hidden)
                x = rng.normal(0, 0.3, shape).astype(np.float16)
                tensors = [numpy_helper.from_array(x, "x")]
                inputs = ["x"]
                if scale_kind != "none":
                    scale_shape = (hidden,) if scale_kind == "shared" else (4, hidden)
                    scale = rng.uniform(0.5, 1.5, scale_shape).astype(np.float32)
                    tensors.append(numpy_helper.from_array(scale, "scale"))
                    inputs.append("scale")
                name = f"norm_h{hidden}_{scale_kind}_r{rows}"
                node = helper.make_node("BranchwiseRMSNorm", inputs, ["y"],
                                        domain="com.microsoft", epsilon=1e-5)
                model = helper.make_model(helper.make_graph(
                    [node], name, [], [helper.make_tensor_value_info("y", TensorProto.FLOAT16, shape)], tensors),
                    opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                    ir_version=10)
                onnx.checker.check_model(model)
                onnx.save(model, destination / (name + ".onnx"))
                manifest.append(dict(fixture=name + ".onnx", elements=int(x.size)))
    (destination.parent / "fixtures.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(manifest)} branch normalization fixtures")


if __name__ == "__main__":
    main()
