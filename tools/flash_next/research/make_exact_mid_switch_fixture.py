from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    root = Path("/home/sayanshaw/repos/qwen38-int4-gdn-prep")
    original = onnx.load(root / "model.onnx", load_external_data=False)
    node = deepcopy(next(n for n in original.graph.node if n.op_type == "MatMulNBits" and
                         "/layers.0/moe/router/" in n.name))
    if len(node.input) != 3:
        raise ValueError("Expected unmodified three-input actual router projection")
    initializers = {v.name: v for v in original.graph.initializer}
    weights = [deepcopy(initializers[name]) for name in node.input[1:]]
    node.input[0] = "X"
    node.output[0] = "y"
    value = np.random.default_rng(20261027).uniform(-1, 1, (1, 1, 2560)).astype(np.float16)
    model = helper.make_model(helper.make_graph([node], "exact-mid-switch", [],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 1, 512])],
        initializer=[numpy_helper.from_array(value, "X"), *weights]),
        opset_imports=original.opset_import, ir_version=original.ir_version)
    for suffix in ("", "_reference"):
        path = root / ("exact-mid-switch" + suffix + ".onnx")
        if path.exists():
            raise FileExistsError(path)
        onnx.save_model(model, path)
        onnx.checker.check_model(str(path))
    print("Created actual-router flag-switch fixture pair with unchanged external weight descriptors")


if __name__ == "__main__":
    main()
