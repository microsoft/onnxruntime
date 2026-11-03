from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.inliner import inline_local_functions

from test_gdn_prep_fusion import fixture


def main():
    root = Path(__file__).parent / "gdn-prep-fixtures"
    root.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261026)
    for batch, length in [(1, 1), (2, 1), (1, 2), (1, 3), (1, 4), (1, 16), (1, 0)]:
        model = fixture(batch, length)
        for v in model.graph.input:
            shape = [d.dim_value for d in v.type.tensor_type.shape.dim]
            value = rng.uniform(-.5, .5, shape).astype(np.float16)
            model.graph.initializer.append(numpy_helper.from_array(value, v.name))
        del model.graph.input[:]
        for result, suffix in [(model, ""), (inline_local_functions(model), "_reference")]:
            onnx.checker.check_model(result)
            path = root / (f"gdn_prep_b{batch}_l{length}" + suffix + ".onnx")
            if path.exists():
                raise FileExistsError(path)
            onnx.save_model(result, path)
    for name, state_shape in [
        ("bad_state_width", [1, 10240, 4]),
        ("bad_state_channels", [1, 10239, 3]),
    ]:
        model = fixture(1, 1)
        for value in model.graph.input:
            shape = state_shape if value.name == "State" else [
                d.dim_value for d in value.type.tensor_type.shape.dim]
            model.graph.initializer.append(numpy_helper.from_array(
                np.zeros(shape, dtype=np.float16), value.name))
        del model.graph.input[:]
        path = root / ("gdn_prep_" + name + ".onnx")
        if path.exists():
            raise FileExistsError(path)
        onnx.save_model(model, path)
        # A valid reference lets the runner reach the fused kernel's layout guard.
        reference = root / ("gdn_prep_" + name + "_reference.onnx")
        reference.hardlink_to(root / "gdn_prep_b1_l1_reference.onnx")
    print("Created seven five-output GDN preparation fixture pairs")


if __name__ == "__main__":
    main()
