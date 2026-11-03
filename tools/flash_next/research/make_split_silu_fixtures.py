import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from fuse_split_silu import fuse
from test_split_silu_fusion import fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "split-silu-fixtures"
    root.mkdir(exist_ok=True)
    if args.negative_only:
        source = root / "split_silu_m1_p320_s4_a0.25.onnx"
        for name, sizes in (("bad_total", [320, 5]), ("empty_prefix", [0, 324])):
            model = onnx.load(source)
            tensor = next(v for v in model.graph.initializer if v.name == "Sizes")
            tensor.CopyFrom(numpy_helper.from_array(np.array(sizes, np.int64), "Sizes"))
            target = root / (name + ".onnx")
            reference = root / (name + "_reference.onnx")
            if target.exists() or reference.exists():
                raise FileExistsError(name)
            onnx.save_model(model, target)
            onnx.save_model(onnx.load(root / "split_silu_m1_p320_s4_a0.25_reference.onnx"), reference)
        print("Created two unsupported split-size cases")
        return
    rng = np.random.default_rng(20261026)
    for rows, prefix, suffix, alpha in (
        (1, 320, 4, .25), (2, 320, 4, .25), (16, 320, 4, .25),
        (2, 7, 3, 1.), (1, 1, 1, -.5), (1, 320, 4, 0.),
    ):
        model = fixture(rows, prefix, suffix, alpha)
        name = f"split_silu_m{rows}_p{prefix}_s{suffix}_a{alpha}"
        x = rng.uniform(-20, 20, (1, rows, prefix + suffix)).astype(np.float16)
        model.graph.initializer.append(numpy_helper.from_array(x, "X"))
        del model.graph.input[:]
        fused, count = fuse(model)
        if count != 1:
            raise ValueError(f"Incorrect fixture count: {count}")
        for result, ending in ((fused, ""), (model, "_reference")):
            onnx.checker.check_model(result)
            path = root / (name + ending + ".onnx")
            if path.exists():
                raise FileExistsError(path)
            onnx.save_model(result, path)
    print("Created six strict two-output fused/unfused FP16 fixture pairs")


if __name__ == "__main__":
    main()
