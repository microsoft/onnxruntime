import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from fuse_hyper_connections import fuse
from test_hyper_connection_fusion import fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "hc-fusion-fixtures-v2"
    root.mkdir(exist_ok=True)
    if args.negative_only:
        for tensor_name, value in (("D", 4), ("S", 2)):
            model = onnx.load(root / "hc_post_m1_h4.onnx")
            tensor = next(v for v in model.graph.initializer if v.name == tensor_name)
            tensor.CopyFrom(numpy_helper.from_array(np.full((4,), value, dtype=np.float16), tensor_name))
            name = "negative_" + tensor_name.lower() + "_vector"
            target = root / (name + ".onnx")
            reference = root / (name + "_reference.onnx")
            if target.exists() or reference.exists():
                raise FileExistsError(name)
            onnx.checker.check_model(model)
            onnx.save_model(model, target)
            onnx.save_model(onnx.load(root / "hc_post_m1_h4_reference.onnx"), reference)
        print("Created two unsupported scalar-parameter shape cases")
        return
    rng = np.random.default_rng(20261026)
    for rows, hidden in ((1, 4), (2, 7), (1, 2560), (16, 2560)):
        base = fixture()
        shapes = {"X": [1, rows, 4 * hidden], "P": [1, rows, 4 * hidden],
                  "Z": [1, rows, hidden], "G": [1, rows, 4],
                  "PG": [1, rows, 4 * hidden], "A": [1, rows, hidden],
                  "DG": [1, rows, 4], "SG": [1, rows, 4], "MG": [1, rows, 4],
                  "Y": [1, rows, 4 * hidden]}
        for value in [*base.graph.input, *base.graph.output, *base.graph.value_info]:
            value.CopyFrom(helper.make_tensor_value_info(value.name, TensorProto.FLOAT16, shapes[value.name]))
        for value in base.graph.input:
            data = rng.uniform(-5, 5, size=shapes[value.name]).astype(np.float16)
            base.graph.initializer.append(numpy_helper.from_array(data, value.name))
        del base.graph.input[:]
        fused, counts = fuse(base)
        if sorted(counts.values()) != [1, 1]:
            raise ValueError(f"Unexpected fixture fusion counts: {counts}")
        for kind, output in (("pre", "A"), ("post", "Y")):
            name = f"hc_{kind}_m{rows}_h{hidden}"
            for model, suffix in ((deepcopy(fused), ""), (deepcopy(base), "_reference")):
                chosen = deepcopy(next(v for v in model.graph.output if v.name == output))
                chosen.name = "y"
                for node in model.graph.node:
                    for index, value in enumerate(node.input):
                        if value == output:
                            node.input[index] = "y"
                    for index, value in enumerate(node.output):
                        if value == output:
                            node.output[index] = "y"
                del model.graph.output[:]
                model.graph.output.append(chosen)
                onnx.checker.check_model(model)
                path = root / (name + suffix + ".onnx")
                if path.exists():
                    raise FileExistsError(path)
                onnx.save_model(model, path)
    print("Created eight fused/unfused FP16 fixture pairs")


if __name__ == "__main__":
    main()
