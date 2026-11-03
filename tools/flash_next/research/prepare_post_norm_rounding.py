import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from fuse_hc_post_norm import fuse
from prepare_hc_post_norm import fixture


def main():
    root = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm")
    output = root / "fixtures-boundaries"
    output.mkdir(exist_ok=False)
    functions = onnx.load("/home/sayanshaw/repos/qwen38-int4-quickgelu-next/model/model.onnx",
                          load_external_data=False).functions
    records = []
    for hidden in (33, 2560):
        for rows in (1, 16):
            for shared in (False, True):
                for case in ("subnormal", "ties"):
                    reference = fixture(hidden, rows, shared, False, functions)
                    for tensor in reference.graph.initializer:
                        value = numpy_helper.to_array(tensor).copy()
                        if tensor.name == "streams":
                            if case == "subnormal":
                                values = np.array([0.0, -0.0, 2**-24, -(2**-24), 2.6e-5,
                                                   -2.6e-5, 2**-14, -(2**-14)], np.float16)
                                value[:] = np.resize(values, value.shape)
                            else:
                                value.fill(1)
                        elif tensor.name == "branch":
                            value[:] = np.resize(np.array([2**-11, 3 * 2**-11], np.float16),
                                                 value.shape) if case == "ties" else 0
                        elif tensor.name in ("gate", "shared", "shared_gate"):
                            value.fill(0)
                        elif tensor.name == "norm":
                            value.fill(1)
                        tensor.CopyFrom(numpy_helper.from_array(value, tensor.name))
                    candidate, count = fuse(reference)
                    assert count == 1
                    name = f"rounding_h{hidden}_r{rows}_shared{int(shared)}_{case}"
                    for suffix, model in (("", candidate), ("_reference", reference)):
                        onnx.checker.check_model(model)
                        onnx.save(model, output / (name + suffix + ".onnx"))
                    records.append(dict(fixture=name + ".onnx", elements_per_run=2 * 4 * hidden * rows))
    (root / "fixtures-boundaries.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(dict(fixture_pairs=len(records))))


if __name__ == "__main__":
    main()
