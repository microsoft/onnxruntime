import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from build_qmoe_node_replay import sha


PARENT = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-matrix-probe/fixtures-v1")
OUTPUT = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-tree-probe/fixtures-v1")


def prepare(parent=PARENT, output=OUTPUT):
    output.mkdir(parents=True, exist_ok=False)
    provenance = {}
    for seed in range(8):
        original = parent / f"seed{seed}.onnx"
        model = onnx.load(original)
        tensors = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
        assert tensors["a"].shape == (16, 4)
        assert tensors["b"].shape[0] == tensors["scales"].shape[0] == 32
        for name in ("b", "scales"):
            assert np.array_equal(tensors[name][:16], tensors[name][16:])
        del model.graph.initializer[:]
        for name, values in tensors.items():
            if name in ("b", "scales"):
                values = np.concatenate([values[:16]] * 5, axis=0)
            model.graph.initializer.append(numpy_helper.from_array(values, name))
        node, = model.graph.node
        attribute, = [value for value in node.attribute if value.name == "N"]
        assert attribute.i == 32
        attribute.i = 80
        model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 80
        onnx.checker.check_model(model)
        destination = output / original.name
        onnx.save(model, destination)
        provenance[original.name] = {
            "parent_sha256": sha(original),
            "fixture_sha256": sha(destination),
        }
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
