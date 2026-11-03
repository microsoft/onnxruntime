import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from build_qmoe_node_replay import sha


PARENT = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-matrix-probe/fixtures-v1")
OUTPUT = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-pair-matrix-probe/fixtures-v1")


def prepare(parent=PARENT, output=OUTPUT):
    records = json.loads((parent / "manifest.json").read_text())["fixtures"]
    output.mkdir(parents=True, exist_ok=False)
    provenance = {}
    for seed in range(8):
        original = parent / f"seed{seed}.onnx"
        record, = [r for r in records if Path(r["path"]).name == original.name]
        assert sha(original) == record["sha256"], original
        model = onnx.load(original)
        tensors = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
        assert tensors["a"].shape == (16, 4)
        del model.graph.initializer[:]
        for name, values in tensors.items():
            if name in ("b", "scales"):
                assert values.shape[0] == 32 and np.array_equal(values[:16], values[16:])
                values = np.concatenate([values[:16]] * 4, axis=0)
            model.graph.initializer.append(numpy_helper.from_array(values, name))
        node, = model.graph.node
        attribute, = [value for value in node.attribute if value.name == "N"]
        assert attribute.i == 32
        attribute.i = 64
        model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 64
        onnx.checker.check_model(model)
        destination = output / original.name
        onnx.save(model, destination)
        provenance[original.name] = dict(parent_sha256=sha(original), fixture_sha256=sha(destination))
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
