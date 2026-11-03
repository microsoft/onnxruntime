import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto


ROOT = Path("/home/sayanshaw/repos")
OUTPUT = ROOT / "qwen38-int4-qmoe-hierarchical-gate/fixtures-v1"


def main():
    model = onnx.load(ROOT / "qwen38-int4-qmoe-node-replay/fixtures/expert_layer0_rows1.onnx",
                      load_external_data=False)
    node = model.graph.node[0]
    node.input[7] = ""
    tensors = {tensor.name: tensor for tensor in model.graph.initializer}
    selected = [tensors[node.input[index]] for index in (2, 3, 4, 5, 6)]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    offset = 0
    offsets = []
    for tensor in selected:
        offsets.append(offset)
        size = int(np.prod(tensor.dims)) * (1 if tensor.data_type == TensorProto.UINT8 else 2)
        tensor.ClearField("raw_data")
        tensor.ClearField("external_data")
        tensor.data_location = TensorProto.EXTERNAL
        for key, value in (("location", "identity.data"), ("offset", str(offset)), ("length", str(size))):
            tensor.external_data.add(key=key, value=value)
        offset += size
    data = OUTPUT / "identity.data"
    with data.open("xb") as stream:
        stream.truncate(offset)
        for expert in range(512):
            stream.seek(offsets[2] + expert * 1280 * 2)
            stream.write(np.array([1.0, 1.3681640625], dtype="<f2").tobytes())
            stream.seek(offsets[3] + (expert * 2560 + expert) * 320)
            stream.write(b"\x89")
            stream.seek(offsets[4] + (expert * 2560 + expert) * 20 * 2)
            stream.write(np.array([1.0], dtype="<f2").tobytes())
    del model.graph.initializer[:]
    model.graph.initializer.extend(selected)
    path = OUTPUT / "gate_identity_rows1.onnx"
    onnx.save(model, path)
    onnx.checker.check_model(str(path))
    digest = hashlib.sha256()
    with data.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    report = dict(synthetic_sparse_identity_fixture=True,
        logical_data_bytes=data.stat().st_size, allocated_data_bytes=data.stat().st_blocks * 512,
        data_sha256=digest.hexdigest(),
        scope="One-hot expert outputs expose exact routing weights and selected membership",
        expected_selected_weight_f16_bits=int(np.float16(0.1).view(np.uint16)))
    assert report["allocated_data_bytes"] < 16 * 1024**2
    (OUTPUT / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
