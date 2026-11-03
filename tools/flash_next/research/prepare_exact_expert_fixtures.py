import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos-root", type=Path, default=Path("/workspace"))
    args = parser.parse_args()
    source = args.repos_root / "qwen38-webgpu-raw"
    destination = args.repos_root / "qwen38-int4-exact-expert/fixtures"
    destination.mkdir(exist_ok=True)
    model = onnx.load(source / "model.onnx", load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    nodes = [node for node in model.graph.node if node.op_type == "QMoE"]
    rng = np.random.default_rng(20261029)
    manifest = []
    for layer in (0, 3, 7, 23, 35, 47):
        original = nodes[layer]
        for rows in (1, 2, 16):
            node = copy.deepcopy(original)
            node.name = f"expert_layer{layer}_rows{rows}"
            node.input[0], node.input[1] = "x", "router"
            node.output[0] = "y"
            tensors = [copy.deepcopy(initializers[name]) for name in node.input[2:] if name]
            weight = tensors[0]
            experts = weight.dims[0]
            hidden = next(t for t in tensors if t.name == node.input[5]).dims[1]
            x = rng.uniform(-0.2, 0.2, (rows, hidden)).astype(np.float16)
            router = rng.normal(0, 2, (rows, experts)).astype(np.float16)
            tensors += [numpy_helper.from_array(x, "x"), numpy_helper.from_array(router, "router")]
            for tensor in tensors:
                for entry in tensor.external_data:
                    if entry.key == "location":
                        target = destination / entry.value
                        if not target.exists():
                            os.link(source / entry.value, target)
                        assert os.stat(source / entry.value).st_ino == os.stat(target).st_ino
            graph = helper.make_graph([node], node.name, [],
                                      [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [rows, hidden])],
                                      tensors)
            fixture = helper.make_model(graph, opset_imports=model.opset_import, ir_version=model.ir_version)
            path = destination / (node.name + ".onnx")
            if path.exists():
                raise FileExistsError(path)
            onnx.save(fixture, path)
            onnx.checker.check_model(str(path))
            manifest.append(dict(fixture=path.name, layer=layer, rows=rows, hidden=hidden,
                                 original_node=original.name))
    (destination.parent / "fixtures.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(manifest)} real-weight QMoE fixtures with hardlinked original external data")


if __name__ == "__main__":
    main()
