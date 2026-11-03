import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path("/home/sayanshaw/repos/qwen38-int4-exact-wide-gpu7")


def save_fixture(path, node, tensors, shape):
    graph = helper.make_graph([node], "exact_wide", [],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT16, shape)], tensors)
    fixture = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 22), helper.make_opsetid("com.microsoft", 1)])
    fixture.ir_version = 10
    onnx.save(fixture, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = Path("/home/sayanshaw/repos/qwen38-int4-quickgelu-next/model")
    graph = onnx.load(source / "model.onnx", load_external_data=False)
    tensors = {tensor.name: tensor for tensor in graph.graph.initializer}
    output = ROOT / "fixtures"
    output.mkdir()
    report = []
    linked = set()
    for original in graph.graph.node:
        if original.op_type != "MatMulNBits":
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in original.attribute}
        k, n = attrs.get("K"), attrs.get("N")
        if attrs.get("bits") != 4 or attrs.get("block_size") != 32:
            continue
        if (k == 10240 and n == 324) or (k == 2560 and n in (512, 640)):
            continue
        if any(original.input[3:]):
            raise ValueError(f"Unexpected optional input: {original.name}")
        index = len(report)
        bound = (0.03125, 0.25, 4.0)[index % 3]
        x = np.random.default_rng(20261028 + index).uniform(-bound, bound, (1, 1, k)).astype(np.float16)
        initializers = []
        for name in original.input[1:]:
            if not name:
                continue
            tensor = TensorProto()
            tensor.CopyFrom(tensors[name])
            for entry in tensor.external_data:
                if entry.key == "location" and entry.value not in linked:
                    src, dst = source / entry.value, output / entry.value
                    try:
                        os.link(src, dst)
                    except PermissionError:
                        print(f"Hardlink permission requires sudo: {src}", flush=True)
                        subprocess.run(["sudo", "-n", "ln", str(src), str(dst)], check=True)
                    assert src.samefile(dst)
                    linked.add(entry.value)
            initializers.append(tensor)
        initializers.append(numpy_helper.from_array(x, "x"))
        node = onnx.NodeProto()
        node.CopyFrom(original)
        node.input[0], node.output[0] = "x", "y"
        path = output / f"actual_{index:03d}.onnx"
        digest = save_fixture(path, node, initializers, [1, 1, n])
        report.append(dict(fixture=path.name, K=k, N=n, input_bound=bound,
                           source_node=original.name, sha256=digest, kind="actual"))
    assert len(report) == 304, len(report)
    for index, (k, n) in enumerate(((32, 1), (64, 3), (96, 7), (1056, 5), (2560, 1))):
        rng = np.random.default_rng(1000 + index)
        x = rng.uniform(-0.25, 0.25, (1, 1, k)).astype(np.float16)
        b = rng.integers(0, 256, (n, k // 32, 16), dtype=np.uint8)
        scales = rng.uniform(0.001, 0.05, (n, k // 32)).astype(np.float16)
        node = helper.make_node("MatMulNBits", ["x", "b", "s"], ["y"],
                                domain="com.microsoft", K=k, N=n, bits=4, block_size=32)
        initializers = [numpy_helper.from_array(v, name) for v, name in ((x, "x"), (b, "b"), (scales, "s"))]
        path = output / f"tail_{index:02d}.onnx"
        digest = save_fixture(path, node, initializers, [1, 1, n])
        report.append(dict(fixture=path.name, K=k, N=n, sha256=digest, kind="synthetic_tail"))
    (ROOT / "fixtures.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Prepared {len(report)} exact wide fixtures; actual weight data hardlinked, not copied.")


if __name__ == "__main__":
    main()
