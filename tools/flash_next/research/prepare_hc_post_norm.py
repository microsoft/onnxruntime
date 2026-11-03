import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from fuse_hc_post_norm import DOMAIN, fuse

REPOS = Path("/home/sayanshaw/repos")


def fixture(hidden, rows, shared, stress, functions):
    rng = np.random.default_rng(hidden * 17 + rows * 5 + shared + stress * 100)
    shape = [1, rows, 4 * hidden]
    arrays = {
        "streams": rng.normal(0, 0.3, shape).astype(np.float16),
        "branch": rng.normal(0, 0.5, [1, rows, hidden]).astype(np.float16),
        "gate": rng.uniform(-5, 5, [1, rows, 4]).astype(np.float16),
        "norm": rng.uniform(0.5, 1.5, [4 * hidden]).astype(np.float16),
        "divisor": np.array(4, np.float16), "multiplier": np.array(2, np.float16),
    }
    inputs = ["streams", "branch", "gate", "divisor", "multiplier"]
    op = "HyperConnectionPostMixGateF16"
    if shared:
        op = "HyperConnectionSharedPostMixF16"
        arrays["shared"] = rng.normal(0, 0.5, [1, rows, hidden]).astype(np.float16)
        arrays["shared_gate"] = rng.uniform(-5, 5, [1, rows, 1]).astype(np.float16)
        inputs += ["shared", "shared_gate"]
    if stress:
        arrays["gate"] = rng.choice(np.array([-80, -1, -0.0, 0, 1, 80], np.float16), [1, rows, 4])
        arrays["streams"] *= np.float16(100)
    nodes = [
        helper.make_node(op, inputs, ["y"], domain=DOMAIN, num_branches=4),
        helper.make_node("BranchwiseRMSNorm", ["y", "norm"], ["normalized"],
                         domain=DOMAIN, num_branches=4, epsilon=1e-6),
    ]
    graph = helper.make_graph(
        nodes, "hc_post_norm_fixture", [],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape) for name in ("y", "normalized")],
        [numpy_helper.from_array(array, name) for name, array in arrays.items()])
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid(DOMAIN, 1)],
        functions=functions, ir_version=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()
    source = REPOS / "qwen38-int4-quickgelu-next/model"
    destination = REPOS / "qwen38-int4-hc-post-norm"
    original = onnx.load(source / "model.onnx", load_external_data=False)
    if not args.model_only:
        fixtures = destination / "fixtures"
        fixtures.mkdir(exist_ok=False)
        manifest = []
        for hidden in (7, 33, 640, 2560):
            for rows in (1, 16):
                for shared in (False, True):
                    for stress in (False, True):
                        reference = fixture(hidden, rows, shared, stress, original.functions)
                        candidate, count = fuse(reference)
                        assert count == 1
                        name = f"postnorm_h{hidden}_r{rows}_shared{int(shared)}_stress{int(stress)}"
                        for suffix, model in (("", candidate), ("_reference", reference)):
                            onnx.checker.check_model(model)
                            onnx.save(model, fixtures / (name + suffix + ".onnx"))
                        manifest.append(dict(fixture=name + ".onnx", hidden=hidden, rows=rows,
                                             shared=shared, stress=stress, elements_per_run=2 * 4 * hidden * rows))
        (destination / "fixtures.json").write_text(json.dumps(manifest, indent=2) + "\n")
    output = destination / "model"
    output.mkdir(exist_ok=False)
    candidate, count = fuse(original)
    assert count == 95, count
    assert [t.SerializeToString() for t in original.graph.initializer] == [
        t.SerializeToString() for t in candidate.graph.initializer]
    for path in source.iterdir():
        if path.name != "model.onnx" and path.is_file():
            target = output / path.name
            try:
                target.hardlink_to(path)
            except PermissionError:
                subprocess.run(["sudo", "-n", "ln", str(path), str(target)], check=True)
            assert target.samefile(path), target
    onnx.save(candidate, output / "model.onnx")
    onnx.checker.check_model(str(output / "model.onnx"))
    print(f"Prepared 32 finite native fixture pairs and {count} full-model post/norm chains; not GPU-qualified")


if __name__ == "__main__":
    main()
