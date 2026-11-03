from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from fuse_mlp_input_projections import DOMAIN, OP, WIDTHS, definition, fuse


REPOS = Path("/home/sayanshaw/repos")
OUTPUT = REPOS / "qwen38-int4-mlp-input-projections"


def fixture(original, rows, scale):
    rng = np.random.default_rng(20261029)
    tensors = []
    for role, width in zip(("router", "gate", "up", "shared"), WIDTHS):
        tensors.extend((
            numpy_helper.from_array(rng.integers(0, 256, (width, 80, 16), dtype=np.uint8), role + "_weight"),
            numpy_helper.from_array(rng.uniform(scale / 2, scale, (width, 80)).astype(np.float16), role + "_scale")))
    function = definition(21)
    graph = helper.make_graph([
        helper.make_node(OP, list(function.input), list(function.output), domain=DOMAIN)],
        "mlp_input_projections",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, rows, 2560])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, rows, width])
         for name, width in zip(function.output, (512, 640, 1))], tensors)
    return helper.make_model(graph, functions=[*original.functions, function],
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid(DOMAIN, 1)], ir_version=10)


def main():
    source = REPOS / "qwen38-int4-hc-post-norm-projection/model"
    original = onnx.load(source / "model.onnx", load_external_data=False)
    candidate, chains = fuse(original)
    assert chains == 48, chains
    assert [t.SerializeToString() for t in original.graph.initializer] == [
        t.SerializeToString() for t in candidate.graph.initializer]
    fixtures = OUTPUT / "fixtures-v1"
    fixtures.mkdir(parents=True, exist_ok=False)
    for rows in (0, 1, 2, 16):
        for scale in (0.015625, 0.125):
            model = fixture(original, rows, scale)
            onnx.checker.check_model(model)
            onnx.save(model, fixtures / f"mlp_rows{rows}_scale{scale}.onnx")
    destination = OUTPUT / "model"
    destination.mkdir(exist_ok=False)
    for path in source.iterdir():
        if path.name != "model.onnx" and path.is_file():
            target = destination / path.name
            try:
                target.hardlink_to(path)
            except PermissionError:
                subprocess.run(["sudo", "-n", "ln", str(path), str(target)], check=True)
            assert target.samefile(path)
    onnx.save(candidate, destination / "model.onnx")
    onnx.checker.check_model(str(destination / "model.onnx"))
    print(f"Prepared eight random-weight fixtures and {chains} model chains; not GPU qualified")


if __name__ == "__main__":
    main()
