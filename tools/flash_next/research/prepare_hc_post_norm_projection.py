import itertools
from pathlib import Path
import subprocess

import onnx
from onnx import TensorProto, helper

from fuse_hc_post_norm_projection import DOMAIN, OPS, definition, fuse


REPOS = Path("/home/sayanshaw/repos")
OUTPUT = REPOS / "qwen38-int4-hc-post-norm-projection"


def fixture(original, weights, shared, rows, scale_shape, alpha):
    op = "HCSharedPostNormF16" if shared else "HCPostNormF16"
    shapes = {"streams": [1, rows, 10240], "branch": [1, rows, 2560],
              "gate": [1, rows, 4], "scale": scale_shape}
    if shared:
        shapes.update(shared=[1, rows, 2560], shared_gate=[1, rows, 1])
    attrs = dict(K=10240, N=324, block_size=32, bits=4, accuracy_level=4,
                 epsilon=1e-6, alpha=alpha)
    outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, rows, width])
               for name, width in (("y", 10240), ("normalized", 10240),
                                   ("activated", None), ("tail", None))]
    graph = helper.make_graph([
        helper.make_node(OPS[op], [*shapes, "sizes", "weights", "scales"],
                         [value.name for value in outputs], domain=DOMAIN, **attrs)],
        "post_norm_projection", [
            *[helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape)
              for name, shape in shapes.items()],
            helper.make_tensor_value_info("sizes", TensorProto.INT64, [2])],
        outputs, list(weights.graph.initializer))
    return helper.make_model(graph, functions=[*original.functions, definition(op, 21)],
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid(DOMAIN, 1)], ir_version=10)


def main():
    source = REPOS / "qwen38-int4-hc-up-premix/model"
    original = onnx.load(source / "model.onnx", load_external_data=False)
    weights = onnx.load(REPOS / "qwen38-int4-static-node-replay/fixtures-v3/matmul_k10240_n324_rows1.onnx")
    fixtures = OUTPUT / "fixtures-v1"
    fixtures.mkdir(parents=True, exist_ok=False)
    count = 0
    for shared, rows, shape, alpha in itertools.product(
            (False, True), (1, 2, 16), ([2560], [4, 2560]), (0.25, 1.0, -0.5)):
        model = fixture(original, weights, shared, rows, shape, alpha)
        onnx.checker.check_model(model)
        name = f"postnorm_shared{int(shared)}_rows{rows}_scale{len(shape)}_alpha{alpha}.onnx"
        onnx.save(model, fixtures / name)
        count += 1
    assert count == 36
    candidate, chains = fuse(original)
    assert chains == 94, chains
    assert [t.SerializeToString() for t in original.graph.initializer] == [
        t.SerializeToString() for t in candidate.graph.initializer]
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
    print(f"Prepared {count} random-weight fixtures and {chains} model chains; not GPU qualified")


if __name__ == "__main__":
    main()
