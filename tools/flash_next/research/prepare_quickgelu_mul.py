from pathlib import Path
import argparse
import subprocess
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from fuse_quickgelu_mul import DOMAIN, OP, definition, fuse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "quickgelu-mul-fixtures"
    root.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261027)
    shapes = ([0, 7], [1], [1, 7], [1, 2560], [1, 4096], [1, 1, 1024],
              [1, 16, 1024], [2, 3, 257])
    for shape in (() if args.model_only else shapes):
        for stress in (False, True):
            name = "qgm_" + "_".join(map(str, shape)) + "_stress" + str(int(stress))
            arrays = [rng.uniform(-5, 5, shape).astype(np.float16) for _ in range(2)]
            if stress:
                arrays[0] = rng.choice(np.array([
                    -80, -16, -9, -8, -1.001, -0.000061, -0.0, 0, 0.000061, 1.001, 8, 16, 80
                ], np.float16), shape)
            initializers = [numpy_helper.from_array(a, n) for a, n in zip(arrays, ["X", "Up"])]
            outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, shape)]
            ref_nodes = [
                helper.make_node("QuickGelu", ["X"], ["Activated"], domain=DOMAIN, alpha=1.0),
                helper.make_node("Mul", ["Activated", "Up"], ["Y"])]
            for suffix, nodes in [("", [helper.make_node(OP, ["X", "Up"], ["Y"], domain=DOMAIN)]),
                                  ("_reference", ref_nodes)]:
                graph = helper.make_graph(nodes, name, [], outputs, initializer=initializers)
                model = helper.make_model(graph, opset_imports=[
                    helper.make_opsetid("", 21), helper.make_opsetid(DOMAIN, 1)],
                    functions=[definition(21)] if not suffix else [])
                model.ir_version = 10
                onnx.checker.check_model(model)
                target = root / (name + suffix + ".onnx")
                if target.exists():
                    raise FileExistsError(target)
                onnx.save_model(model, target)
    source = Path("/home/sayanshaw/repos/qwen38-int4-shared-post")
    output = Path("/home/sayanshaw/repos/qwen38-int4-quickgelu-next/model")
    output.mkdir(parents=True, exist_ok=True)
    original = onnx.load(source / "model.onnx", load_external_data=False)
    model, count = fuse(original)
    assert count == 48, count
    assert [v.SerializeToString() for v in model.graph.initializer] == [
        v.SerializeToString() for v in original.graph.initializer]
    for path in source.iterdir():
        if path.name != "model.onnx" and path.is_file():
            target = output / path.name
            if target.exists():
                assert target.samefile(path), target
                continue
            try:
                target.hardlink_to(path)
            except PermissionError:
                subprocess.run(["sudo", "-n", "ln", str(path), str(target)], check=True)
            assert target.samefile(path), target
    onnx.save_model(model, output / "model.onnx")
    onnx.checker.check_model(str(output / "model.onnx"))
    print("PASS prepared 16 native pairs and 48 production QuickGelu/Mul chains; not GPU-qualified")


if __name__ == "__main__":
    main()
