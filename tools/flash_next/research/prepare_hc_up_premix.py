from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from prepare_hc_projection_silu import DOMAIN, REPOS

OP = "HCUpPreMixF16"


def definition():
    matrix = helper.make_node("MatMulNBits", ["x", "weights", "scales"], ["projected"], domain=DOMAIN)
    for name in ("K", "N", "block_size", "bits", "accuracy_level"):
        matrix.attribute.add(name=name, ref_attr_name=name, type=onnx.AttributeProto.INT)
    premix = helper.make_node("HyperConnectionPreMixSigmoidF16", ["streams", "projected"], ["y"], domain=DOMAIN)
    for name, kind in (("num_branches", onnx.AttributeProto.INT),
                       ("reduction_scale", onnx.AttributeProto.FLOAT)):
        premix.attribute.add(name=name, ref_attr_name=name, type=kind)
    return helper.make_function(DOMAIN, OP, ["streams", "x", "weights", "scales"], ["y"],
        [matrix, premix], [helper.make_opsetid("", 21), helper.make_opsetid(DOMAIN, 1)],
        attributes=["K", "N", "block_size", "bits", "accuracy_level", "num_branches", "reduction_scale"])


def fuse(original):
    model = onnx.ModelProto()
    model.CopyFrom(original)
    producers = {o: n for n in model.graph.node for o in n.output}
    consumers = {}
    for n in model.graph.node:
        for value in n.input:
            consumers.setdefault(value, []).append(n)
    outputs = {o.name for o in model.graph.output}
    replacements = {}
    removed = set()
    for n in model.graph.node:
        if n.op_type != "HyperConnectionPreMixSigmoidF16" or n.domain != DOMAIN:
            continue
        parent = producers.get(n.input[1])
        if parent is None or parent.op_type != "MatMulNBits" or parent.domain != DOMAIN:
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in parent.attribute}
        mix = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        if any(attrs.get(k) != v for k, v in (("K", 320), ("N", 10240), ("block_size", 32), ("bits", 4))):
            continue
        if len(parent.input) != 3 or len(parent.output) != 1 or len(consumers[parent.output[0]]) != 1:
            continue
        if parent.output[0] in outputs or mix.get("num_branches") != 4:
            continue
        if (set(attrs) - {"K", "N", "block_size", "bits", "accuracy_level"} or
                set(mix) - {"num_branches", "reduction_scale"}):
            continue
        attrs.setdefault("accuracy_level", 4)
        attrs.update(num_branches=4, reduction_scale=mix.get("reduction_scale", 1.0))
        replacements[tuple(n.output)] = helper.make_node(OP,
            [n.input[0], parent.input[0], parent.input[1], parent.input[2]], list(n.output),
            name=n.name + "/HCUpPreMix", domain=DOMAIN, **attrs)
        removed.add(parent.output[0])
    nodes = [replacements.get(tuple(n.output), n) for n in model.graph.node
             if not n.output or n.output[0] not in removed]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    if replacements:
        assert not any(f.name == OP and f.domain == DOMAIN for f in model.functions)
        model.functions.append(definition())
    return model, len(replacements)


def main():
    source = REPOS / "qwen38-int4-hc-projection-silu/model"
    original = onnx.load(source / "model.onnx", load_external_data=False)
    directory = REPOS / "qwen38-int4-hc-up-premix"
    fixtures = directory / "fixtures-v1"
    fixtures.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261029)
    weights = rng.integers(0, 256, size=(10240, 10, 16), dtype=np.uint8)
    scales = rng.uniform(0.001, 0.01, size=(10240, 10)).astype(np.float16)
    functions = [f for f in original.functions if f.name == "HyperConnectionPreMixSigmoidF16"]
    assert len(functions) == 1
    for rows in (1, 2, 16):
        for reduction in (0.25, 1.0, -0.5):
            graph = helper.make_graph([helper.make_node(OP, ["streams", "x", "weights", "scales"], ["y"],
                domain=DOMAIN, K=320, N=10240, bits=4, block_size=32, accuracy_level=4,
                num_branches=4, reduction_scale=reduction)], "up_premix",
                [helper.make_tensor_value_info("streams", TensorProto.FLOAT16, [1, rows, 10240]),
                 helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, rows, 320])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, rows, 2560])],
                [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(scales, "scales")])
            model = helper.make_model(graph, functions=functions + [definition()],
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid(DOMAIN, 1)], ir_version=10)
            onnx.checker.check_model(model)
            onnx.save(model, fixtures / f"up_rows{rows}_reduction{reduction}.onnx")
    destination = directory / "model"
    destination.mkdir(exist_ok=False)
    candidate, count = fuse(original)
    assert count == 97
    assert [t.SerializeToString() for t in original.graph.initializer] == [
        t.SerializeToString() for t in candidate.graph.initializer]
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
    print("Prepared 9 changing-input fixtures and 97 exact up-projection/pre-mix chains; not GPU qualified")


if __name__ == "__main__":
    main()
