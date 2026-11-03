from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPOS = Path("/home/sayanshaw/repos")
DOMAIN = "com.microsoft"
OP = "HCProjectionSiLUF16"


def definition():
    matmul = helper.make_node("MatMulNBits", ["x", "weights", "scales"], ["projected"], domain=DOMAIN)
    for name in ("K", "N", "block_size", "bits", "accuracy_level"):
        matmul.attribute.add(name=name, ref_attr_name=name, type=onnx.AttributeProto.INT)
    split = helper.make_node("SplitScaledSiLUF16", ["projected", "sizes"], ["activated", "tail"], domain=DOMAIN)
    split.attribute.add(name="alpha", ref_attr_name="alpha", type=onnx.AttributeProto.FLOAT)
    return helper.make_function(DOMAIN, OP, ["x", "sizes", "weights", "scales"],
        ["activated", "tail"], [matmul, split],
        [helper.make_opsetid("", 21), helper.make_opsetid(DOMAIN, 1)],
        attributes=["K", "N", "block_size", "bits", "accuracy_level", "alpha"])


def fuse(original):
    model = onnx.ModelProto()
    model.CopyFrom(original)
    producers = {output: node for node in model.graph.node for output in node.output}
    consumers = {}
    for node in model.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    outputs = {o.name for o in model.graph.output}
    replacements = {}
    removed = set()
    for node in model.graph.node:
        if node.domain != DOMAIN or node.op_type != "SplitScaledSiLUF16":
            continue
        parent = producers.get(node.input[0])
        if parent is None or parent.op_type != "MatMulNBits" or parent.domain != DOMAIN:
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in parent.attribute}
        if any(attrs.get(k) != v for k, v in (("K", 10240), ("N", 324), ("block_size", 32), ("bits", 4))):
            continue
        if len(parent.input) != 3 or len(parent.output) != 1 or len(consumers[parent.output[0]]) != 1:
            continue
        if parent.output[0] in outputs or set(attrs) - {"K", "N", "block_size", "bits", "accuracy_level"}:
            continue
        attrs.setdefault("accuracy_level", 4)
        split_attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if set(split_attrs) - {"alpha"}:
            continue
        attrs["alpha"] = split_attrs.get("alpha", 1.0)
        replacements[tuple(node.output)] = helper.make_node(OP,
            [parent.input[0], node.input[1], parent.input[1], parent.input[2]], list(node.output),
            name=node.name + "/HCProjectionSiLU", domain=DOMAIN, **attrs)
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
    source = REPOS / "qwen38-int4-hc-post-norm/model"
    original = onnx.load(source / "model.onnx", load_external_data=False)
    output = REPOS / "qwen38-int4-hc-projection-silu"
    fixtures = output / "fixtures-v1"
    fixtures.mkdir(parents=True, exist_ok=False)
    split_definition = [f for f in original.functions if f.name == "SplitScaledSiLUF16"]
    assert len(split_definition) == 1
    random_weights = onnx.load(REPOS / "qwen38-int4-static-node-replay/fixtures-v3/matmul_k10240_n324_rows1.onnx")
    for rows in (1, 2, 16):
        for alpha in (0.25, 1.0, -0.5):
            attrs = dict(K=10240, N=324, block_size=32, bits=4, accuracy_level=4, alpha=alpha)
            graph = helper.make_graph(
                [helper.make_node(OP, ["x", "sizes", "weights", "scales"], ["activated", "tail"],
                                  domain=DOMAIN, **attrs)],
                f"projection_rows{rows}_alpha{alpha}",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, rows, 10240]),
                 helper.make_tensor_value_info("sizes", TensorProto.INT64, [2])],
                [helper.make_tensor_value_info(o, TensorProto.FLOAT16, [1, rows, None])
                 for o in ("activated", "tail")], list(random_weights.graph.initializer))
            model = helper.make_model(graph, functions=split_definition + [definition()],
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid(DOMAIN, 1)], ir_version=10)
            onnx.checker.check_model(model)
            onnx.save(model, fixtures / f"projection_rows{rows}_alpha{alpha}.onnx")
    destination = output / "model"
    destination.mkdir(exist_ok=False)
    candidate, count = fuse(original)
    assert count == 96, count
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
    print(f"Prepared 9 changing-input fixtures and {count} projection/activation chains; not GPU qualified")


if __name__ == "__main__":
    main()
