import hashlib
import json
import math
from pathlib import Path
import subprocess

import onnx
from onnx import TensorProto, helper

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent
REPOS = Path("/home/sayanshaw/repos")
MODEL = REPOS / "qwen38-int4-mlp-input-projections/model/model.onnx"
DATA = REPOS / "qwen38-int4-hc-up-zero-tail/actual-fixtures-v3"


def range_sha(path, offset, length):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = length
        while remaining:
            chunk = stream.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError(f"Truncated original tensor range: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def main():
    output = ROOT / "hc-up-zero-tail-actual-v3-generation.json"
    if output.exists() or DATA.exists():
        raise FileExistsError("Fresh actual HC-up fixture generation required")
    if sha(MODEL) != "8037bd0a14a3ae5307cd2d626e90d3be26a6a7b8893062246ee23487599400f9":
        raise RuntimeError("Changed frozen model graph")
    model = onnx.load(MODEL, load_external_data=False)
    nodes = [node for node in model.graph.node
             if node.domain == "com.microsoft" and node.op_type == "HCUpPreMixF16"]
    if len(nodes) != 97:
        raise RuntimeError("Unexpected frozen HC-up node count")
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    functions = [function for function in model.functions
                 if function.domain == "com.microsoft"
                 and function.name in ("HCUpPreMixF16", "HyperConnectionPreMixSigmoidF16")]
    if {function.name for function in functions} != {"HCUpPreMixF16", "HyperConnectionPreMixSigmoidF16"}:
        raise RuntimeError("Missing frozen HC-up function definitions")
    DATA.mkdir(parents=True)
    aliases, ranges, fixtures = {}, [], []
    for index in (0, 48, 96):
        node = nodes[index]
        if len(node.input) != 4 or len(node.output) != 1:
            raise RuntimeError("Unexpected frozen HC-up interface")
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if any(attrs.get(key) != value for key, value in
               (("K", 320), ("N", 10240), ("bits", 4), ("block_size", 32), ("num_branches", 4))):
            raise RuntimeError("Unexpected frozen HC-up shape or quantization")
        tensors = []
        for name, source_name, dtype, count in (
                ("weights", node.input[2], TensorProto.UINT8, 10240 * 10 * 16),
                ("scales", node.input[3], TensorProto.FLOAT16, 10240 * 10)):
            original = initializers[source_name]
            if original.data_type != dtype or math.prod(original.dims) != count:
                raise RuntimeError("Unexpected original HC-up weight tensor")
            if original.data_location != TensorProto.EXTERNAL or original.raw_data:
                raise RuntimeError("Actual HC-up fixture requires read-only external tensor aliases")
            metadata = {entry.key: entry.value for entry in original.external_data}
            source = (MODEL.parent / metadata["location"]).resolve(strict=True)
            if not source.is_relative_to(REPOS.resolve()):
                raise RuntimeError("Frozen external tensor is outside the approved repository root")
            if source not in aliases:
                alias = DATA / f"source-{len(aliases)}.bin"
                try:
                    alias.hardlink_to(source)
                except PermissionError:
                    subprocess.run(["sudo", "-n", "ln", str(source), str(alias)], check=True)
                if not alias.samefile(source):
                    raise RuntimeError("Read-only fixture alias does not resolve to original bytes")
                aliases[source] = alias
            tensor = onnx.TensorProto()
            tensor.CopyFrom(original)
            tensor.name = name
            for entry in tensor.external_data:
                if entry.key == "location":
                    entry.value = aliases[source].name
            length = count * (2 if dtype == TensorProto.FLOAT16 else 1)
            if int(metadata.get("length", length)) != length:
                raise RuntimeError("Unexpected original tensor byte length")
            offset = int(metadata.get("offset", 0))
            if offset < 0:
                raise RuntimeError("Negative original tensor offset")
            ranges.append({"anchor": index, "tensor": source_name, "source": str(source),
                           "alias": str(aliases[source]), "offset": offset, "length": length,
                           "range_sha256": range_sha(source, offset, length),
                           "tensor_proto_sha256": hashlib.sha256(original.SerializeToString()).hexdigest()})
            tensors.append(tensor)
        for rows in (1, 2, 16):
            candidate = helper.make_node("HCUpPreMixF16", ["streams", "x", "weights", "scales"], ["y"],
                                         domain="com.microsoft")
            candidate.attribute.extend(node.attribute)
            graph = helper.make_graph(
                [candidate], "actual_hc_up",
                [helper.make_tensor_value_info("streams", TensorProto.FLOAT16, [1, rows, 10240]),
                 helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, rows, 320])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, rows, 2560])], tensors)
            fixture = helper.make_model(graph, functions=functions, ir_version=10,
                                        opset_imports=[helper.make_opsetid("", 21),
                                                       helper.make_opsetid("com.microsoft", 1)])
            path = DATA / f"anchor{index}-rows{rows}.onnx"
            onnx.save(fixture, path)
            onnx.checker.check_model(str(path))
            fixtures.append({"path": str(path), "anchor": index, "rows": rows, "sha256": sha(path)})
    report = {"status": "generated_not_qualified", "model": str(MODEL), "model_sha256": sha(MODEL),
              "fixtures": fixtures, "original_tensor_ranges": ranges,
              "no_model_weight_copies": True, "external_data_packaging": "in-directory original-inode hardlinks",
              "quality_qualified": False}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "fixtures": len(fixtures),
                      "read_only_shard_aliases": len(aliases)}), flush=True)


if __name__ == "__main__":
    main()
