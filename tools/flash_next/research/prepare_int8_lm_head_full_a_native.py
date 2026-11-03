import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent
REPO = Path("/home/sayanshaw/repos/qwen38-int4-int8-lm-head-full-a")
PROOF = REPO / "native-proof-v1"
DATA = REPO / "fixtures-v1"
MODEL = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
CASES = [
    (1, 2560, 8, 32, False), (1, 2560, 16, 32, False), (1, 2560, 32, 32, False),
    (1, 2560, 16, 16, False), (1, 2560, 16, 80, False),
    (1, 2560, 16, 128, False), (1, 2560, 16, 512, False),
    (1, 2560, 7, 32, False), (1, 2560, 33, 32, False),
    (2, 2560, 16, 32, False), (1, 2048, 16, 32, False), (1, 2560, 16, 32, True),
]


def region_sha(path, offset, length):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = length
        while remaining:
            data = stream.read(min(1024 * 1024, remaining))
            if not data:
                raise RuntimeError("Truncated actual tensor region")
            digest.update(data)
            remaining -= len(data)
    return digest.hexdigest()


def prepare_fixtures():
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    assert sha(MODEL / "model.onnx") == "8037bd0a14a3ae5307cd2d626e90d3be26a6a7b8893062246ee23487599400f9"
    assert sha(MODEL / "genai_config.json") == "8b878a2c79a0763448f02a5105aa98a3e32f43ded848f93a1df9a071607e16b6"
    DATA.mkdir(parents=True, exist_ok=False)
    records = []
    for index, (m, k, n, block, fp32) in enumerate(CASES):
        dtype, typ = (np.float32, TensorProto.FLOAT) if fp32 else (np.float16, TensorProto.FLOAT16)
        rng = np.random.default_rng(9300 + index)
        weights = rng.integers(120, 137, size=(n, k // block, block), dtype=np.uint8)
        choices = np.array([0.0, -0.0, 0.00390625, -0.00390625, 0.03125], dtype=dtype)
        scales = choices[rng.integers(0, len(choices), size=(n, k // block))]
        graph = helper.make_graph([
            helper.make_node("MatMulNBits", ["a", "b", "scales"], ["y"],
                domain="com.microsoft", K=k, N=n, bits=8, block_size=block, accuracy_level=4)],
            "int8-head-synthetic", [helper.make_tensor_value_info("a", typ, [m, k])],
            [helper.make_tensor_value_info("y", typ, [m, n])],
            [numpy_helper.from_array(weights, "b"), numpy_helper.from_array(scales, "scales")])
        model = helper.make_model(graph, opset_imports=[
            helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)], ir_version=10)
        path = DATA / f"case{index}-m{m}-k{k}-n{n}-b{block}-{'f32' if fp32 else 'f16'}.onnx"
        onnx.checker.check_model(model)
        onnx.save(model, path)
        eligible = m == 1 and k == 2560 and n in (8, 16, 32) and not fp32
        records.append(dict(path=str(path), m=m, k=k, n=n, block_size=block,
                            fp32=fp32, eligible=eligible, actual_weights=False, sha256=sha(path)))
    model = onnx.load(MODEL / "model.onnx", load_external_data=False)
    nodes = [node for node in model.graph.node if node.name == "/lm_head/MatMul_Q8"]
    assert len(nodes) == 1
    node = nodes[0]
    attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
    assert attrs == dict(K=2560, N=248320, bits=8, block_size=32, accuracy_level=4)
    assert len(node.input) == 3
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    (DATA / "model.onnx.data").symlink_to(MODEL / "model.onnx.data")
    tensors, provenance = [], []
    for name, target, shape, typ in (
        (node.input[1], "b", [248320, 80, 32], TensorProto.UINT8),
        (node.input[2], "scales", [248320, 80], TensorProto.FLOAT16),
    ):
        tensor = copy.deepcopy(initializers[name])
        assert list(tensor.dims) == shape and tensor.data_type == typ
        assert tensor.data_location == TensorProto.EXTERNAL
        external = {item.key: item.value for item in tensor.external_data}
        assert external["location"] == "model.onnx.data"
        provenance.append(dict(name=name, shape=shape, dtype=typ, external=external,
            payload_sha256=region_sha(MODEL / external["location"], int(external["offset"]), int(external["length"]))))
        tensor.name = target
        tensors.append(tensor)
    graph = helper.make_graph([
        helper.make_node("MatMulNBits", ["a", "b", "scales"], ["y"], domain="com.microsoft", **attrs)],
        "actual-int8-lm-head", [helper.make_tensor_value_info("a", TensorProto.FLOAT16, [1, 2560])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 248320])], tensors)
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)], ir_version=10)
    path = DATA / "actual-lm-head.onnx"
    onnx.save(model, path)
    onnx.checker.check_model(str(path))
    records.append(dict(path=str(path), m=1, k=2560, n=248320, block_size=32, fp32=False,
                        eligible=True, actual_weights=True, sha256=sha(path), tensors=provenance))
    (DATA / "manifest.json").write_text(json.dumps(dict(fixtures=records,
        graph_sha256=sha(MODEL / "model.onnx"), config_sha256=sha(MODEL / "genai_config.json"),
        external_weights_symlink_to_frozen_model=True), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        prepare_fixtures()
        PROOF.mkdir(parents=True, exist_ok=False)
        for name, source in (("independent_compare.cpp", "int8_lm_head_compare.cpp"),
                             ("hc_original_reference_io.inc", "hc_original_reference_io.inc"),
                             ("reference_io_test.cpp", "hc_reference_io_test.cpp")):
            (PROOF / name).write_text((ROOT / source).read_text())
        print(PROOF)
    else:
        proof = Path(str(PROOF).replace("/home/sayanshaw/repos", "/workspace"))
        if (proof / "build-manifest.json").exists():
            raise FileExistsError(proof / "build-manifest.json")
        sdk = Path("/workspace/qwen38-fp8-webgpu-runtime/sdk/lib")
        binaries = []
        for name in ("independent_compare", "reference_io_test"):
            binary = proof / name
            if binary.exists():
                raise FileExistsError(binary)
            subprocess.run(["c++", "-O3", "-std=c++20", str(proof / (name + ".cpp")),
                "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
                f"-L{sdk}", f"-Wl,-rpath,{sdk}", "-lonnxruntime", "-o", str(binary)], check=True)
            binaries.append(binary)
        subprocess.run([str(binaries[1]), str(proof / "io-test.bin")], check=True)
        evidence = [Path(__file__), ROOT / "int8_lm_head_compare.cpp", ROOT / "hc_original_reference_io.inc",
                    proof / "independent_compare.cpp", proof / "hc_original_reference_io.inc",
                    proof / "reference_io_test.cpp", *binaries]
        (proof / "build-manifest.json").write_text(json.dumps(dict(
            status="typed_int8_head_comparator_built_io_tested",
            rounds=16, repeats_per_round=4, float16_float32=True,
            source_sha256={str(path): sha(path) for path in evidence}), indent=2) + "\n")
        print("typed_int8_head_comparator_built_io_tested", flush=True)
