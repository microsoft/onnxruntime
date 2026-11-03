from pathlib import Path

from build_qmoe_node_replay import replace_once
from prepare_dense_paired_matrix_reset_proofs import make_runner as reset_runner


ROOT = Path(__file__).parent
FIXTURE = r'''
MODEL_DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
MODEL_GRAPH = MODEL_DIRECTORY / "model.onnx"
MODEL_CONFIG = MODEL_DIRECTORY / "genai_config.json"


def fixture(k, n):
    import hashlib
    from onnx.external_data_helper import load_external_data_for_tensor
    if sha(MODEL_GRAPH) != "8037bd0a14a3ae5307cd2d626e90d3be26a6a7b8893062246ee23487599400f9":
        raise RuntimeError("Changed frozen model graph")
    if sha(MODEL_CONFIG) != "8b878a2c79a0763448f02a5105aa98a3e32f43ded848f93a1df9a071607e16b6":
        raise RuntimeError("Changed frozen model config")
    model = onnx.load(MODEL_GRAPH, load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    selected = None
    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if (attrs.get("K"), attrs.get("N")) == (k, n):
            if attrs.get("bits") != 4 or attrs.get("block_size") != 32 or len(node.input) != 3:
                raise RuntimeError("Unexpected actual-weight layout")
            selected = node
            break
    if selected is None:
        raise RuntimeError(f"No actual model shape: {(k, n)}")
    tensors = []
    provenance = dict(k=k, n=n, node=selected.name, tensors=[])
    for source_name, target in zip(selected.input[1:3], ("b", "scales")):
        tensor = initializers[source_name]
        external = {item.key: item.value for item in tensor.external_data}
        if tensor.data_location == TensorProto.EXTERNAL:
            load_external_data_for_tensor(tensor, str(MODEL_DIRECTORY))
        array = numpy_helper.to_array(tensor, base_dir=str(MODEL_DIRECTORY))
        expected = (n, k // 32, 16) if target == "b" else (n, k // 32)
        dtype = np.uint8 if target == "b" else np.float16
        if array.shape != expected or array.dtype != dtype or not np.isfinite(array).all():
            raise RuntimeError("Invalid actual-weight tensor")
        provenance["tensors"].append(dict(name=source_name, external=external,
            payload_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
            shape=list(array.shape), dtype=str(array.dtype)))
        tensors.append(numpy_helper.from_array(array, target))
    path = DATA / f"k{k}-n{n}.onnx"
    if path.exists():
        raise FileExistsError(path)
    graph = helper.make_graph(
        [helper.make_node("MatMulNBits", ["a", "b", "scales"], ["y"],
            name="actual_dense", domain="com.microsoft",
            K=k, N=n, bits=4, block_size=32, accuracy_level=4)],
        "dense-paired-matrix-actual",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT16, [1, k])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, n])], tensors)
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    provenance["fixture_sha256"] = sha(path)
    record = DATA / f"k{k}-n{n}-actual-tensors.json"
    record.write_text(json.dumps(provenance, indent=2) + "\n")
    return path

'''


def make_runner():
    source = reset_runner()
    source = source.replace('DIRECTORY / "fixtures-v1"', 'DIRECTORY / "actual-fixtures-v1"')
    source = source.replace("dense-paired-matrix-reset-v1-", "dense-paired-matrix-reset-actual-v1-")
    source = source.replace("(10240, 320), (2560, 1),", "(10240, 320),")
    source = source.replace("packed_dense_original_provider_synthetic_shapes_supported",
                            "packed_dense_original_provider_actual_weights_supported")
    source = source.replace("actual_model_weights_used=False", "actual_model_weights_used=True")
    start = source.index("\ndef fixture(k, n):")
    stop = source.index('\nif __name__ == "__main__":', start)
    source = source[:start] + FIXTURE + source[stop:]
    source = replace_once(source, '    prerequisite = ROOT / "fp16-dot4-dynamic-sweep-v1-outcome.json"',
        '''    synthetic_path = ROOT / "dense-paired-matrix-reset-v1-outcome.json"
    synthetic = json.loads(synthetic_path.read_text())
    if synthetic["status"] != "packed_dense_original_provider_synthetic_shapes_supported":
        raise RuntimeError("Synthetic original-provider gate not supported")
    for path, digest in synthetic["source_and_evidence_sha256"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Changed synthetic evidence: {path}")
    prerequisite = ROOT / "fp16-dot4-dynamic-sweep-v1-outcome.json"''')
    source = replace_once(source, "    evidence = [Path(__file__),", "    evidence = [synthetic_path, MODEL_GRAPH, MODEL_CONFIG, Path(__file__),")
    source = replace_once(source, "        evidence.append(model)",
        '        evidence.extend((model, DATA / f"k{k}-n{n}-actual-tensors.json"))')
    source = replace_once(source,
        '(("capture", 0, ORIGINAL), ("compare", 1, provider))',
        '(("capture", 0, ORIGINAL), ("default", 0, provider), ("compare", 1, provider))')
    source = replace_once(source, "[workspace(binary), mode, workspace(library),",
        '[workspace(binary), "compare" if mode == "default" else mode, workspace(library),')
    source = replace_once(source, 'if mode == "capture" or n == 1:',
                          'if mode in ("capture", "default") or n == 1:')
    source = replace_once(source, '            phases.append(dict(mode=mode,',
        '''            if mode == "default" and mismatches:
                raise RuntimeError("Compiled default-off provider differs from original")
            phases.append(dict(mode=mode,''')
    source = source.replace('phases[1]["mismatches"]', 'phases[-1]["mismatches"]')
    return source


if __name__ == "__main__":
    target = ROOT / "run_dense_paired_matrix_actual.py"
    with target.open("x") as stream:
        stream.write(make_runner())
    print(target.name)
