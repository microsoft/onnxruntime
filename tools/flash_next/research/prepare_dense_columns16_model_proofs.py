from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def adapt(source):
    return (source.replace("dense-packed-columns-v2", "dense-columns16-v1")
            .replace("qwen38-int4-dense-packed-columns/build-v2", "qwen38-int4-dense-columns16/build-v1")
            .replace("dense_packed_columns", "dense_columns16")
            .replace("packed_columns", "columns16")
            .replace("MatMulNBitsPackedColumnsF16", "DenseColumns16"))


def model_source():
    source = adapt((ROOT / "run_dense_packed_columns_model.py").read_text())
    start = source.index('    native_path = ROOT / f"{VARIANT}-native-gpu7-{STAMP}.qualification.json"')
    stop = source.index("    candidate_native_path = native_path", start)
    source = source[:start] + '''    native_path = ROOT / "dense-columns16-actual-v1-outcome.json"
    synthetic_path = ROOT / "dense-columns16-v1-outcome.json"
    invalid_path = ROOT / "dense-columns16-v1-invalid-outcome.json"
    actual = json.loads(native_path.read_text())
    synthetic = json.loads(synthetic_path.read_text())
    invalid = json.loads(invalid_path.read_text())
    assert actual["status"] == "dense_columns16_original_provider_actual_weights_supported"
    assert synthetic["status"] == "dense_columns16_original_provider_synthetic_shapes_supported"
    assert invalid["status"] == "fresh_invalid_columns16_selector_rejected"
    assert len(actual["shapes"]) == 8 and len(synthetic["shapes"]) == 9
    assert all(len(row["phases"]) == 3 and
        [phase["mode"] for phase in row["phases"]] == ["capture", "default", "compare"] and
        all(phase["mismatches"] == 0 for phase in row["phases"]) and
        row["phases"][-1]["selected_dispatches"] == 64 for row in actual["shapes"])
    for proof in (actual, synthetic, invalid):
        for path, digest in proof["source_and_evidence_sha256"].items():
            assert sha(Path(path)) == digest, path
    parent = json.loads(parent_path.read_text())
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    build_path = DIRECTORY / "build-manifest.json"
    build = json.loads(build_path.read_text())
    provider = DIRECTORY / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == build["provider_sha256"]
    assert build["baseline_provider_sha256"] == parent["provider_sha256"]
    for name, digest in build["source_sha256"].items():
        assert sha(DIRECTORY / name) == digest, name
    for name, digest in parent["source_sha256"].items():
        assert sha(Path(name)) == digest, name
    native = dict(provider_sha256=build["provider_sha256"],
                  baseline_provider_sha256=build["baseline_provider_sha256"])
''' + source[stop:]
    source = replace_once(source, "parent_path, native_path, fp32_path,",
                          "parent_path, native_path, synthetic_path, invalid_path,")
    return source


def screen_source():
    return adapt((ROOT / "run_dense_packed_columns_screen.py").read_text())


if __name__ == "__main__":
    for name, source in (
        ("run_dense_columns16_model.py", model_source()),
        ("run_dense_columns16_screen.py", screen_source()),
    ):
        target = ROOT / name
        with target.open("x") as stream:
            stream.write(source)
        print(target.name)
