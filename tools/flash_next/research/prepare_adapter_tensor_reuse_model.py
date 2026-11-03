from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def base(source):
    return (source.replace("ready-replay-key-trim-v2", "adapter-tensor-vector-reuse-v7")
            .replace("qwen38-int4-ready-replay-key-trim/build-v2", "qwen38-int4-adapter-tensor-reuse/build-v7")
            .replace("build_ready_replay_key_trim", "build_adapter_tensor_reuse")
            .replace("run_ready_replay_key", "run_adapter_tensor_reuse")
            .replace("trim", "reuse"))


def model_source(source):
    source = base(source)
    source = source.replace("ORT_WEBGPU_READY_REPLAY_KEY_STATS_FILE", "ORT_WEBGPU_ADAPTER_TENSOR_VECTOR_STATS_FILE")
    source = source.replace("-key-stats.jsonl", "-adapter-stats.jsonl")
    source = replace_once(source,
        '    native_path = ROOT / f"{VARIANT}-native-gpu7-20261031.qualification.json"',
        '    native_path = ROOT / f"{VARIANT}-native-gpu7-20261101.qualification.json"')
    start = source.index('    assert native["status"] ==')
    end = source.index('    for name, digest in native["source_sha256"].items():', start)
    source = source[:start] + '''    assert native["status"] == "adapter_tensor_vector_process_modes_dense_gdn_and_invalid_selector_qualified"
    assert native["finite_bitwise_comparisons"] == 237408768
    assert native["process_modes_tested"] == [0, 1]
    assert native["invalid_selector_rejected_in_fresh_process"]
    assert native["all_152_original_objects_byte_identical"]
    assert native["original_expanded_gdn_reference_tested"]
''' + source[end:]
    source = replace_once(source, '    assert sha(build_path) == native["build_manifest_sha256"]',
        '''    assert sha(build_path) == native["build_manifest_sha256"]
    assert build["all_152_original_source_objects_byte_identical"]
    assert build["complete_sdk_copy_for_relative_include_wiring"]''')
    start = source.index("    key_path =")
    end = source.index('    log = prefix.with_suffix(".log").read_text()', start)
    source = source[:start] + '''    vector_path = ROOT / (prefix.name + "-adapter-stats.jsonl")
    vector_coverage = None
    if args.mode == "reuse":
        rows = [json.loads(line) for line in vector_path.read_text().splitlines()]
        assert rows and all(row["operator"] == "AdapterTensorVectors" and
                            row["requests"] == row["returns"] and
                            row["nonempty_capacity_reuses"] > 0 and
                            row["retained_input_elements"] == row["retained_output_elements"] == 0
                            for row in rows), rows
        vector_coverage = dict(requests=sum(row["requests"] for row in rows),
                               returns=sum(row["returns"] for row in rows),
                               nonempty_capacity_reuses=sum(row["nonempty_capacity_reuses"] for row in rows))
        if args.phase == "coverage":
            assert vector_coverage["requests"] >= result["dense_replay_hits"]
        statistics.append(vector_path)
    else:
        assert not vector_path.exists()
''' + source[end:]
    return replace_once(source, "ready_replay_key_coverage=key_coverage,",
                        "adapter_tensor_vector_coverage=vector_coverage,")


def main():
    for old, new, transform in (
        ("run_ready_replay_key_model.py", "run_adapter_tensor_reuse_model.py", model_source),
        ("run_ready_replay_key_screen.py", "run_adapter_tensor_reuse_screen.py", base)):
        target = ROOT / new
        with target.open("x") as stream:
            stream.write(transform((ROOT / old).read_text()))
        print(target.name)


if __name__ == "__main__":
    main()
