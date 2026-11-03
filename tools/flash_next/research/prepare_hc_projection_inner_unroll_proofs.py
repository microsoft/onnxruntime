from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent
VARIANT = "hc-projection-inner-unroll-v1"


def base(source):
    return (source.replace("dense-ordered-unroll2-v1", VARIANT)
            .replace("qwen38-int4-dense-ordered-unroll2/build-v1", "qwen38-int4-hc-projection-inner-unroll/build-v1")
            .replace("build_dense_ordered_unroll", "build_hc_projection_inner_unroll")
            .replace("run_dense_ordered_unroll", "run_hc_projection_inner_unroll")
            .replace("dense_ordered_unroll", "hc_projection_inner_unroll")
            .replace("ORT_WEBGPU_DENSE_ORDERED_UNROLL2_STATS_FILE", "ORT_WEBGPU_HC_PROJECTION_INNER_UNROLL_STATS_FILE"))


def native_source(source):
    source = replace_once(source, '\n\nif __name__ == "__main__":\n    main()\n', "\n")
    source = replace_once(source, "from qualify_quickgelu_mul_long_context",
                          "from build_hc_projection_inner_unroll import FLAG\n"
                          "from functools import partial\n"
                          "import math\nimport onnx\n\nfrom qualify_quickgelu_mul_long_context")
    source = replace_once(source, "\ndef main(", "\ndef run_mode(")
    source = replace_once(source, "compound_candidate_not_replay_cached=False):",
                          "compound_candidate_not_replay_cached=False, process_mode=1):")
    source = replace_once(source, '    assert sha(directory / "native_compare") == build["comparator_sha256"]',
                          '''    assert sha(directory / "native_compare") == build["comparator_sha256"]
    assert build["all_152_original_source_objects_byte_identical"] and build["no_object_layout_changes"]
    for name, digest in build["source_sha256"].items():
        assert sha(directory / name) == digest, name''')
    source = replace_once(source,
        '        f"ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; exec " +',
        '''        f"{FLAG}={process_mode} "
        f"ORT_WEBGPU_HC_PROJECTION_INNER_UNROLL_STATS_FILE=/evidence/webgpu-perf-oct20/{prefix.name}-shader-stats.jsonl "
        f"ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; exec " +''')
    source = replace_once(source, "    stats_verifier(rows)",
        '''    stats_verifier(rows)
    shader_stats = ROOT / (prefix.name + "-shader-stats.jsonl")
    shaders = [json.loads(line) for line in shader_stats.read_text().splitlines()]
    assert shaders and all(row["operator"] == "HCProjectionInnerUnrollShader" and
        row["mode"] == process_mode and row["component_steps"] == 4 for row in shaders), shaders''')
    source = source.replace("replay_variant_keys_tested=not compound_candidate_not_replay_cached",
                            "replay_variant_keys_tested=False, process_static_shader_identity_tested=True")
    return source + r'''


def main():
    directory = Path("/home/sayanshaw/repos/qwen38-int4-hc-projection-inner-unroll/build-v1")
    build_path = directory / "build-manifest.json"
    build = json.loads(build_path.read_text())
    evidence = [build_path, Path(__file__), ROOT / "build_hc_projection_inner_unroll.py",
                ROOT / "prepare_hc_projection_inner_unroll_proofs.py"]
    for mode in (0, 1):
        prefix = ROOT / f"hc-projection-inner-unroll-v1-native-mode{mode}-gpu7-20261101"
        run_mode(directory=directory, prefix=prefix,
                 report_status="hc_inner_unroll_four_outputs_finite_bitwise_and_replay_qualified",
                 parent_path=ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json",
                 stats_verifier=partial(verify_static_stats, requests=96), process_mode=mode)
        evidence.extend(prefix.with_suffix(s) for s in (".qualification.json", ".guard.json", ".log"))
        evidence.extend((ROOT / (prefix.name + "-stats.jsonl"), ROOT / (prefix.name + "-shader-stats.jsonl")))
    fixtures = sorted(Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/fixtures-v1").glob("*.onnx"))
    eligible = []
    for path in fixtures:
        graph = onnx.load(path, load_external_data=False).graph
        tensor = graph.input[0].type.tensor_type
        if tensor.elem_type == onnx.TensorProto.FLOAT16 and math.prod(
                dim.dim_value for dim in tensor.shape.dim) == 10240:
            eligible.append(path)
    assert eligible
    prefix = ROOT / "hc-projection-inner-unroll-v1-invalid-gpu7-20261101"
    assert not prefix.with_suffix(".guard.json").exists()
    invalid = ROOT / (prefix.name + "-invalid.log")
    assert not invalid.exists()
    comparator, provider, fixture = [str(path).replace("/home/sayanshaw/repos", "/workspace", 1)
        for path in (directory / "native_compare", directory / "libonnxruntime_providers_webgpu.so", eligible[0])]
    target = "/evidence/webgpu-perf-oct20/" + invalid.name
    command = ("set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; "
        f"if env {FLAG}=2 ORT_WEBGPU_HC_POST_NORM_PROJECTION=1 ORT_WEBGPU_HC_PROJECTION_SILU=1 "
        f"ORT_WEBGPU_HC_EXACT_DENSE=1 {comparator} {provider} {fixture} > {target} 2>&1; then "
        "echo 'FAIL: invalid HC inner-unroll selector accepted'; exit 1; fi; "
        f"grep -F '{FLAG} must be 0 or 1' {target}")
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "120",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    evidence.extend((invalid, prefix.with_suffix(".guard.json"), prefix.with_suffix(".log"), *fixtures))
    report = dict(status="hc_projection_inner_unroll_process_modes_and_invalid_selector_qualified",
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        comparator_sha256=build["comparator_sha256"], build_manifest_sha256=sha(build_path),
        finite_bitwise_comparisons=1214287872, process_modes_tested=[0, 1],
        invalid_selector_rejected_in_fresh_process=True, all_four_outputs_compared=True,
        all_152_original_objects_byte_identical=True, private_kernel_layout_only=True,
        source_sha256={str(path): sha(path) for path in evidence}, quality_qualified=False, headline_eligible=False)
    path = ROOT / "hc-projection-inner-unroll-v1-native-gpu7-20261101.qualification.json"
    assert not path.exists()
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "finite_bitwise_comparisons")}))


if __name__ == "__main__":
    main()
'''


def model_source(source):
    source = base(source)
    source = source.replace("hc_projection_inner_unroll_process_modes_dense_gdn_and_invalid_selector_qualified",
                            "hc_projection_inner_unroll_process_modes_and_invalid_selector_qualified")
    source = replace_once(source, '    assert native["finite_bitwise_comparisons"] == 237408768',
                          '    assert native["finite_bitwise_comparisons"] == 1214287872')
    source = replace_once(source, '    assert native["original_expanded_gdn_reference_tested"]\n', "")
    source = replace_once(source, 'row["operator"] == "DenseOrderedUnrollShader"', 'row["operator"] == "HCProjectionInnerUnrollShader"')
    source = replace_once(source, 'row["mode"] == 1 and row["tile_size_k"] > 0',
                          'row["mode"] == 1 and row["component_steps"] == 4')
    return replace_once(source, 'tile_sizes=sorted({row["tile_size_k"] for row in rows})',
                        'ordered_component_steps=4')


def main():
    for old, new, transform in (
        ("run_hc_bitcast_native.py", "run_hc_projection_inner_unroll_native.py", native_source),
        ("run_dense_ordered_unroll_model.py", "run_hc_projection_inner_unroll_model.py", model_source),
        ("run_dense_ordered_unroll_screen.py", "run_hc_projection_inner_unroll_screen.py", base)):
        target = ROOT / new
        with target.open("x") as stream:
            stream.write(transform((ROOT / old).read_text()))
        print(target.name)


if __name__ == "__main__":
    main()
