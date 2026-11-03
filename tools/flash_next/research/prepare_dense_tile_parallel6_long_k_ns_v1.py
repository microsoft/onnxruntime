import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent


def builder_source(source):
    source = source.replace("dense_tile_parallel2_long_k_transform", "dense_tile_parallel6_long_k_transform")
    source = source.replace("qwen38-int4-dense-tile-parallel2-long-k/build-v1",
                            "qwen38-int4-dense-tile-parallel6-long-k-timestamp-ns/build-v1")
    source = replace_once(source, '"physical_threads": 256, "independent_tile_workers": 2',
                          '"physical_threads": 768, "independent_tile_workers": 6')
    source = replace_once(
        source, '    source = OUTPUT / original.name',
        '    precision_root = Path("/workspace/qwen38-int4-gpu-timestamp-ns/build-v1")\n'
        '    precision_manifest = precision_root / "build-manifest.json"\n'
        '    precise = json.loads(precision_manifest.read_text())\n'
        '    qualification_path = ROOT / "gpu-timestamp-ns-v1-qualification.json"\n'
        '    qualification = json.loads(qualification_path.read_text())\n'
        '    if (not qualification["precision_qualified"] or not precise["original_context_object_byte_identical"]\n'
        '            or qualification["provider_sha256"] != precise["provider_sha256"]\n'
        '            or precise["baseline_provider_sha256"] != parent["provider_sha256"]):\n'
        '        raise RuntimeError("Unqualified precise original-compute control")\n'
        '    for path, digest in precise["source_sha256"].items():\n'
        '        if sha(Path(path)) != digest:\n'
        '            raise RuntimeError(f"Changed precision input: {path}")\n'
        '    if sha(precision_root / "libonnxruntime_providers_webgpu.so") != precise["provider_sha256"]:\n'
        '        raise RuntimeError("Changed qualified precise control")\n'
        '    context_object = "/workspace/qwen38-int4-static-node-replay/build-v8/webgpu_context.cc.o"\n'
        '    precise_object = precision_root / "webgpu_context.cc.o"\n'
        '    if link.count(context_object) != 1:\n'
        '        raise RuntimeError("Original context object is not unique")\n'
        '    link[link.index(context_object)] = str(precise_object)\n'
        '    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)\n'
        '    if sha(library) != precise["provider_sha256"]:\n'
        '        raise RuntimeError("Original-compute precise control does not reproduce")\n'
        '    source = OUTPUT / original.name')
    source = replace_once(source, '        Path(__file__), ROOT / "dense_tile_parallel3_transform.py",',
                          '        Path(__file__), precision_manifest, qualification_path, precise_object,\n'
                          '        ROOT / "dense_tile_parallel6_long_k_transform.py",\n'
                          '        ROOT / "prepare_dense_tile_parallel6_long_k_ns_v1.py",\n'
                          '        ROOT / "dense_tile_parallel3_transform.py",')
    return replace_once(
        source, '"baseline_provider_sha256": parent["provider_sha256"], "baseline_byte_identical": True,',
        '"baseline_provider_sha256": precise["provider_sha256"], "baseline_byte_identical": True,\n'
        '        "original_compute_carrier_sha256": parent["provider_sha256"],\n'
        '        "nanosecond_timing": True, "original_precise_control_byte_identical": True,\n'
        '        "control_definition": precise["control_definition"],')


def controller_source(source):
    source = source.replace("dense_tile_parallel2_long_k_transform", "dense_tile_parallel6_long_k_transform")
    source = source.replace("qwen38-int4-dense-tile-parallel2-long-k/build-v1",
                            "qwen38-int4-dense-tile-parallel6-long-k-timestamp-ns/build-v1")
    source = replace_once(source, 'VARIANT = "dense-tile-parallel2-long-k-v1"',
                          'VARIANT = "dense-tile-parallel6-long-k-ns-v1"')
    source = replace_once(source, "from run_dawn_scope_qualification_v1 import PARENT, REPOS,",
                          "from gpu_timestamp_ns_transform import precise_duration_us\n"
                          "from run_dawn_scope_qualification_v1 import REPOS,")
    source = replace_once(source, "\nBUILD = REPOS /",
                          '\nPARENT = REPOS / "qwen38-int4-gpu-timestamp-ns/build-v1"\n\nBUILD = REPOS /')
    source = replace_once(source, "threads = 256 if selected else 128", "threads = 768 if selected else 128")
    source = replace_once(source, 'manifest["physical_threads"] != 256', 'manifest["physical_threads"] != 768')
    source = replace_once(source, 'manifest["shared_storage_bytes"] != 11264',
                          'manifest["shared_storage_bytes"] != 27648')
    source = replace_once(source, '    original = PARENT / "libonnxruntime_providers_webgpu.so"',
                          '    original = PARENT / "libonnxruntime_providers_webgpu.so"\n'
                          '    precise = json.loads((PARENT / "build-manifest.json").read_text())\n'
                          '    if not manifest["nanosecond_timing"] or not manifest["original_precise_control_byte_identical"]:\n'
                          '        raise RuntimeError("Missing precise original-compute control proof")')
    source = replace_once(source, 'or manifest["baseline_provider_sha256"] !=\n'
                          '            "3a283ecfb892613b19a6f49d15b726e928b41027c8b3d838bd8cd1dfed9939d3"):',
                          'or manifest["baseline_provider_sha256"] != precise["provider_sha256"]):')
    source = replace_once(source, '    return [event["dur"] for event in kernels[16:]]',
                          '    precise_values = [precise_duration_us(event) for event in kernels]\n'
                          '    return precise_values[16:]')
    source = source.replace('"dense_tile_parallel2_transform.py"', '"dense_tile_parallel6_long_k_transform.py"')
    source = replace_once(source, '"operator_ratios_are_not_model_decode_ratios": True,',
                          '"operator_ratios_are_not_model_decode_ratios": True,\n'
                          '        "nanosecond_timing": True, "independent_tile_workers": 6,')
    return source.replace("dense_tile_parallel2_long_k_actual_operator_screen_completed",
                          "dense_tile_parallel6_long_k_precise_operator_screen_completed")


def main():
    parent_builder = ROOT / "build_dense_tile_parallel2_long_k_v1.py"
    parent_controller = ROOT / "run_dense_tile_parallel2_long_k_v1.py"
    builder = ROOT / "build_dense_tile_parallel6_long_k_ns_v1.py"
    controller = ROOT / "run_dense_tile_parallel6_long_k_ns_v1.py"
    report = ROOT / "dense-tile-parallel6-long-k-ns-v1-generation.json"
    if any(path.exists() for path in (builder, controller, report)):
        raise FileExistsError("Fresh six-worker generation required")
    sources = {builder: builder_source(parent_builder.read_text()),
               controller: controller_source(parent_controller.read_text())}
    for path, source in sources.items():
        compile(source, str(path), "exec")
    for path, source in sources.items():
        with path.open("x") as stream:
            stream.write(source)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified", "workers": 6, "waves": 1,
                   "source_sha256": {str(path): sha(path) for path in
                                     (Path(__file__), ROOT / "dense_tile_parallel6_long_k_transform.py",
                                      parent_builder, parent_controller, builder, controller)}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
