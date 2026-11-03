import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
VARIANT = "dense-tile-parallel2-long-k-ns-native-v1"


def controller_source(source):
    source = source.replace("dense_tile_parallel3_long_k_transform", "dense_tile_parallel2_long_k_transform")
    source = source.replace("from dense_tile_parallel3_transform", "from dense_tile_parallel2_transform")
    source = source.replace("from run_dense_tile_parallel3_v1 import matched",
                            "from run_dense_tile_parallel2_long_k_v1 import matched")
    source = replace_once(source, "from run_dawn_scope_qualification_v1 import PARENT, REPOS,",
                          "from gpu_timestamp_ns_transform import precise_duration_us\n"
                          "from run_dawn_scope_qualification_v1 import REPOS,")
    source = replace_once(source, "\nBUILD = REPOS /",
                          '\nPARENT = REPOS / "qwen38-int4-gpu-timestamp-ns/build-v1"\n'
                          'COMPUTE = REPOS / "qwen38-int4-dense-tile-parallel2-long-k/build-v1"\n'
                          "\nBUILD = REPOS /")
    source = replace_once(source, "qwen38-int4-dense-tile-parallel3-long-k/build-v1",
                          "qwen38-int4-dense-tile-parallel2-long-k-timestamp-ns/build-v2")
    source = source.replace("native-full-v3", "native-full-v1").replace("native-proof-v3", "native-proof-v1")
    source = replace_once(source, 'VARIANT = "dense-tile-parallel3-long-k-native-v3"',
                          f'VARIANT = "{VARIANT}"')
    source = replace_once(source, '":384,1,1:"', '":256,1,1:"')
    source = replace_once(source, '    private = [event for event in kernels',
                          '    for event in kernels:\n'
                          '        precise_duration_us(event)\n'
                          '    private = [event for event in kernels')
    source = replace_once(source, 'manifest["supported_n"] != 2560', 'manifest["supported_n"] != [2560]')
    source = replace_once(
        source, '    original = PARENT / "libonnxruntime_providers_webgpu.so"',
        '    original = PARENT / "libonnxruntime_providers_webgpu.so"\n'
        '    control = json.loads((PARENT / "build-manifest.json").read_text())\n'
        '    if (not manifest["nanosecond_timing"] or not manifest["all_candidate_compute_code_unchanged"]\n'
        '            or not manifest["original_precise_control_byte_identical"]\n'
        '            or control["baseline_provider_sha256"] !=\n'
        '            "3a283ecfb892613b19a6f49d15b726e928b41027c8b3d838bd8cd1dfed9939d3"):\n'
        '        raise RuntimeError("Missing original-compute precise-control provenance")')
    source = replace_once(
        source, 'or manifest["baseline_provider_sha256"] !=\n'
        '            "3a283ecfb892613b19a6f49d15b726e928b41027c8b3d838bd8cd1dfed9939d3"',
        'or manifest["baseline_provider_sha256"] != control["provider_sha256"]')
    source = replace_once(source, '(BUILD / "matmul_nbits.cc")', '(COMPUTE / "matmul_nbits.cc")')
    source = replace_once(source, '"dense_tile_parallel3_transform.py", ROOT / "dense_tile_parallel3_narrow_transform.py"',
                          '"dense_tile_parallel2_transform.py", ROOT / "gpu_timestamp_ns_transform.py"')
    source = source.replace('"run_dense_tile_parallel3_v1.py"', '"run_dense_two_long_k_ns_v2.py"')
    source = replace_once(source, '("dense-tile-parallel3-v1-outcome.json",\n'
                          '                     "dense-tile-parallel3-narrow-v1-outcome.json")',
                          '("dense-tile-parallel2-v1-outcome.json",\n'
                          '                     "dense-tile-parallel2-long-k-ns-v2-outcome.json")')
    source = replace_once(
        source, '        proof = json.loads(path.read_text())\n        windows = {}',
        '        proof = json.loads(path.read_text())\n'
        '        if "ns-v2" in filename and (not proof["operator_support"]\n'
        '                or not proof["nanosecond_timing"] or proof["provider_sha256"] != manifest["provider_sha256"]):\n'
        '            raise RuntimeError("Missing unchanged-gate precise operator admission")\n'
        '        windows = {}')
    return replace_once(source, '"native_model_admission_eligible": True, "performance_qualified": False,',
                        '"native_model_admission_eligible": True, "performance_qualified": False,\n'
                        '        "nanosecond_timing": True, "all_candidate_compute_code_unchanged": True,')


def main():
    operator_path = ROOT / "dense-tile-parallel2-long-k-ns-v2-outcome.json"
    operator = json.loads(operator_path.read_text())
    if not operator["operator_support"] or not operator["nanosecond_timing"]:
        raise RuntimeError("No supported precise operator; no full-native generation")
    parents = [ROOT / name for name in (
        "dense_long_k_fixed_native_v3.cpp", "build_dense_long_k_native_v3.py", "run_dense_long_k_native_v3.py")]
    outputs = [ROOT / name for name in (
        "dense_two_long_k_ns_native_v1.cpp", "build_dense_two_long_k_ns_native_v1.py",
        "run_dense_two_long_k_ns_native_v1.py")]
    report = ROOT / "dense-two-long-k-ns-native-v1-generation.json"
    if any(path.exists() for path in (*outputs, report)):
        raise FileExistsError("Fresh full-native generation required")
    native = parents[0].read_text().replace("ORT_WEBGPU_DENSE_TILE_PARALLEL3_LONG_K",
                                          "ORT_WEBGPU_DENSE_TILE_PARALLEL2_LONG_K")
    builder = parents[1].read_text().replace(
        "qwen38-int4-dense-tile-parallel3-long-k/native-full-v3",
        "qwen38-int4-dense-tile-parallel2-long-k-timestamp-ns/native-full-v1").replace(
            "dense_long_k_fixed_native_v3.cpp", outputs[0].name)
    sources = [native, builder, controller_source(parents[2].read_text())]
    for path, source in zip(outputs, sources, strict=True):
        if path.suffix == ".py":
            compile(source, str(path), "exec")
    for path, source in zip(outputs, sources, strict=True):
        with path.open("x") as stream:
            stream.write(source)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified",
                   "coverage": "Same 27fixtures,12independentfixedprocesses,FP32/multirowfallback,profilingoffreplay,pending1/64,116582784finiteelementsperphase. Only private selector identity/control/timestamp consumer differ.",
                   "source_sha256": {str(path): sha(path) for path in
                                     (Path(__file__), operator_path, *parents, *outputs)}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
