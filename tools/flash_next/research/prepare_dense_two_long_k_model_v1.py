import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent


def controller_source(source):
    source = source.replace("dense_tile_parallel3_long_k_transform", "dense_tile_parallel2_long_k_transform")
    source = replace_once(source, "from run_dense_long_k_native_v3 import BUILD",
                          "from run_dense_two_long_k_ns_native_v1 import COMPUTE as BUILD, BUILD as PRECISE_BUILD")
    source = source.replace("dense-tile-parallel3-long-k-model-v1", "dense-tile-parallel2-long-k-model-v1")
    source = source.replace("benchmark_dense_long_k.py", "benchmark_dense_two_long_k_model_v1.py")
    source = source.replace("prepare_dense_long_k_model_v1.py", "prepare_dense_two_long_k_model_v1.py")
    source = replace_once(source, '":384,1,1:"', '":256,1,1:"')
    source = replace_once(source, '"dense-tile-parallel3-long-k-native-v3-outcome.json"',
                          '"dense-tile-parallel2-long-k-ns-native-v1-outcome.json"')
    source = replace_once(
        source,
        '    expected_provider_sha = native["provider_sha256"] if flag else build["baseline_provider_sha256"]\n'
        '    if (not native["native_model_admission_eligible"] or sha(provider) != expected_provider_sha\n'
        '            or sha(BUILD / "libonnxruntime_providers_webgpu.so") != native["provider_sha256"]):',
        '    precise = json.loads((PRECISE_BUILD / "build-manifest.json").read_text())\n'
        '    expected_provider_sha = precise["unmodified_candidate_provider_sha256"] if flag else build["baseline_provider_sha256"]\n'
        '    if (not native["native_model_admission_eligible"] or not native["nanosecond_timing"]\n'
        '            or not precise["all_candidate_compute_code_unchanged"] or not precise["unmodified_candidate_byte_identical"]\n'
        '            or precise["original_compute_carrier_sha256"] != build["baseline_provider_sha256"]\n'
        '            or sha(PRECISE_BUILD / "libonnxruntime_providers_webgpu.so") != native["provider_sha256"]\n'
        '            or sha(BUILD / "libonnxruntime_providers_webgpu.so") != precise["unmodified_candidate_provider_sha256"]\n'
        '            or sha(provider) != expected_provider_sha):')
    source = replace_once(source, 'GUARD, native_path, launcher, provider, BUILD / "build-manifest.json",',
                          'GUARD, native_path, launcher, provider, BUILD / "build-manifest.json",\n'
                          '        PRECISE_BUILD / "build-manifest.json", PRECISE_BUILD / "libonnxruntime_providers_webgpu.so",')
    return source


def main():
    native_path = ROOT / "dense-tile-parallel2-long-k-ns-native-v1-outcome.json"
    native = json.loads(native_path.read_text())
    if (native["status"] != "dense_long_k_fixed_process_native_golden_qualified"
            or not native["native_model_admission_eligible"] or not native["nanosecond_timing"]):
        raise RuntimeError("Complete independent full-native qualification required")
    parent = ROOT / "run_dense_long_k_model_v1.py"
    wrapper_parent = ROOT / "benchmark_dense_long_k.py"
    output = ROOT / "run_dense_two_long_k_model_v1.py"
    wrapper = ROOT / "benchmark_dense_two_long_k_model_v1.py"
    report = ROOT / "dense-two-long-k-model-v1-generation.json"
    if any(path.exists() for path in (output, wrapper, report)):
        raise FileExistsError("Fresh two-worker model scope required")
    sources = {
        output: controller_source(parent.read_text()),
        wrapper: wrapper_parent.read_text().replace("ORT_WEBGPU_DENSE_TILE_PARALLEL3_LONG_K",
                                                   "ORT_WEBGPU_DENSE_TILE_PARALLEL2_LONG_K"),
    }
    for path, source in sources.items():
        compile(source, str(path), "exec")
    for path, source in sources.items():
        with path.open("x") as stream:
            stream.write(source)
    with report.open("x") as stream:
        json.dump({
            "status": "generated_not_model_qualified",
            "model_provider_definition": "Unmodified63276computequalifiedwithNscontext46dc;naturalmodeluses63276versusoriginal3a,notNsprofiles.",
            "same_natural_request_scope_and_gates": True,
            "source_sha256": {str(path): sha(path) for path in
                              (Path(__file__), native_path, parent, wrapper_parent, output, wrapper)},
        }, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
