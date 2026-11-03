import json

from build_qmoe_node_replay import replace_once, sha
from prepare_webgpu_int4_metrics import ROOT


PROFILE_HELPERS = '''
def model_programs(events, flag):
    if flag not in (0, 1):
        raise ValueError("Invalid long-K model profile scope")
    kernels = [event for event in events if event.get("cat") == "Kernel"]
    private = [event for event in kernels if event.get("name", "").rsplit("&", 1)[-1] == NAME]
    if len(private) != (48 if flag else 0):
        raise RuntimeError("Wrong actual long-K model dispatch count")
    for event in private:
        if (event.get("pid") != -1 or event.get("tid") != -1 or event.get("ph") != "X"
                or not math.isfinite(event.get("dur", 0)) or event.get("dur", 0) <= 0
                or ":384,1,1:" not in event.get("args", {}).get("cache_key", "")):
            raise RuntimeError("Invalid actual long-K GPU event/workgroup")
    return dict(collections.Counter(
        "DenseColumns8DoubleBuffer" if event.get("name", "").rsplit("&", 1)[-1] == NAME
        else event.get("name", "").rsplit("&", 1)[-1] for event in kernels))


def verify_model_profiles(raw, flag):
    profiles, counts = [], []
    for sample in raw["samples"]:
        files = sample.get("decode_profile_files", [])
        if len(files) != 1 or sample["profiled_decode_tokens"] != 1 or sample["profile_after_tokens"] != 35:
            raise RuntimeError("Expected exactly one late decode GPU profile per request")
        profile = host_path(files[0])
        counts.append(model_programs(json.loads(profile.read_text()), flag))
        profiles.append(profile)
    return profiles, counts


'''


def model_source(source):
    source = source.replace("context_flag_cache", "dense_long_k").replace(
        "context-flag-cache-v1", "dense-tile-parallel3-long-k-model-v1")
    source = replace_once(source, "import json\n", "import collections\nimport json\nimport math\n")
    source = replace_once(source, "from run_dense_long_k_native import BUILD, FLAG",
                          "from dense_tile_parallel3_long_k_transform import FLAG, NAME\n"
                          "from run_dense_long_k_native_v3 import BUILD\n"
                          "from run_argmax_parallel_bitwise import host_path\n"
                          "from run_dawn_scope_qualification_v1 import PARENT, container")
    source = replace_once(
        source,
        '    shell = replace_once(shell,\n'
        '        "/workspace/qwen38-int4-fc2-private-a/build-v1/libonnxruntime_providers_webgpu.so",\n'
        '        "/workspace/qwen38-int4-context-flag-cache/build-v1/libonnxruntime_providers_webgpu.so")',
        '    provider = (BUILD if flag else PARENT) / "libonnxruntime_providers_webgpu.so"\n'
        '    shell = replace_once(shell,\n'
        '        "/workspace/qwen38-int4-fc2-private-a/build-v1/libonnxruntime_providers_webgpu.so", container(provider))\n'
        '    if trace:\n'
        '        shell = replace_once(shell, "--tokens 256 --repetitions 3",\n'
        '                             f"--tokens 36 --repetitions 1 --profile-decode-after 35 "\n'
        '                             f"--profile-prefix /evidence/webgpu-perf-oct20/{prefix.name}-profile")')
    source = replace_once(source, "def run(context, full, flag, trial, trace):", PROFILE_HELPERS +
                          "def run(context, full, flag, trial, trace):")
    source = replace_once(source, 'native_path = ROOT / "dense-tile-parallel3-long-k-model-v1-native-outcome.json"',
                          'native_path = ROOT / "dense-tile-parallel3-long-k-native-v3-outcome.json"')
    source = replace_once(source, '"dense_long_k_native_finite_bitwise_qualified"',
                          '"dense_long_k_fixed_process_native_golden_qualified"')
    source = replace_once(
        source,
        '    if sha(BUILD / "libonnxruntime_providers_webgpu.so") != native["provider_sha256"]:\n'
        '        raise RuntimeError("Changed native-qualified provider")',
        '    build = json.loads((BUILD / "build-manifest.json").read_text())\n'
        '    provider = (BUILD if flag else PARENT) / "libonnxruntime_providers_webgpu.so"\n'
        '    expected_provider_sha = native["provider_sha256"] if flag else build["baseline_provider_sha256"]\n'
        '    if (not native["native_model_admission_eligible"] or sha(provider) != expected_provider_sha\n'
        '            or sha(BUILD / "libonnxruntime_providers_webgpu.so") != native["provider_sha256"]):\n'
        '        raise RuntimeError("Changed native-qualified or independent original provider")')
    source = replace_once(source, 'raw["plugin_sha256"] != native["provider_sha256"]',
                          'raw["plugin_sha256"] != expected_provider_sha')
    source = replace_once(
        source,
        'raw["profiling"] or raw["profiling_scope"] is not None or raw["comparison"] is not None or\n'
        '            raw["profile_decode_after_tokens"] is not None',
        'raw["profiling"] != bool(trace) or raw["profiling_scope"] != ("decode_only" if trace else None) or\n'
        '            raw["comparison"] is not None or raw["profile_decode_after_tokens"] != (35 if trace else None)')
    source = replace_once(source, "context, 256)\n    log =", "context, 36 if trace else 256)\n    log =")
    source = replace_once(source, "    sources = (Path(__file__),",
                          "    profiles, program_counts = verify_model_profiles(raw, flag) if trace else ([], [])\n"
                          "    sources = (Path(__file__),")
    source = replace_once(source, 'GUARD, native_path, launcher, prefix.with_suffix(".json"),',
                          'GUARD, native_path, launcher, provider, BUILD / "build-manifest.json",\n'
                          '        ROOT / "prepare_dense_long_k_model_v1.py", *profiles, prefix.with_suffix(".json"),')
    source = replace_once(source, "started_ns=started, completed_ns=ended, quality_qualified=False,",
                          "started_ns=started, completed_ns=ended, quality_qualified=False,\n"
                          "        original_provider_control=flag == 0, provider_sha256=expected_provider_sha,\n"
                          "        normalized_program_counts=program_counts,")
    source = replace_once(source, '            sources = [Path(__file__), *[prefix_for(',
                          '            routing = [json.loads(prefix_for(context, True, flag, 0, 1).with_suffix(\n'
                          '                ".qualification.json").read_text()) for flag in (0, 1)]\n'
                          '            if routing[0]["normalized_program_counts"] != routing[1]["normalized_program_counts"]:\n'
                          '                raise RuntimeError("Unchanged model GPU program counts differ after exact long-K substitution")\n'
                          '            sources = [Path(__file__), *[prefix_for(')
    return source


def main():
    parent = ROOT / "run_context_flag_cache_model.py"
    output = ROOT / "run_dense_long_k_model_v1.py"
    report = ROOT / "dense-long-k-model-v1-generation.json"
    if output.exists() or report.exists():
        raise FileExistsError("Fresh long-K model generation required")
    native = json.loads((ROOT / "dense-tile-parallel3-long-k-native-v3-outcome.json").read_text())
    if native["status"] != "dense_long_k_fixed_process_native_golden_qualified" or not native["native_model_admission_eligible"]:
        raise RuntimeError("Independent native qualification required before model preparation")
    source = model_source(parent.read_text())
    compile(source, str(output), "exec")
    with output.open("x") as stream:
        stream.write(source)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_model_qualified",
                   "independent_original_provider_controls": True,
                   "source_sha256": {str(path): sha(path) for path in
                                     (ROOT / __file__, parent, output, ROOT / "benchmark_dense_long_k.py")}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
