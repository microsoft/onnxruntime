from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def base(source):
    source = source.replace("cpu-greedy-argmax-lanes8-local8-v1", "dense-columns8-cpu-local8-v1")
    source = source.replace("cpu_greedy_lanes8_original_prefixes_qualified", "columns8_cpu_local8_original_prefixes_qualified")
    source = source.replace("cpu_greedy_decode_gain_both_modes", "columns8_cpu_local8_decode_gain_both_modes")
    return source.replace('"lanes8"', '"columns8"').replace("'lanes8'", "'columns8'")


def standard_source():
    source = base((ROOT / "run_cpu_greedy_argmax_lanes8_local8_standard.py").read_text())
    source = replace_once(source, "from build_cpu_greedy_argmax_lanes8 import FLAG",
        "from build_cpu_greedy_argmax_lanes8 import FLAG as CPU_FLAG\nfrom build_dense_columns8 import FLAG")
    source = replace_once(source,
        '''    if mode == "columns8":
        command[-1] = command[-1].replace(OLD_RUNTIME, str(RUNTIME).replace("/home/sayanshaw/repos", "/workspace"))''',
        '''    command[-1] = command[-1].replace(OLD_RUNTIME, str(RUNTIME).replace("/home/sayanshaw/repos", "/workspace"))
    command[-1] = f"export {CPU_FLAG}=1; " + command[-1]''')
    start = source.index('    runtime = (RUNTIME if args.mode == "columns8" else')
    stop = source.index("    assert sha(runtime) == runtime_hash", start)
    source = source[:start] + '''    runtime = RUNTIME / "libonnxruntime-genai.so"
    runtime_hash = build["runtime_sha256"]
''' + source[stop:]
    source = replace_once(source, "    prerequisites = []", '''    gpu_path = ROOT / "dense-columns8-v1-context2048-repeat-screen-20261031.qualification.json"
    gpu = json.loads(gpu_path.read_text())
    assert gpu["status"] == "repeated_columns8_decode_gain_both_modes"
    cpu_all_path = ROOT / "cpu-greedy-argmax-lanes8-local8-v1-all-contexts-20261101.qualification.json"
    cpu_all = json.loads(cpu_all_path.read_text())
    assert cpu_all["status"] == "repeated_cpu_greedy_local8_supported_all_contexts"
    for proof in (gpu, cpu_all):
        for name, digest in proof["source_sha256"].items():
            assert sha(Path(name)) == digest, name
    gpu_build = Path("/home/sayanshaw/repos/qwen38-int4-dense-columns8/build-v1")
    gpu_build_path = gpu_build / "build-manifest.json"
    gpu_manifest = json.loads(gpu_build_path.read_text())
    assert gpu_manifest["provider_sha256"] == gpu["provider_sha256"]
    for name, digest in gpu_manifest["source_sha256"].items():
        assert sha(gpu_build / name) == digest, name
    if args.mode == "columns8":
        provider = gpu_build / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == (gpu_manifest["provider_sha256"] if args.mode == "columns8" else parent["provider_sha256"])
    native = dict(provider_sha256=sha(provider), baseline_provider_sha256=parent["provider_sha256"])
    prerequisites = []''')
    source = replace_once(source,
        'assert proof["status"] == expected and proof["provider_sha256"] == native["provider_sha256"]',
        '''assert proof["status"] == expected
        expected_provider = gpu_manifest["provider_sha256"] if "-screen-" in path.name else native["provider_sha256"]
        assert proof["provider_sha256"] == expected_provider''')
    source = replace_once(source, "runtime, cpu_path, invalid_path, locality_path, template_path, controls_path,",
        "runtime, cpu_path, invalid_path, locality_path, gpu_path, cpu_all_path, gpu_build_path, template_path, controls_path,")
    source = replace_once(source, 'original_runtime_control=args.mode == "control",',
        'original_runtime_control=False, original_provider_control=args.mode == "control",\n'
        '        cpu_argmax_flag=CPU_FLAG, cpu_argmax_flag_value=1,')
    return source


def screen_source():
    source = base((ROOT / "run_cpu_greedy_argmax_lanes8_local8_screen.py").read_text())
    source = source.replace("run_cpu_greedy_argmax_lanes8_local8_standard", "run_dense_columns8_cpu_local8_standard")
    source = replace_once(source, "    build = json.loads((RUNTIME", '''    gpu_build = Path("/home/sayanshaw/repos/qwen38-int4-dense-columns8/build-v1")
    gpu_manifest = json.loads((gpu_build / "build-manifest.json").read_text())
    build = json.loads((RUNTIME''')
    source = replace_once(source, 'assert proof["original_runtime_control"] == (mode == "control")',
        '''assert not proof["original_runtime_control"]
            assert proof["original_provider_control"] == (mode == "control")
            assert proof["cpu_argmax_flag_value"] == 1
            assert "export ORT_GENAI_CPU_GREEDY_ARGMAX_LANES8=1; " in guard["command"][-1]''')
    source = replace_once(source,
        'assert proof["provider_sha256"] == raw["plugin_sha256"] == "0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4"',
        '''expected_provider = (gpu_manifest["provider_sha256"] if mode == "columns8" else
                                 "0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4")
            assert proof["provider_sha256"] == raw["plugin_sha256"] == expected_provider''')
    source = replace_once(source,
        'expected_runtime = build["runtime_sha256"] if mode == "columns8" else build["parent_runtime_sha256"]',
        'expected_runtime = build["runtime_sha256"]')
    source = replace_once(source,
        'provider_sha256="0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4",',
        'provider_sha256=gpu_manifest["provider_sha256"],\n'
        '        original_baseline_provider_sha256="0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4",')
    source = source.replace('parent_runtime_sha256=build["parent_runtime_sha256"]',
                            'parent_runtime_sha256=build["runtime_sha256"]')
    return source.replace("bracketed_separate_load_original_runtime_net_screen",
                          "bracketed_separate_load_original_provider_same_cpu_runtime_local8_net_screen")


if __name__ == "__main__":
    for name, source in (
        ("run_dense_columns8_cpu_local8_standard.py", standard_source()),
        ("run_dense_columns8_cpu_local8_screen.py", screen_source()),
    ):
        target = ROOT / name
        with target.open("x") as stream:
            stream.write(source)
        print(target.name)
