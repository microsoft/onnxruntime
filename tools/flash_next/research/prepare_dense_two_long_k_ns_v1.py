import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent


def controller_source(source):
    source = source.replace("qwen38-int4-dense-tile-parallel2-long-k/build-v1",
                            "qwen38-int4-dense-tile-parallel2-long-k-timestamp-ns/build-v1")
    source = source.replace('VARIANT = "dense-tile-parallel2-long-k-v1"',
                            'VARIANT = "dense-tile-parallel2-long-k-ns-v1"')
    source = replace_once(
        source, "from run_dawn_scope_qualification_v1 import PARENT, REPOS,",
        "from gpu_timestamp_ns_transform import precise_duration_us\n"
        "from run_dawn_scope_qualification_v1 import REPOS,")
    source = replace_once(source, "\nBUILD = REPOS /",
                          '\nPARENT = REPOS / "qwen38-int4-gpu-timestamp-ns/build-v1"\n'
                          "\nBUILD = REPOS /")
    source = replace_once(
        source, "    original = PARENT / \"libonnxruntime_providers_webgpu.so\"",
        '    original = PARENT / "libonnxruntime_providers_webgpu.so"\n'
        '    control_manifest = json.loads((PARENT / "build-manifest.json").read_text())\n'
        '    if (not manifest["nanosecond_timing"] or not manifest["all_candidate_compute_code_unchanged"]\n'
        '            or not manifest["original_precise_control_byte_identical"]\n'
        '            or control_manifest["baseline_provider_sha256"] !=\n'
        '            "3a283ecfb892613b19a6f49d15b726e928b41027c8b3d838bd8cd1dfed9939d3"):\n'
        '        raise RuntimeError("Missing precise-control original compute provenance")')
    source = replace_once(
        source, 'or manifest["baseline_provider_sha256"] !=\n'
        '            "3a283ecfb892613b19a6f49d15b726e928b41027c8b3d838bd8cd1dfed9939d3"):',
        'or manifest["baseline_provider_sha256"] != control_manifest["provider_sha256"]):')
    source = replace_once(
        source, '    return [event["dur"] for event in kernels[16:]]',
        '    precise = [precise_duration_us(event) for event in kernels]\n'
        '    return precise[16:]')
    source = replace_once(
        source, '"operator_ratios_are_not_model_decode_ratios": True,',
        '"operator_ratios_are_not_model_decode_ratios": True,\n'
        '        "nanosecond_timing": True, "all_candidate_compute_code_unchanged": True,\n'
        '        "control_definition": manifest["control_definition"],')
    return source.replace("dense_tile_parallel2_long_k_actual_operator_screen_completed",
                           "precise_gpu_dense_tile_parallel2_long_k_operator_screen_completed")


def main():
    parent = ROOT / "run_dense_tile_parallel2_long_k_v1.py"
    output = ROOT / "run_dense_two_long_k_ns_v1.py"
    report = ROOT / "dense-two-long-k-ns-v1-generation.json"
    if report.exists() or output.exists():
        raise FileExistsError("Fresh precise-timestamp operator generation required")
    source = controller_source(parent.read_text())
    compile(source, str(output), "exec")
    with output.open("x") as stream:
        stream.write(source)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified",
                   "reason": "Replace rounded-us observation with qualified raw Dawn nanoseconds on both original-compute control and unchanged two-worker shader. Keep144calls,16warmups,128measured,actualfixtures,finitegoldens,>1%/120gatesunchanged; preserveold118/128hold.",
                   "source_sha256": {str(path): sha(path) for path in
                                     (Path(__file__), ROOT / "gpu_timestamp_ns_transform.py",
                                      ROOT / "build_dense_two_long_k_ns_v1.py", parent, output)}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
