import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent


def builder_source(source):
    source = replace_once(
        source, "qwen38-int4-dense-tile-parallel2-long-k-timestamp-ns/build-v2",
        "qwen38-int4-hc-up-packed160-timestamp-ns/build-v1")
    source = replace_once(
        source, "qwen38-int4-dense-tile-parallel2-long-k/build-v1",
        "qwen38-int4-hc-up-packed160/build-v1")
    source = source.replace("original_dense_object_byte_identical", "original_hc_up_object_byte_identical")
    return source.replace('"dense_compare"', '"hc_up_compare"')


def controller_source(source):
    source = replace_once(
        source, "qwen38-int4-hc-up-packed160/build-v1",
        "qwen38-int4-hc-up-packed160-timestamp-ns/build-v1")
    source = replace_once(source, 'VARIANT = "hc-up-packed160-v1"',
                          'VARIANT = "hc-up-packed160-ns-v1"')
    source = replace_once(source, "from run_dawn_scope_qualification_v1 import PARENT, REPOS,",
                          "from gpu_timestamp_ns_transform import precise_duration_us\n"
                          "from run_dawn_scope_qualification_v1 import REPOS,")
    source = replace_once(source, "\nBUILD = REPOS /",
                          '\nPARENT = REPOS / "qwen38-int4-gpu-timestamp-ns/build-v1"\n'
                          "\nBUILD = REPOS /")
    source = replace_once(
        source, '    original = PARENT / "libonnxruntime_providers_webgpu.so"',
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
    source = replace_once(source, '    return [event["dur"] for event in kernels[33::2]]',
                          '    precise = [precise_duration_us(event) for event in kernels]\n'
                          '    return precise[33::2]')
    source = replace_once(
        source, '"status": "hc_up_packed160_actual_operator_screen_completed",',
        '"status": "precise_gpu_hc_up_packed160_actual_operator_screen_completed",\n'
        '        "operator_ratios_are_not_model_decode_ratios": True,\n'
        '        "nanosecond_timing": True, "all_candidate_compute_code_unchanged": True,\n'
        '        "control_definition": manifest["control_definition"],')
    return source


def main():
    parent_builder = ROOT / "build_dense_two_long_k_ns_v2.py"
    parent_controller = ROOT / "run_hc_up_packed160_v1.py"
    builder = ROOT / "build_hc_up_packed160_ns_v1.py"
    controller = ROOT / "run_hc_up_packed160_ns_v1.py"
    report_path = ROOT / "hc-up-packed160-ns-v1-generation.json"
    if any(path.exists() for path in (builder, controller, report_path)):
        raise FileExistsError("Fresh precise HC-up observation required")
    sources = {builder: builder_source(parent_builder.read_text()),
               controller: controller_source(parent_controller.read_text())}
    for path, source in sources.items():
        compile(source, str(path), "exec")
    for path, source in sources.items():
        with path.open("x") as stream:
            stream.write(source)
    with report_path.open("x") as stream:
        json.dump({
            "status": "generated_not_qualified",
            "reason": "New qualified raw-ns measurement on both original3a packed2 control and unchanged packed160 shader; preserve old87/128hold. Exact same 288events,128secondcalls,16warmuppairs,finitegoldens,>1%/120gates.",
            "source_sha256": {str(path): sha(path) for path in
                              (Path(__file__), parent_builder, parent_controller, builder, controller,
                               ROOT / "gpu_timestamp_ns_transform.py")},
        }, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
