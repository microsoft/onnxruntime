import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent


def builder_source(source):
    source = replace_once(source, "qwen38-int4-dense-tile-parallel2-long-k-timestamp-ns/build-v2",
                          "qwen38-int4-hc-tile-parallel2-timestamp-ns/build-v1")
    source = replace_once(source, "qwen38-int4-dense-tile-parallel2-long-k/build-v1",
                          "qwen38-int4-hc-tile-parallel2/build-v1")
    source = source.replace("original_dense_object_byte_identical", "original_hc_object_byte_identical")
    return replace_once(source,
                        '    binary = OUTPUT / "dense_compare"\n'
                        '    binary.hardlink_to(CANDIDATE / "dense_compare")\n'
                        '    if not binary.samefile(CANDIDATE / "dense_compare"):\n'
                        '        raise RuntimeError("Native comparator alias changed")',
                        '    binary = Path("/workspace/qwen38-int4-dawn-compute-scope-reuse/native-v1/native_compare")')


def controller_source(source):
    source = replace_once(source, "from run_dawn_scope_qualification_v1 import NATIVE, PARENT, REPOS,",
                          "from gpu_timestamp_ns_transform import precise_duration_us\n"
                          "from run_dawn_scope_qualification_v1 import NATIVE, REPOS,")
    source = replace_once(source, "\nBUILD = REPOS /",
                          '\nPARENT = REPOS / "qwen38-int4-gpu-timestamp-ns/build-v1"\n\nBUILD = REPOS /')
    source = replace_once(source, "qwen38-int4-hc-tile-parallel2/build-v1",
                          "qwen38-int4-hc-tile-parallel2-timestamp-ns/build-v1")
    source = replace_once(source, 'VARIANT = "hc-tile-parallel2-v1"', 'VARIANT = "hc-tile-parallel2-ns-v1"')
    source = replace_once(source, '    return [event["dur"] for event in kernels[16:]]',
                          '    durations = [precise_duration_us(event) for event in kernels]\n'
                          '    return durations[16:]')
    source = replace_once(source, '    generation = json.loads((NATIVE / "generation.json").read_text())',
                          '    generation = json.loads((NATIVE / "generation.json").read_text())\n'
                          '    if (not build["nanosecond_timing"] or not build["all_candidate_compute_code_unchanged"]\n'
                          '            or not build["original_precise_control_byte_identical"]):\n'
                          '        raise RuntimeError("Missing precise original-compute control and unchanged HC compute")')
    return source.replace('"hc_tile_parallel2_native_cross_provider_qualified"',
                          '"hc_tile_parallel2_precise_native_cross_provider_qualified"')


def main():
    parent_builder = ROOT / "build_dense_two_long_k_ns_v2.py"
    parent_controller = ROOT / "run_hc_tile_parallel2_v1.py"
    builder = ROOT / "build_hc_tile_parallel2_ns_v1.py"
    controller = ROOT / "run_hc_tile_parallel2_ns_v1.py"
    report = ROOT / "hc-tile-parallel2-ns-v1-generation.json"
    if any(path.exists() for path in (builder, controller, report)):
        raise FileExistsError("Fresh precise HC projection observation required")
    sources = {builder: builder_source(parent_builder.read_text()),
               controller: controller_source(parent_controller.read_text())}
    for path, source in sources.items():
        compile(source, str(path), "exec")
    for path, source in sources.items():
        with path.open("x") as stream:
            stream.write(source)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified",
                   "reason": "Qualified raw-nanosecond method on original3a control and unchanged b9 HC two-worker compute. Preserve old109/128hold and thresholds; fresh36fixture math/replay before128call operator gates.",
                   "source_sha256": {str(path): sha(path) for path in
                                     (Path(__file__), parent_builder, parent_controller, builder, controller,
                                      ROOT / "gpu_timestamp_ns_transform.py")}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
