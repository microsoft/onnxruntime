import json
from pathlib import Path
import subprocess

from build_dense_columns16_spatial_double_buffer import (
    PROGRAM as PARENT_PROGRAM, transform as parent_transform, timing_comparator, SOURCE, ROOT)
from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_DENSE_COLUMNS8_SPATIAL_DOUBLE_BUFFER"
CLASS = "DenseColumns8SpatialDoubleBufferProgram"
OUTPUT = Path("/workspace/qwen38-int4-dense-columns8-spatial-double-buffer/build-v1")


def make_program():
    source = PARENT_PROGRAM.replace("DenseColumns16SpatialDoubleBuffer", "DenseColumns8SpatialDoubleBuffer")
    source = replace_once(source, "partials: array<f16, 512>", "partials: array<f16, 256>")
    source = replace_once(source, "i < 512u; i += 512u", "i < 256u; i += 256u")
    source = replace_once(source, "i < 256u; i += 512u", "i < 256u; i += 256u")
    source = replace_once(source, "let col0 = workgroup_idx * 16u", "let col0 = workgroup_idx * 8u")
    source = replace_once(source, "if (local_idx < 16u)", "if (local_idx < 8u)")
    return replace_once(source, 'y.SetByOffset("workgroup_idx * 16u + local_idx"',
                        'y.SetByOffset("workgroup_idx * 8u + local_idx"')


PROGRAM = make_program()


def transform(source):
    result = replace_once(parent_transform(source), PARENT_PROGRAM, PROGRAM)
    result = result.replace("ORT_WEBGPU_DENSE_COLUMNS16_SPATIAL_DOUBLE_BUFFER", FLAG)
    result = result.replace("spatial16", "spatial8")
    result = replace_once(result, "DenseColumns16SpatialDoubleBufferProgram program;", CLASS + " program;")
    for before, after in (("N_ % 16 == 0", "N_ % 8 == 0"),
                          ("maxComputeWorkgroupSizeX >= 512", "maxComputeWorkgroupSizeX >= 256"),
                          ("maxComputeInvocationsPerWorkgroup >= 512", "maxComputeInvocationsPerWorkgroup >= 256"),
                          ("maxComputeWorkgroupStorageSize >= 5120", "maxComputeWorkgroupStorageSize >= 4608"),
                          (".SetWorkgroupSize(512).SetDispatchGroupSize(N_ / 16);",
                           ".SetWorkgroupSize(256).SetDispatchGroupSize(N_ / 8);")):
        result = replace_once(result, before, after)
    return result


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(model_eligible=False, output_columns=8, physical_threads=256,
             shared_storage_bytes=4608, activation_buffers=2, columns_per_thread=1,
             supported_workgroup_count_preserved=True, spatial_column_concurrency=True,
             supported_double_buffer_selector_preserved=True,
             original_dot_word_tile_and_shared_partial_order_preserved=True,
             original_provider_full_kernel_comparison_required=True))
    for name, prepare in (("paired_matrix_compare", lambda source: source),
                          ("unprofiled_compare", timing_comparator)):
        cpp = OUTPUT / f"{name}.cpp"
        cpp.write_text(prepare((ROOT / "dense_paired_matrix_compare.cpp").read_text()))
        binary = OUTPUT / name
        subprocess.run(["c++", "-O3", "-std=c++20", str(cpp),
            "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
            "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
            "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
            "-lonnxruntime", "-o", str(binary)], check=True)
        report["source_sha256"][cpp.name] = sha(cpp)
        report["paired_comparator_sha256" if name == "paired_matrix_compare" else "unprofiled_comparator_sha256"] = sha(binary)
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True,
                  unprofiled_timing_includes_host_run_and_readback=True)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
