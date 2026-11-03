import json
from pathlib import Path
import subprocess

from build_dense_paired_matrix_reset import PROGRAM as PARENT_PROGRAM, transform as parent_transform, SOURCE, ROOT
from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_DENSE_PAIRED_MATRIX_BATCH8"
CLASS = "DensePairedMatrixBatch8Program"
OUTPUT = Path("/workspace/qwen38-int4-dense-paired-matrix-batch8/build-v1")
STORAGE_BYTES = 3 * 8192 * 2 + 8192 * 4 + 4096 * 2 + 512 * 2


def program_source():
    source = PARENT_PROGRAM.replace("DensePairedMatrixReset", "DensePairedMatrixBatch8")
    prefix = source[:source.index("let sg = local_idx / 32u;")]
    assert prefix.count("array<f16, 1024>") == 3
    prefix = prefix.replace("array<f16, 1024>", "array<f16, 8192>")
    prefix = replace_once(prefix, "array<f32, 1024>", "array<f32, 8192>")
    prefix = replace_once(prefix, "var<workgroup> partials:",
                          "var<workgroup> half_dots: array<f16, 4096>;\nvar<workgroup> partials:")
    start = source.index("      var inputs: array<vec4<f16>, 4>;")
    end_marker = "      workgroupBarrier();\n    }\n  }\n"
    end = source.index(end_marker, start) + len("      workgroupBarrier();\n")
    stage = source[start:end]
    stage = stage.replace("sg * 8u", "stream_group * 8u")
    stage = replace_once(stage,
        "        if (half == 0u) { dots[j] = dot_value; }\n"
        "        else { sums[j] += dots[j] + dot_value; }",
        "        half_dots[slot * 512u + stream_group * 128u + item] = dot_value;")
    tail = source[source.index("for (var j = 0u; j < 4u; j++) {\n  let item = lane"):]
    tail = "if (sg < 4u) {\n" + tail.replace("sg * 8u", "stream_group * 8u")
    tail = replace_once(tail, "}\nworkgroupBarrier();\nif (local_idx < 16u)",
                        "}\n}\nworkgroupBarrier();\nif (local_idx < 16u)")
    return prefix + """let sg = local_idx / 32u;
let lane = local_idx % 32u;
let slot = sg / 4u;
let stream_group = sg % 4u;
let word = slot / 2u;
let half = slot % 2u;
let base = sg * 256u;
var totals: array<f16, 4>;
for (var j = 0u; j < 4u; j++) { totals[j] = f16(0); }
for (var tile = 0u; tile * 32u < uniforms.blocks; tile++) {
""" + stage + """  if (sg < 4u) {
    for (var j = 0u; j < 4u; j++) {
      let item = lane + j * 32u;
      var sum = f16(0);
      for (var original_word = 0u; original_word < 4u; original_word++) {
        let address = original_word * 1024u + stream_group * 128u + item;
        sum += half_dots[address] + half_dots[address + 512u];
      }
      totals[j] += sum;
    }
  }
}
""" + tail


PROGRAM = program_source()


def transform(source):
    result = replace_once(parent_transform(source), PARENT_PROGRAM, PROGRAM)
    result = result.replace("ORT_WEBGPU_DENSE_PAIRED_MATRIX_RESET", FLAG)
    result = replace_once(result, "DensePairedMatrixResetProgram program;", CLASS + " program;")
    result = replace_once(result, "maxComputeWorkgroupStorageSize >= 11264",
                          f"maxComputeWorkgroupStorageSize >= {STORAGE_BYTES} &&\n"
                          "      context.DeviceLimits().maxComputeWorkgroupSizeX >= 1024 &&\n"
                          "      context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 1024")
    return replace_once(result, ".SetWorkgroupSize(128).SetDispatchGroupSize(N_ / 16);",
                         ".SetWorkgroupSize(1024).SetDispatchGroupSize(N_ / 16);")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(model_eligible=False, shared_storage_bytes=STORAGE_BYTES,
             physical_threads=1024, subgroups=32, concurrent_word_half_slots=8,
             original_k_streams=32, streams_per_slot_subgroup=8, output_columns=16,
             original_word_half_and_tile_rounding_preserved=True,
             explicit_private_array_initialization=True,
             original_provider_full_kernel_comparison_required=True))
    cpp = OUTPUT / "paired_matrix_compare.cpp"
    cpp.write_text((ROOT / "dense_paired_matrix_compare.cpp").read_text())
    binary = OUTPUT / "paired_matrix_compare"
    subprocess.run(["c++", "-O3", "-std=c++20", str(cpp),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(binary)], check=True)
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True,
                  paired_comparator_sha256=sha(binary))
    report["source_sha256"]["paired_matrix_compare.cpp"] = sha(cpp)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
