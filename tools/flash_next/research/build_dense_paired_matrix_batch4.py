import json
from pathlib import Path
import subprocess

from build_dense_paired_matrix_batch8 import PROGRAM as PARENT_PROGRAM, transform as parent_transform, SOURCE, ROOT
from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_DENSE_PAIRED_MATRIX_BATCH4"
CLASS = "DensePairedMatrixBatch4Program"
OUTPUT = Path("/workspace/qwen38-int4-dense-paired-matrix-batch4/build-v1")
STORAGE_BYTES = 3 * 4096 * 2 + 4096 * 4 + 2048 * 2 + 512 * 2


def program_source():
    source = PARENT_PROGRAM.replace("DensePairedMatrixBatch8", "DensePairedMatrixBatch4")
    source = source.replace("array<f16, 8192>", "array<f16, 4096>")
    source = replace_once(source, "array<f32, 8192>", "array<f32, 4096>")
    source = replace_once(source, "half_dots: array<f16, 4096>", "half_dots: array<f16, 2048>")
    source = replace_once(source, "let word = slot / 2u;\n", "")
    source = replace_once(source,
        "for (var tile = 0u; tile * 32u < uniforms.blocks; tile++) {\n",
        "for (var tile = 0u; tile * 32u < uniforms.blocks; tile++) {\n"
        "  var sums: array<f16, 4>;\n"
        "  for (var j = 0u; j < 4u; j++) { sums[j] = f16(0); }\n"
        "  for (var batch = 0u; batch < 2u; batch++) {\n"
        "    let word = batch * 2u + slot / 2u;\n")
    return replace_once(source, """  if (sg < 4u) {
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
""", """    if (sg < 4u) {
      for (var j = 0u; j < 4u; j++) {
        let item = lane + j * 32u;
        for (var original_word = 0u; original_word < 2u; original_word++) {
          let address = original_word * 1024u + stream_group * 128u + item;
          sums[j] += half_dots[address] + half_dots[address + 512u];
        }
      }
    }
    workgroupBarrier();
  }
  if (sg < 4u) {
    for (var j = 0u; j < 4u; j++) { totals[j] += sums[j]; }
  }
}
""")


PROGRAM = program_source()


def transform(source):
    result = replace_once(parent_transform(source), PARENT_PROGRAM, PROGRAM)
    result = result.replace("ORT_WEBGPU_DENSE_PAIRED_MATRIX_BATCH8", FLAG)
    result = replace_once(result, "DensePairedMatrixBatch8Program program;", CLASS + " program;")
    result = replace_once(result, "maxComputeWorkgroupStorageSize >= 91136",
                          f"maxComputeWorkgroupStorageSize >= {STORAGE_BYTES}")
    result = replace_once(result, "maxComputeWorkgroupSizeX >= 1024", "maxComputeWorkgroupSizeX >= 512")
    result = replace_once(result, "maxComputeInvocationsPerWorkgroup >= 1024",
                          "maxComputeInvocationsPerWorkgroup >= 512")
    return replace_once(result, ".SetWorkgroupSize(1024).SetDispatchGroupSize(N_ / 16);",
                        ".SetWorkgroupSize(512).SetDispatchGroupSize(N_ / 16);")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(model_eligible=False, shared_storage_bytes=STORAGE_BYTES,
             physical_threads=512, subgroups=16, concurrent_word_half_slots=4,
             serial_slot_batches=2, original_k_streams=32, output_columns=16,
             original_word_half_and_tile_rounding_preserved=True,
             explicit_private_array_initialization=True,
             barrier_before_shared_storage_reuse=True,
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
