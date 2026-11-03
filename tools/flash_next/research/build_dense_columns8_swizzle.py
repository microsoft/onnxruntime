import json
from pathlib import Path
import subprocess

from build_dense_columns8_double_buffer import PROGRAM as ORIGINAL_PROGRAM
from build_dense_columns8_word_unroll import PROGRAM as REPLACED_PROGRAM, transform as selector_transform, SOURCE, ROOT
from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_DENSE_COLUMNS8_SWIZZLE"
CLASS = "DenseColumns8SwizzleProgram"
OUTPUT = Path("/workspace/qwen38-int4-dense-columns8-swizzle/build-v1")


def make_program():
    source = ORIGINAL_PROGRAM.replace("DenseColumns8DoubleBuffer", "DenseColumns8Swizzle")
    source = replace_once(source, "fn decode_low_nibbles(word: u32)",
        """fn dense_swizzled_a_index(index: u32) -> u32 {
  let stream = index / 8u;
  return stream * 8u + ((index % 8u) ^ ((stream >> 1u) & 7u));
}
fn decode_low_nibbles(word: u32)""")
    for index in ("i", "stream * 8u + word * 2u", "stream * 8u + word * 2u + 1u"):
        source = replace_once(source, f"tile_a[(tile % 2u) * 256u + {index}]",
                              f"tile_a[(tile % 2u) * 256u + dense_swizzled_a_index({index})]")
    return source


PROGRAM = make_program()


def transform(source):
    result = replace_once(selector_transform(source), REPLACED_PROGRAM, PROGRAM)
    result = result.replace("ORT_WEBGPU_DENSE_COLUMNS8_WORD_UNROLL", FLAG)
    result = result.replace("word_unroll", "swizzle")
    result = replace_once(result, "DenseColumns8WordUnrollProgram program;", CLASS + " program;")
    result = replace_once(result, "  auto compute = [&]() {",
        '  const bool use_swizzle = swizzle == 1 &&\n'
        '      context.AdapterInfo().vendor == std::string_view{"nvidia"};\n'
        "  auto compute = [&]() {")
    result = replace_once(result, "    if (columns8 && swizzle == 1) {",
                          "    if (columns8 && use_swizzle) {")
    return replace_once(result, "columns8 ? (swizzle == 1 ? 8 : 7)",
                        "columns8 ? (use_swizzle ? 8 : 7)")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(model_eligible=False, output_columns=8, physical_threads=128,
             shared_storage_bytes=4608, activation_buffers=2,
             supported_double_buffer_selector_preserved=True,
             xor_activation_storage_permutation=True,
             original_dot_and_tile_order_preserved=True,
             word_unroll_not_stacked=True,
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
