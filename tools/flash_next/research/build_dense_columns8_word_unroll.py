import json
from pathlib import Path
import subprocess

from build_dense_columns8_double_buffer import PROGRAM as PARENT_PROGRAM, transform as parent_transform, SOURCE, ROOT
from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_DENSE_COLUMNS8_WORD_UNROLL"
CLASS = "DenseColumns8WordUnrollProgram"
OUTPUT = Path("/workspace/qwen38-int4-dense-columns8-word-unroll/build-v1")


def make_program():
    source = PARENT_PROGRAM.replace("DenseColumns8DoubleBuffer", "DenseColumns8WordUnroll")
    start = source.index('    code << R"(\n    for (var word =')
    body_start = source.index(')";\n', start) + len(')";\n')
    end_marker = '    code << "    }\\n";\n'
    body_end = source.index(end_marker, body_start)
    body = source[body_start:body_end]
    replacement = ""
    for word, component in enumerate("xyzw"):
        replacement += f'''    code << R"(
    {{
      let x0 = tile_a[(tile % 2u) * 256u + stream * 8u + {word * 2}u];
      let x1 = tile_a[(tile % 2u) * 256u + stream * 8u + {word * 2 + 1}u];
)";
'''
        step = body
        for column in range(2):
            step = step.replace(f"packed{column}[word]", f"packed{column}.{component}")
        replacement += step + end_marker
    return source[:start] + replacement + source[body_end + len(end_marker):]


PROGRAM = make_program()


def transform(source):
    result = parent_transform(source)
    result = replace_once(result, "class GroupedPrefillRows4Program final",
                          PROGRAM + "\nclass GroupedPrefillRows4Program final")
    signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    result = replace_once(result, signature, signature + f'''
  static const int word_unroll = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(word_unroll == 0 || word_unroll == 1, "{FLAG} must be 0 or 1");
''')
    start = result.index("    if (columns8) {\n")
    end = result.index("\n    }", start) + len("\n    }")
    original = result[start:end]
    selected = original.replace("if (columns8)", "if (columns8 && word_unroll == 1)")
    selected = selected.replace("DenseColumns8DoubleBufferProgram program;", CLASS + " program;")
    result = result[:start] + selected + "\n" + original + result[end:]
    return replace_once(result, "compute, 0, columns8 ? 7 :",
                        "compute, 0, columns8 ? (word_unroll == 1 ? 8 : 7) :")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(model_eligible=False, output_columns=8, physical_threads=128,
             shared_storage_bytes=4608, activation_buffers=2,
             supported_double_buffer_selector_preserved=True,
             constant_packed_word_components=True, original_four_word_order_preserved=True,
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
