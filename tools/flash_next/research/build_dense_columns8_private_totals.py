import json
from pathlib import Path
import subprocess

from build_dense_columns8_double_buffer import PROGRAM as ORIGINAL_PROGRAM
from build_dense_columns8_word_unroll import PROGRAM as REPLACED_PROGRAM, transform as selector_transform, SOURCE, ROOT
from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_DENSE_COLUMNS8_PRIVATE_TOTALS"
CLASS = "DenseColumns8PrivateTotalsProgram"
OUTPUT = Path("/workspace/qwen38-int4-dense-columns8-private-totals/build-v1")


def make_program():
    source = ORIGINAL_PROGRAM.replace("DenseColumns8DoubleBuffer", "DenseColumns8PrivateTotals")
    source = replace_once(source,
        "for (var i = local_idx; i < 256u; i += 128u) { partials[i] = f16(0); }",
        "var totals = vec2<f16>(0);")
    source = replace_once(source,
        '    code << "    partials[(column_group + 0u) * 32u + stream] += sum0;\\n";\n'
        '    code << "    partials[(column_group + 4u) * 32u + stream] += sum1;\\n";',
        '    code << "    totals += vec2<f16>(sum0, sum1);\\n";')
    return replace_once(source, """  }
}
workgroupBarrier();
if (local_idx < 8u)""", """  }
}
partials[column_group * 32u + stream] = totals.x;
partials[(column_group + 4u) * 32u + stream] = totals.y;
workgroupBarrier();
if (local_idx < 8u)""")


PROGRAM = make_program()


def transform(source):
    result = replace_once(selector_transform(source), REPLACED_PROGRAM, PROGRAM)
    result = result.replace("ORT_WEBGPU_DENSE_COLUMNS8_WORD_UNROLL", FLAG)
    result = result.replace("word_unroll", "private_totals")
    return replace_once(result, "DenseColumns8WordUnrollProgram program;", CLASS + " program;")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(model_eligible=False, output_columns=8, physical_threads=128,
             shared_storage_bytes=4608, activation_buffers=2,
             supported_double_buffer_selector_preserved=True,
             private_fp16_vector_totals=2, explicit_zero_initialization=True,
             final_unique_shared_partial_store=True,
             original_dot_word_and_tile_order_preserved=True,
             held_word_unroll_and_swizzle_not_stacked=True,
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
