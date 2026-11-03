import json
from pathlib import Path
import subprocess

from build_dense_paired_matrix import PROGRAM as ORIGINAL_PROGRAM, transform as original_transform, SOURCE, ROOT
from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_DENSE_PAIRED_MATRIX_RESET"
CLASS = "DensePairedMatrixResetProgram"
OUTPUT = Path("/workspace/qwen38-int4-dense-paired-matrix-reset/build-v1")
PROGRAM = ORIGINAL_PROGRAM.replace("DensePairedMatrix", "DensePairedMatrixReset")
PROGRAM = replace_once(PROGRAM, "var totals: array<f16, 4>;",
    "var totals: array<f16, 4>;\nfor (var j = 0u; j < 4u; j++) { totals[j] = f16(0); }")
PROGRAM = replace_once(PROGRAM, "  var sums: array<f16, 4>;",
    "  var sums: array<f16, 4>;\n  for (var j = 0u; j < 4u; j++) { sums[j] = f16(0); }")


def transform(source):
    result = replace_once(original_transform(source), ORIGINAL_PROGRAM, PROGRAM)
    result = result.replace("ORT_WEBGPU_DENSE_PAIRED_MATRIX", FLAG)
    return replace_once(result, "DensePairedMatrixProgram program;", CLASS + " program;")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(diagnostic_only=False, model_eligible=False, shared_storage_bytes=11264,
             original_k_streams=32, streams_per_subgroup=8, output_columns=16,
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
