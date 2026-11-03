import json
from pathlib import Path
import subprocess

from build_fp16_dot4_matrix_probe import FLAG as ORIGINAL_FLAG, PROGRAM as ORIGINAL_PROGRAM, transform as original_transform
from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


ROOT = Path(__file__).parent
FLAG = "ORT_WEBGPU_FP16_DOT4_MATRIX_F32_PROBE"
SOURCE = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")
OUTPUT = Path("/workspace/qwen38-int4-fp16-dot4-matrix-f32-probe/build-v1")


def program_transform(source):
    return (source.replace("Fp16Dot4MatrixProbe", "Fp16Dot4MatrixF32Probe")
            .replace("result: array<f16, 256>", "result: array<f32, 256>")
            .replace("subgroup_matrix_result<f16, 16, kRows>", "subgroup_matrix_result<f32, 16, kRows>")
            .replace('"result[i]"', '"f16(result[i])"'))


PROGRAM = program_transform(ORIGINAL_PROGRAM)


def transform(source):
    result = original_transform(source)
    result = replace_once(result, ORIGINAL_PROGRAM, PROGRAM)
    result = result.replace(ORIGINAL_FLAG, FLAG).replace(
        "Fp16Dot4MatrixProbeProgram program", "Fp16Dot4MatrixF32ProbeProgram program")
    result = result.replace("Type::F16, Type::F16", "Type::F16, Type::F32")
    return result.replace("supported F16-result matrix config", "supported F32-result matrix config")


def comparator_transform(source):
    source = replace_once(source, "size_t comparisons = 0, mismatches = 0;",
                          "size_t comparisons = 0, mismatches = 0, zero_sign_mismatches = 0, nonzero_mismatches = 0;")
    source = replace_once(source, "              ++mismatches;",
        """              if (matrix.ToFloat() == 0.0f && scalar.ToFloat() == 0.0f)
                ++zero_sign_mismatches;
              else
                ++nonzero_mismatches;
              ++mismatches;""")
    return replace_once(source, "    if (mismatches != 0) return 2;",
        '''    std::cout << "DOT4 classifications zero_sign_mismatches=" << zero_sign_mismatches
              << " nonzero_mismatches=" << nonzero_mismatches << std::endl;
    if (mismatches != 0) return 2;''')


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
                   dict(diagnostic_only=True, model_eligible=False, matrix_terms=4,
                        padded_matrix_K=16, matrix_result_type="f32",
                        single_native_matrix_accumulation=True))
    runner = OUTPUT / "dot4_compare.cpp"
    runner.write_text(comparator_transform((ROOT / "fp16_dot4_matrix_compare.cpp").read_text()))
    subprocess.run(["c++", "-O3", "-std=c++20", str(runner),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "dot4_compare")], check=True)
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True,
                  diagnostic_runner_sha256=sha(OUTPUT / "dot4_compare"))
    report["source_sha256"]["dot4_compare.cpp"] = sha(runner)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
