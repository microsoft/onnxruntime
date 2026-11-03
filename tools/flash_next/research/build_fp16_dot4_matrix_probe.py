import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


ROOT = Path(__file__).parent
FLAG = "ORT_WEBGPU_FP16_DOT4_MATRIX_PROBE"
SOURCE = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")
OUTPUT = Path("/workspace/qwen38-int4-fp16-dot4-matrix-probe/build-v1")
PROGRAM = r'''
class Fp16Dot4MatrixProbeProgram final : public Program<Fp16Dot4MatrixProbeProgram> {
 public:
  explicit Fp16Dot4MatrixProbeProgram(uint32_t rows) : Program{"Fp16Dot4MatrixProbe"}, rows_{rows} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseElementTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scale = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << "const kRows: u32 = " << rows_ << "u;\n" << R"(
var<workgroup> lhs: array<f16, 256>;
var<workgroup> rhs: array<f16, 256>;
var<workgroup> result: array<f16, 256>;
)";
    auto& code = shader.MainFunctionBody();
    code << "let row_base = workgroup_idx * kRows;\n"
         << "for (var i = local_idx; i < kRows * 16u; i += 32u) {\n"
         << " let row = i / 16u; let k = i % 16u;\n"
         << " if (k < 4u) { lhs[i] = " << a.GetByOffset("(row_base + row) * 4u + k") << "; }\n"
         << " else { lhs[i] = f16(0); }\n}\n"
         << "for (var i = local_idx; i < 256u; i += 32u) {\n"
         << " let k = i / 16u; let col = i % 16u;\n"
         << " if (k < 4u) {\n"
         << "  let packed = " << b.GetByOffset("col") << ";\n"
         << "  let coefficient = f16(i32((packed.x >> (k * 4u)) & 15u) - 8);\n"
         << "  rhs[i] = coefficient * " << scale.GetByOffset("col") << ";\n"
         << " } else { rhs[i] = f16(0); }\n}\nworkgroupBarrier();\n"
         << "let left = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, kRows>, row_major>(&lhs, 0u, 16u);\n"
         << "let right = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>, row_major>(&rhs, 0u, 16u);\n"
         << "var accumulator: subgroup_matrix_result<f16, 16, kRows>;\n"
         << "accumulator = subgroupMatrixMultiplyAccumulate(left, right, accumulator);\n"
         << "subgroupMatrixStore<row_major>(&result, 0u, accumulator, 16u);\nworkgroupBarrier();\n"
         << "for (var i = local_idx; i < kRows * 16u; i += 32u) {\n"
         << " let row = i / 16u; let col = i % 16u;\n"
         << " let xv = vec4<f16>(" << a.GetByOffset("(row_base + row) * 4u")
         << ", " << a.GetByOffset("(row_base + row) * 4u + 1u")
         << ", " << a.GetByOffset("(row_base + row) * 4u + 2u")
         << ", " << a.GetByOffset("(row_base + row) * 4u + 3u") << ");\n"
         << " let wv = vec4<f16>(rhs[col], rhs[16u + col], rhs[32u + col], rhs[48u + col]);\n"
         << " " << y.SetByOffset("(row_base + row) * 32u + col", "result[i]") << "\n"
         << " " << y.SetByOffset("(row_base + row) * 32u + 16u + col", "dot(xv, wv)") << "\n}\n";
    return Status::OK();
  }
 private:
  uint32_t rows_;
};
'''


def transform(source):
    source = replace_once(source, "class GroupedPrefillRows4Program final",
                          PROGRAM + "\nclass GroupedPrefillRows4Program final")
    start = source.index("Status ApplyMatMulNBits(")
    position = source.index("  const int grouped_prefill_rows", start)
    branch = f'''
  const int dot4_probe = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(dot4_probe == 0 || dot4_probe == 1, "{FLAG} must be 0 or 1");
  if (dot4_probe == 1) {{
    ORT_RETURN_IF_NOT(batch_count == 1 && M == 16 && dispatch_M == M && N == 32 && K == 4 &&
        block_size == 32 && nbits == 4 && !has_bias && !has_zero_points && !has_weight_idx &&
        a->IsDataType<MLFloat16>() && scales->IsDataType<MLFloat16>(),
        "FP16 dot4 matrix probe requires the diagnostic M16 K4 N32 fixture");
    using Type = wgpu::SubgroupMatrixComponentType;
    auto config = SelectSubgroupMatrixConfig(context, {{
        {{Type::F16, Type::F16, 16, 16, 16, 32, false}},
        {{Type::F16, Type::F16, 8, 16, 16, 32, false}}}});
    ORT_RETURN_IF_NOT(config.has_value(), "FP16 dot4 matrix probe has no supported F16-result matrix config");
    Fp16Dot4MatrixProbeProgram program{{config->M}};
    program.AddInput({{a, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddInput({{b, ProgramTensorMetadataDependency::TypeAndRank, 16}})
        .AddInput({{scales, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddOutput({{y, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .SetWorkgroupSize(32).SetDispatchGroupSize(16 / config->M);
    return context.RunProgram(program);
  }}

'''
    return source[:position] + branch + source[position:]


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
                   dict(diagnostic_only=True, model_eligible=False, matrix_terms=4, padded_matrix_K=16))
    runner = OUTPUT / "dot4_compare.cpp"
    runner.write_text((ROOT / "fp16_dot4_matrix_compare.cpp").read_text())
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
