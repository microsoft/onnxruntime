import json
from pathlib import Path
import subprocess

from build_fp16_dot4_matrix_f32_probe import FLAG as OLD_FLAG, PROGRAM as OLD_PROGRAM, transform as old_transform
from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


ROOT = Path(__file__).parent
FLAG = "ORT_WEBGPU_FP16_DOT4_PAIR_MATRIX_PROBE"
SOURCE = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")
OUTPUT = Path("/workspace/qwen38-int4-fp16-dot4-pair-matrix-probe/build-v1")
PROGRAM = r'''
class Fp16Dot4PairMatrixProbeProgram final : public Program<Fp16Dot4PairMatrixProbeProgram> {
 public:
  explicit Fp16Dot4PairMatrixProbeProgram(uint32_t rows) : Program{"Fp16Dot4PairMatrixProbe"}, rows_{rows} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseElementTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scale = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << "const kRows: u32 = " << rows_ << "u;\n" << R"(
var<workgroup> weights: array<vec4<f16>, 16>;
var<workgroup> lhs: array<f16, 256>;
var<workgroup> rhs: array<f16, 256>;
var<workgroup> result: array<f32, 256>;
fn small_pair(value: f32) -> bool {
  return abs(value) < 0.00006103515625;
}
fn boundary_pair(value: f32) -> bool {
  let remainder = bitcast<u32>(value) & 0x1fffu;
  return (remainder >= 0x0fffu && remainder <= 0x1001u) || abs(value) >= 65504.0;
}
)";
    auto& code = shader.MainFunctionBody();
    code << "let row_base = workgroup_idx;\n"
         << "let xv = vec4<f16>(" << a.GetByOffset("row_base * 4u")
         << ", " << a.GetByOffset("row_base * 4u + 1u")
         << ", " << a.GetByOffset("row_base * 4u + 2u")
         << ", " << a.GetByOffset("row_base * 4u + 3u") << ");\n"
         << "if (local_idx < 16u) {\n"
         << " let packed = " << b.GetByOffset("local_idx") << ";\n"
         << " let codes = (vec4<u32>(packed.x) >> vec4<u32>(0u, 4u, 8u, 12u)) & vec4<u32>(15u);\n"
         << " weights[local_idx] = vec4<f16>(vec4<i32>(codes) - vec4<i32>(8)) * "
         << scale.GetByOffset("local_idx") << ";\n"
         << "}\nworkgroupBarrier();\n"
         << R"(
for (var i = local_idx; i < kRows * 16u; i += 32u) {
  let row = i / 16u; let k = i % 16u;
  var value = f16(0);
  if (row == 0u && k == 0u) { value = f16(1); }
  if (row == 0u && k == 1u) { value = xv.w; }
  if (row == 1u && k == 2u) { value = f16(1); }
  if (row == 1u && k == 3u) { value = xv.z; }
  lhs[i] = value;
}
for (var i = local_idx; i < 256u; i += 32u) {
  let k = i / 16u; let col = i % 16u; let wv = weights[col];
  var value = f16(0);
  if (k == 0u) { value = xv.x * wv.x; }
  if (k == 1u) { value = wv.w; }
  if (k == 2u) { value = xv.y * wv.y; }
  if (k == 3u) { value = wv.z; }
  rhs[i] = value;
}
workgroupBarrier();
let left = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, kRows>, row_major>(&lhs, 0u, 16u);
let right = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>, row_major>(&rhs, 0u, 16u);
var accumulator: subgroup_matrix_result<f32, 16, kRows>;
accumulator = subgroupMatrixMultiplyAccumulate(left, right, accumulator);
subgroupMatrixStore<row_major>(&result, 0u, accumulator, 16u);
workgroupBarrier();
if (local_idx < 16u) {
  let col = local_idx; let wv = weights[col];
  let raw0 = result[col]; let raw1 = result[16u + col];
  let zero0 = small_pair(raw0); let zero1 = small_pair(raw1);
  let bound0 = boundary_pair(raw0); let bound1 = boundary_pair(raw1);
  let correct0 = zero0 || bound0; let correct1 = zero1 || bound1;
  let native0 = f16(raw0); let native1 = f16(raw1);
  var pair0 = native0; var pair1 = native1;
  if (correct0) { pair0 = fma(xv.w, wv.w, rhs[col]); }
  if (correct1) { pair1 = fma(xv.z, wv.z, rhs[32u + col]); }
  let mask = select(0u, 1u, correct0) | select(0u, 2u, correct1) |
             select(0u, 4u, zero0) | select(0u, 8u, zero1) |
             select(0u, 16u, bound0) | select(0u, 32u, bound1);
)"
         << " " << y.SetByOffset("row_base * 64u + col", "native0 + native1") << "\n"
         << " " << y.SetByOffset("row_base * 64u + 16u + col", "pair0 + pair1") << "\n"
         << " " << y.SetByOffset("row_base * 64u + 32u + col", "dot(xv, wv)") << "\n"
         << " " << y.SetByOffset("row_base * 64u + 48u + col", "f16(mask)") << "\n}\n";
    return Status::OK();
  }
 private:
  uint32_t rows_;
};
'''


def transform(source):
    result = old_transform(source)
    result = replace_once(result, OLD_PROGRAM, PROGRAM)
    result = result.replace(OLD_FLAG, FLAG).replace(
        "Fp16Dot4MatrixF32ProbeProgram program", "Fp16Dot4PairMatrixProbeProgram program")
    result = replace_once(result, "N == 32 && K == 4", "N == 64 && K == 4")
    result = result.replace("M16 K4 N32 fixture", "M16 K4 N64 fixture")
    return replace_once(result, ".SetDispatchGroupSize(16 / config->M)", ".SetDispatchGroupSize(16)")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
                   dict(diagnostic_only=True, model_eligible=False, matrix_result_type="f32",
                        pair_tree="fma3_rounded0_plus_fma2_rounded1",
                        scalar_zero_subnormal_boundary_correction=True))
    runner = OUTPUT / "pair_matrix_compare.cpp"
    runner.write_text((ROOT / "fp16_dot4_pair_matrix_compare.cpp").read_text())
    subprocess.run(["c++", "-O3", "-std=c++20", str(runner),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(OUTPUT / "pair_matrix_compare")], check=True)
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True,
                  diagnostic_runner_sha256=sha(OUTPUT / "pair_matrix_compare"))
    report["source_sha256"]["pair_matrix_compare.cpp"] = sha(runner)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
