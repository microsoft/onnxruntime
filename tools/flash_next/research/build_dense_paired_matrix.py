import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


ROOT = Path(__file__).parent
FLAG = "ORT_WEBGPU_DENSE_PAIRED_MATRIX"
CLASS = "DensePairedMatrixProgram"
SOURCE = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc")
OUTPUT = Path("/workspace/qwen38-int4-dense-paired-matrix/build-v1")
STORAGE_BYTES = 11264
PROGRAM = r'''
class DensePairedMatrixProgram final : public Program<DensePairedMatrixProgram> {
 public:
  DensePairedMatrixProgram() : Program{"DensePairedMatrix"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"blocks", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << R"(
var<workgroup> lhs: array<f16, 1024>;
var<workgroup> rhs: array<f16, 1024>;
var<workgroup> first: array<f16, 1024>;
var<workgroup> result: array<f32, 1024>;
var<workgroup> partials: array<f16, 512>;
fn decode_low_nibbles(word: u32) -> vec4<f16> {
  let bottom = bitcast<vec2<f16>>((word & 0x000f000fu) | 0x64006400u) - vec2<f16>(1032.0);
  let top = bitcast<vec2<f16>>(((word >> 8u) & 0x000f000fu) | 0x64006400u) - vec2<f16>(1032.0);
  return vec4<f16>(bottom.x, top.x, bottom.y, top.y);
}
fn correct_pair(value: f32) -> bool {
  let magnitude = abs(value);
  let remainder = bitcast<u32>(value) & 0x1fffu;
  return magnitude < 0.00006103515625 || magnitude >= 65504.0 ||
         (remainder >= 0x0fffu && remainder <= 0x1001u);
}
)";
    auto& code = shader.MainFunctionBody();
    code << R"(
let sg = local_idx / 32u;
let lane = local_idx % 32u;
let base = sg * 256u;
var totals: array<f16, 4>;
for (var tile = 0u; tile * 32u < uniforms.blocks; tile++) {
  var sums: array<f16, 4>;
  for (var word = 0u; word < 4u; word++) {
    var dots: array<f16, 4>;
    for (var half = 0u; half < 2u; half++) {
      var inputs: array<vec4<f16>, 4>;
      var weights: array<vec4<f16>, 4>;
      for (var j = 0u; j < 4u; j++) {
        let item = lane + j * 32u;
        let stream = item / 16u; let col = item % 16u;
        let block = tile * 32u + sg * 8u + stream;
        var xv = vec4<f16>(0); var wv = vec4<f16>(0);
        if (block < uniforms.blocks) {
)" << "          let packed = " << b.GetByOffset("(workgroup_idx * 16u + col) * uniforms.blocks + block") << ";\n"
         << "          let scale = " << scales.GetByOffset("(workgroup_idx * 16u + col) * uniforms.blocks + block") << ";\n"
         << "          xv = " << a.GetByOffset("block * 8u + word * 2u + half") << ";\n"
         << R"(
          let low = decode_low_nibbles(packed[word]);
          let high = decode_low_nibbles(packed[word] >> 4u);
          if (half == 0u) { wv = vec4<f16>(low.x, high.x, low.y, high.y) * scale; }
          else { wv = vec4<f16>(low.z, high.z, low.w, high.w) * scale; }
        }
        inputs[j] = xv; weights[j] = wv;
        let offset = base + stream * 32u + col;
        first[offset] = xv.x * wv.x;
        first[offset + 16u] = xv.y * wv.y;
        rhs[offset] = wv.w;
        rhs[offset + 16u] = wv.z;
      }
      for (var i = lane; i < 256u; i += 32u) {
        let row = i / 16u; let k = i % 16u;
        let block = tile * 32u + sg * 8u + row / 2u;
        var value = f16(0);
        if (row == k && block < uniforms.blocks) {
)" << "          let xv = " << a.GetByOffset("block * 8u + word * 2u + half") << ";\n"
         << R"(
          value = select(xv.w, xv.z, (row & 1u) != 0u);
        }
        lhs[base + i] = value;
      }
      workgroupBarrier();
      for (var i = lane; i < 256u; i += 32u) {
        result[base + i] = f32(first[base + i]);
      }
      workgroupBarrier();
      let left = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>, row_major>(&lhs, base, 16u);
      let right = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>, row_major>(&rhs, base, 16u);
      var accumulator = subgroupMatrixLoad<subgroup_matrix_result<f32, 16, 16>, row_major>(&result, base, 16u);
      accumulator = subgroupMatrixMultiplyAccumulate(left, right, accumulator);
      subgroupMatrixStore<row_major>(&result, base, accumulator, 16u);
      workgroupBarrier();
      for (var j = 0u; j < 4u; j++) {
        let item = lane + j * 32u;
        let stream = item / 16u; let col = item % 16u;
        let offset = base + stream * 32u + col;
        let xv = inputs[j]; let wv = weights[j];
        let raw0 = result[offset]; let raw1 = result[offset + 16u];
        var pair0 = f16(raw0); var pair1 = f16(raw1);
        if (correct_pair(raw0)) { pair0 = fma(xv.w, wv.w, first[offset]); }
        if (correct_pair(raw1)) { pair1 = fma(xv.z, wv.z, first[offset + 16u]); }
        let dot_value = pair0 + pair1;
        if (half == 0u) { dots[j] = dot_value; }
        else { sums[j] += dots[j] + dot_value; }
      }
      workgroupBarrier();
    }
  }
  for (var j = 0u; j < 4u; j++) { totals[j] += sums[j]; }
}
for (var j = 0u; j < 4u; j++) {
  let item = lane + j * 32u;
  let stream = sg * 8u + item / 16u; let col = item % 16u;
  partials[col * 32u + stream] = totals[j];
}
workgroupBarrier();
if (local_idx < 16u) {
  var value = f16(0);
  for (var stream = 0u; stream < 32u; stream++) {
    value += partials[local_idx * 32u + stream];
  }
)" << y.SetByOffset("workgroup_idx * 16u + local_idx", "value") << "\n}\n";
    return Status::OK();
  }
};
'''


def transform(source):
    result = replace_once(source, "class GroupedPrefillRows4Program final",
                          PROGRAM + "\nclass GroupedPrefillRows4Program final")
    signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    result = replace_once(result, signature, signature + f'''
  static const int paired_matrix_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(paired_matrix_mode == 0 || paired_matrix_mode == 1, "{FLAG} must be 0 or 1");
''')
    result = replace_once(result, "  auto compute = [&]() {", f'''
  std::optional<SubgroupMatrixConfig> paired_matrix_config;
  if (paired_matrix_mode == 1 && dense_bitcast_route && block32_mode == 1 &&
      N_ % 16 == 0 && b->IsDataType<uint8_t>() &&
      b->Shape().Size() == N_ * (K_ / 2) &&
      scales->Shape().Size() == N_ * (K_ / 32) &&
      context.DeviceLimits().maxComputeWorkgroupStorageSize >= {STORAGE_BYTES}) {{
    using Type = wgpu::SubgroupMatrixComponentType;
    paired_matrix_config = SelectSubgroupMatrixConfig(context, {{
        {{Type::F16, Type::F32, 16, 16, 16, 32, false}}}});
  }}
  auto compute = [&]() {{
    if (paired_matrix_config.has_value()) {{
      {CLASS} program;
      program.AddInput({{a, ProgramTensorMetadataDependency::TypeAndRank, 4}})
          .AddInput({{b, ProgramTensorMetadataDependency::TypeAndRank, 16}})
          .AddInput({{scales, ProgramTensorMetadataDependency::TypeAndRank, 1}})
          .AddOutput({{y, ProgramTensorMetadataDependency::TypeAndRank, 1}})
          .AddUniformVariables({{onnxruntime::narrow<uint32_t>(K_ / 32)}})
          .SetWorkgroupSize(128).SetDispatchGroupSize(N_ / 16);
      return context.RunProgram(program);
    }}''')
    return replace_once(result,
        "compute, 0, dense_bitcast_route ? (block32_mode == 1 ? 3 : 1 + dense_tile4) : 0);",
        "compute, 0, paired_matrix_config.has_value() ? 4 : "
        "(dense_bitcast_route ? (block32_mode == 1 ? 3 : 1 + dense_tile4) : 0));")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(diagnostic_only=False, model_eligible=False, shared_storage_bytes=STORAGE_BYTES,
             original_k_streams=32, streams_per_subgroup=8, output_columns=16,
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
