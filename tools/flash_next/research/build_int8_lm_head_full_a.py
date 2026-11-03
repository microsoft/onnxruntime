import json
from pathlib import Path

from build_qmoe_node_replay import replace_once
from build_scoped_webgpu_source import build, PARENT


ROOT = Path(__file__).parent
SOURCE = PARENT / "matmul_nbits.cc"
OUTPUT = Path("/workspace/qwen38-int4-int8-lm-head-full-a/build-v1")
FLAG = "ORT_WEBGPU_INT8_LM_HEAD_FULL_A"
CLASS = "Int8LmHeadFullAProgram"
NAME = "Int8LmHeadFullAF16"
STORAGE_BYTES = 640 * 8 + 256 * 2


PROGRAM = r'''
class Int8LmHeadFullAProgram final : public Program<Int8LmHeadFullAProgram> {
 public:
  Int8LmHeadFullAProgram() : Program{"Int8LmHeadFullAF16"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"block_size", ProgramUniformVariableDataType::Uint32},
      {"blocks_per_col", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << R"(
var<workgroup> tile_a: array<vec4<f16>, 640>;
var<workgroup> partials: array<f16, 256>;
)";
    auto& code = shader.MainFunctionBody();
    code << R"(
let stream = local_idx % 32u;
let column_group = local_idx / 32u;
for (var i = local_idx; i < 640u; i += 128u) {
)" << "  tile_a[i] = " << a.GetByOffset("i") << ";\n"
         << R"(
}
workgroupBarrier();
for (var tile = 0u; tile < 5u; tile++) {
  let chunk = tile * 32u + stream;
  for (var slot = 0u; slot < 8u; slot += 4u) {
    let col = workgroup_idx * 8u + slot + column_group;
)" << "    let packed = " << b.GetByOffset("col * 160u + chunk") << ";\n"
         << "    let scale_value = " << scales.GetByOffset(
             "col * uniforms.blocks_per_col + (chunk * 16u) / uniforms.block_size") << ";\n"
         << R"(
    var sum = f16(0);
    for (var word = 0u; word < 4u; word++) {
      let weights = (vec4<f16>(unpack4xU8(packed[word])) - vec4<f16>(128)) * scale_value;
      sum += dot(tile_a[chunk * 4u + word], weights);
    }
    partials[(slot + column_group) * 32u + stream] += sum;
  }
}
workgroupBarrier();
if (local_idx < 8u) {
  var value = f16(0);
  for (var lane = 0u; lane < 32u; lane++) {
    value += partials[local_idx * 32u + lane];
  }
)" << y.SetByOffset("workgroup_idx * 8u + local_idx", "value") << "\n}\n";
    return Status::OK();
  }
};
'''


def transform(source):
    source = replace_once(source, "class GroupedPrefillRows4Program final",
                          PROGRAM + "\nclass GroupedPrefillRows4Program final")
    signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    source = replace_once(source, signature, signature + f'''
  static const int lm_head_full_a_selector = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(lm_head_full_a_selector == 0 || lm_head_full_a_selector == 1,
                    "{FLAG} must be 0 or 1");
''')
    source = replace_once(source, "  auto compute = [&]() {", f'''
  const bool lm_head_full_a = lm_head_full_a_selector == 1 &&
      context.AdapterInfo().vendor == std::string_view{{"nvidia"}} &&
      bits_ == 8 && K_ == 2560 && (N_ == 8 || N_ == 16 || N_ == 32 || N_ == 248320) &&
      block_size_ >= 16 && block_size_ % 16 == 0 && K_ % block_size_ == 0 &&
      !has_zero_points && bias == nullptr && !context.EnableMatmulFp32Accumulation() &&
      a->IsDataType<MLFloat16>() && scales->IsDataType<MLFloat16>() &&
      a->Shape().Size() == K_ && b->IsDataType<uint8_t>() && b->Shape().Size() == N_ * K_ &&
      scales->Shape().Size() == N_ * (K_ / block_size_) &&
      context.DeviceLimits().maxComputeWorkgroupStorageSize >= {STORAGE_BYTES} &&
      context.DeviceLimits().maxComputeWorkgroupSizeX >= 128 &&
      context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 128;
  auto compute = [&]() {{
    if (lm_head_full_a) {{
      {CLASS} program;
      program.AddInput({{a, ProgramTensorMetadataDependency::TypeAndRank, 4}})
          .AddInput({{b, ProgramTensorMetadataDependency::TypeAndRank, 16}})
          .AddInput({{scales, ProgramTensorMetadataDependency::TypeAndRank, 1}})
          .AddOutput({{y, ProgramTensorMetadataDependency::TypeAndRank, 1}})
          .AddUniformVariables({{onnxruntime::narrow<uint32_t>(block_size_),
                                onnxruntime::narrow<uint32_t>(K_ / block_size_)}})
          .SetWorkgroupSize(128).SetDispatchGroupSize(N_ / 8);
      return context.RunProgram(program);
    }}''')
    return replace_once(source,
        "compute, 0, dense_bitcast_route ? (block32_mode == 1 ? 3 : 1 + dense_tile4) : 0);",
        "compute, 0, lm_head_full_a ? 8 : "
        "(dense_bitcast_route ? (block32_mode == 1 ? 3 : 1 + dense_tile4) : 0));")


if __name__ == "__main__":
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(int8_only=True, lm_head_vocabulary=248320, target_k=2560,
             synthetic_vocabulary_sizes=[8, 16, 32], physical_threads=128,
             output_columns=8, original_k_streams=32, shared_storage_bytes=STORAGE_BYTES,
             activation_load_once=True, original_fp16_tile_partial_updates_preserved=True,
             model_eligible=False))
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
