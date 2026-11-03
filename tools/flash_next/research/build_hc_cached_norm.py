from pathlib import Path
import textwrap

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_HC_CACHED_NORM"
PROGRAM = "HCPostNormProjectionCachedNormF16"
OUTPUT = Path("/workspace/qwen38-int4-hc-cached-norm/build-v2")
PARENT = Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1")
STORAGE_BYTES = 11520 * 2 + 256 * 2


def projection_program(source):
    start = source.index("class HCPostNormProjectionProgram final")
    stop = source.index("class HCPostNormProjectionF16", start)
    original = source[start:stop]
    start = original.index('         << "let lane = local_idx % 32u;')
    stop = original.index("    return Status::OK();", start)
    projection = "    code\n" + original[start:stop]
    return f'''class HCPostNormProjectionCachedNormProgram final : public Program<HCPostNormProjectionCachedNormProgram> {{
 public:
  HCPostNormProjectionCachedNormProgram() : Program{{"{PROGRAM}"}} {{}}
  Status GenerateShaderCode(ShaderHelper& shader) const override {{
    const auto& normalized = shader.AddInput("normalized");
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("projection_scales", ShaderUsage::UseValueTypeAlias);
    const auto& activated = shader.AddOutput("activated", ShaderUsage::UseElementTypeAlias);
    const auto& tail = shader.AddOutput("tail");
    shader.AdditionalImplementation() <<
        "var<workgroup> post_values: array<f16, 11520>;\\n"
        "var<workgroup> partials: array<f16, 256>;\\n"
        "fn stable_sigmoid(v: f32) -> f32 {{\\n"
        "  if (v >= 0.0) {{ return 1.0 / (1.0 + exp(-v)); }}\\n"
        "  let e = exp(v); return e / (1.0 + e);\\n}}\\n";
    auto& code = shader.MainFunctionBody();
    code << "for (var i = local_idx; i < 10240u; i += 256u) {{\\n"
         << "  let address = (i / 32u) * 36u + i % 32u;\\n"
         << "  post_values[address] = " << normalized.GetByOffset("i") << ";\\n"
         << "}}\\nworkgroupBarrier();\\n";
{projection}    return Status::OK();
  }}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {{"prefix", ProgramUniformVariableDataType::Uint32}},
      {{"alpha", ProgramUniformVariableDataType::Float32}});
}};

'''


def transform(source):
    private = projection_program(source)
    norm_start = source.index("    HCPostNormProjectionBaselineNormProgram norm")
    norm_stop = source.index("    if (projection_fused)", norm_start)
    norm = textwrap.indent(source[norm_start:norm_stop], "  ")
    source = replace_once(source, "class HCPostNormProjectionF16", private + "class HCPostNormProjectionF16")
    source = replace_once(source,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    const int cached_norm = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(cached_norm == 0 || cached_norm == 1, "{FLAG} must be 0 or 1");''')
    return replace_once(source, "    if (fused) {",
        f'''    if (fused && cached_norm == 1) {{
{norm}      HCPostNormProjectionCachedNormProgram program;
      program.AddInput({{normalized, ProgramTensorMetadataDependency::Type}})
          .AddInput({{weights, ProgramTensorMetadataDependency::TypeAndRank, 16}})
          .AddInput({{scales, ProgramTensorMetadataDependency::TypeAndRank}})
          .AddOutput({{activated, ProgramTensorMetadataDependency::Type}})
          .AddOutput({{tail, ProgramTensorMetadataDependency::Type}})
          .AddUniformVariables({{{{prefix}}, {{alpha_}}}})
          .SetWorkgroupSize(256).SetDispatchGroupSize(41);
      return context.RunProgram(program);
    }}
    if (fused) {{''')


if __name__ == "__main__":
    build(output=OUTPUT, flag=FLAG, source_transform=transform, parent_dir=PARENT,
          replay_variant_keyed=False)
