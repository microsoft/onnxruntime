class SmallMPrefillLayoutProgram final : public Program<SmallMPrefillLayoutProgram> {
 public:
  SmallMPrefillLayoutProgram() : Program{"SmallMPrefillLayout"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& input = shader.AddInput("input", ShaderUsage::UseValueTypeAlias);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias);
    shader.MainFunctionBody()
        << "if (global_idx >= uniforms.size) { return; }\n"
        << "let row = (global_idx % 256u) / 16u;\n"
        << "let col = (global_idx / 256u) * 16u + global_idx % 16u;\n"
        << output.SetByOffset("global_idx", input.GetByOffset("row * uniforms.K + col")) << "\n";
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"K", ProgramUniformVariableDataType::Uint32},
      {"size", ProgramUniformVariableDataType::Uint32});
};

class SmallMPrefillProgram final : public Program<SmallMPrefillProgram> {
 public:
  explicit SmallMPrefillProgram(bool fp32)
      : Program{fp32 ? "SmallMPrefillMatMulNBitsF32" : "SmallMPrefillMatMulNBitsF16"}, fp32_{fp32} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddInput("input_a", ShaderUsage::UseElementTypeAlias);
    const auto& b = shader.AddInput("input_b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << "alias accum_t = " << (fp32_ ? "f32" : "f16") << ";\n"
        << "var<workgroup> buf_b: array<f16, 5120>;\n"
        << "var<workgroup> stage: array<accum_t, 1024>;\n"
        << "fn dequant_b(n_base: u32, k_block: u32, tid: u32) {\n"
        << "  let k_offset = (tid % 2u) * 16u;\n"
        << "  for (var half = 0u; half < 2u; half++) {\n"
        << "    let local_n = tid / 2u + half * 64u;\n"
        << "    let n = n_base + local_n;\n"
        << "    let buf_base = local_n * 40u + k_offset;\n"
        << "    if (n < uniforms.N) {\n"
        << "      let packed_idx = (n * uniforms.K + k_block + k_offset) / 8u;\n"
        << "      let scale = f16(" << scales.GetByOffset("n * (uniforms.K / 32u) + k_block / 32u") << ");\n"
        << "      for (var step = 0u; step < 2u; step++) {\n"
        << "        let packed = " << b.GetByOffset("packed_idx + step") << ";\n"
        << "        let low = (vec4<f16>(unpack4xU8(packed & 0x0f0f0f0fu)) - vec4<f16>(8)) * scale;\n"
        << "        let high = (vec4<f16>(unpack4xU8((packed >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8)) * scale;\n"
        << "        let base = buf_base + step * 8u;\n"
        << "        buf_b[base] = low.x; buf_b[base + 1u] = high.x;\n"
        << "        buf_b[base + 2u] = low.y; buf_b[base + 3u] = high.y;\n"
        << "        buf_b[base + 4u] = low.z; buf_b[base + 5u] = high.z;\n"
        << "        buf_b[base + 6u] = low.w; buf_b[base + 7u] = high.w;\n"
        << "      }\n"
        << "    } else {\n"
        << "      for (var i = 0u; i < 16u; i++) { buf_b[buf_base + i] = f16(0); }\n"
        << "    }\n"
        << "  }\n"
        << "}\n";
    auto& code = shader.MainFunctionBody();
    code << R"WGSL(
let n_base = workgroup_idx * 128u;
let subgroup = local_idx / 32u;
let lane = local_idx % 32u;
var c0: subgroup_matrix_result<accum_t, 16, 16>;
var c1: subgroup_matrix_result<accum_t, 16, 16>;
for (var k_block = 0u; k_block < uniforms.K; k_block += 32u) {
  dequant_b(n_base, k_block, local_idx);
  workgroupBarrier();
  for (var k = 0u; k < 32u; k += 16u) {
    let a = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>, row_major>(
        &input_a, (k_block + k) * 16u, 16u);
    let b_base = subgroup * 32u * 40u + k;
    let b0 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>, col_major>(
        &buf_b, b_base, 40u);
    let b1 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>, col_major>(
        &buf_b, b_base + 16u * 40u, 40u);
    c0 = subgroupMatrixMultiplyAccumulate(a, b0, c0);
    c1 = subgroupMatrixMultiplyAccumulate(a, b1, c1);
  }
  workgroupBarrier();
}
let stage_base = subgroup * 256u;
)WGSL";
    for (uint32_t tile = 0; tile < 2; ++tile) {
      code << "subgroupMatrixStore<row_major>(&stage, stage_base, c" << tile << ", 16u);\n"
           << "workgroupBarrier();\n"
           << "for (var i = lane; i < 256u; i += 32u) {\n"
           << "  let row = i / 16u;\n"
           << "  let col = n_base + subgroup * 32u + " << tile * 16u << "u + i % 16u;\n"
           << "  if (col < uniforms.N) {\n"
           << output.SetByOffset("row * uniforms.N + col", "f16(stage[stage_base + i])") << "\n"
           << "  }\n"
           << "}\n"
           << "workgroupBarrier();\n";
    }
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32});

 private:
  bool fp32_;
};

Status ApplySmallMPrefill(const Tensor* a, const Tensor* b, const Tensor* scales,
                         uint32_t N, uint32_t K, const SubgroupMatrixConfig& config,
                         onnxruntime::webgpu::ComputeContext& context, Tensor* y) {
  const bool fp32 = config.resultComponentType == wgpu::SubgroupMatrixComponentType::F32;
  auto prepacked = context.CreateGPUTensor(a->DataType(), TensorShape{16, K});
  SmallMPrefillLayoutProgram layout;
  layout.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 1})
      .AddOutput({&prepacked, ProgramTensorMetadataDependency::TypeAndRank, 1})
      .SetWorkgroupSize(128).SetDispatchGroupSize(CeilDiv(16u * K, 128u))
      .AddUniformVariables({K, 16u * K});
  ORT_RETURN_IF_ERROR(context.RunProgram(layout));
  SmallMPrefillProgram program{fp32};
  program.AddInput({&prepacked, ProgramTensorMetadataDependency::TypeAndRank, TensorShape{16, K}, 1})
      .AddInput({b, ProgramTensorMetadataDependency::TypeAndRank, 4})
      .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank, 1})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, TensorShape{1, 16, N}, 1})
      .SetWorkgroupSize(128).SetSubgroupSize(32)
      .SetDispatchGroupSize(CeilDiv(N, 128u))
      .AddUniformVariables({N, K})
      .CacheHint(fp32);
  return context.RunProgram(program);
}
