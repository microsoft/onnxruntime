class QmoeFC2FinalMixProgram final : public Program<QmoeFC2FinalMixProgram> {
 public:
  explicit QmoeFC2FinalMixProgram(bool bias) : Program{"QmoeFC2FinalMixF16"}, bias_{bias} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    if (bias_) shader.AddInput("bias", ShaderUsage::UseElementTypeAlias);
    shader.AddInput("router_values", ShaderUsage::UseElementTypeAlias);
    shader.AddInput("experts", ShaderUsage::UseElementTypeAlias);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation()
        << "var<workgroup> tile_a: array<vec4<f16>, 1600>;\n"
        << "var<workgroup> partials: array<array<f16, 32>, 20>;\n";
    auto& code = shader.MainFunctionBody();
    code << "let base_col = workgroup_idx * 2u;\n"
         << "let rank = local_idx / 32u;\n"
         << "let lane = local_idx % 32u;\n"
         << "let expert = experts[rank];\n"
         << "for (var id = local_idx; id < 1600u; id += 320u) {\n"
         << "  tile_a[id] = " << a.GetByOffset("id") << ";\n"
         << "}\n"
         << "workgroupBarrier();\n"
         << "if (lane < 20u) {\n"
         << "  for (var c = 0u; c < 2u; c++) {\n"
         << "    let index = (expert * 2560u + base_col + c) * 20u + lane;\n"
         << "    let packed = " << b.GetByOffset("index") << ";\n"
         << "    let scale = " << scales.GetByOffset("index") << ";\n"
         << "    var sum = f16(0);\n"
         << "    for (var i = 0u; i < 4u; i++) {\n"
         << "      let low = vec4<f16>(unpack4xU8(packed[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "      let high = vec4<f16>(unpack4xU8((packed[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "      let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale;\n"
         << "      let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale;\n"
         << "      sum += dot(tile_a[rank * 160u + lane * 8u + i * 2u], w0) + "
            "dot(tile_a[rank * 160u + lane * 8u + i * 2u + 1u], w1);\n"
         << "    }\n"
         << "    partials[rank * 2u + c][lane] += sum;\n"
         << "  }\n"
         << "}\n"
         << "workgroupBarrier();\n"
         << "if (local_idx < 20u) {\n"
         << "  let r = local_idx / 2u;\n"
         << "  let c = local_idx % 2u;\n"
         << "  var value = f16(0);\n"
         << "  for (var i = 0u; i < 32u; i++) { value += partials[local_idx][i]; }\n";
    if (bias_) code << "  value += bias[experts[r] * 2560u + base_col + c];\n";
    code << "  partials[local_idx][0] = value;\n"
         << "}\n"
         << "workgroupBarrier();\n"
         << "if (local_idx < 2u) {\n"
         << "  var acc = output_element_t(0);\n"
         << "  for (var r = 0u; r < uniforms.k; r++) {\n"
         << "    let expert_idx = experts[r];\n"
         << "    let router_value = router_values[expert_idx];\n"
         << "    acc += router_value * partials[r * 2u + local_idx][0];\n"
         << "  }\n"
         << output.SetByOffset("base_col + local_idx", "acc") << "\n"
         << "}\n";
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"k", ProgramUniformVariableDataType::Uint32},
      {"num_experts", ProgramUniformVariableDataType::Uint32});

 private:
  bool bias_;
};

bool QMoE::CanFuseFc2FinalMix(ComputeContext& context, bool enabled) const {
  const auto* weights = context.Input<Tensor>(5);
  const auto* scales = context.Input<Tensor>(6);
  const auto* bias = context.Input<Tensor>(7);
  const auto& limits = context.DeviceLimits();
  return enabled && CanFuseFc1Activation(context, true) && fc2_expert_weight_bits_ == 4 &&
         context.Input<Tensor>(12) == nullptr && weights != nullptr && scales != nullptr &&
         weights->IsDataType<uint8_t>() && scales->IsDataType<MLFloat16>() &&
         weights->Shape().NumDimensions() == 3 && weights->Shape().GetDims()[1] == 2560 &&
         weights->Shape().GetDims()[2] == 320 &&
         scales->Shape().NumDimensions() == 3 && scales->Shape().GetDims()[1] == 2560 &&
         scales->Shape().GetDims()[2] == 20 &&
         weights->Shape().GetDims()[0] == scales->Shape().GetDims()[0] &&
         context.Input<Tensor>(2)->Shape().GetDims()[1] == 1280 &&
         (bias == nullptr || bias->IsDataType<MLFloat16>()) &&
         limits.maxComputeWorkgroupSizeX >= 320 && limits.maxComputeInvocationsPerWorkgroup >= 320 &&
         limits.maxComputeWorkgroupStorageSize >= 14080;
}
