class ExactExpertProgram final : public Program<ExactExpertProgram> {
 public:
  ExactExpertProgram(uint32_t blocks, uint32_t columns, uint32_t columns_per_group, bool broadcast, bool has_bias)
      : Program{"ExactExpert"}, blocks_{blocks}, columns_{columns},
        columns_per_group_{columns_per_group}, broadcast_{broadcast}, has_bias_{has_bias} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseValueTypeAlias);
    const ShaderVariableHelper* bias = nullptr;
    if (has_bias_) bias = &shader.AddInput("bias", ShaderUsage::UseValueTypeAlias);
    const auto& experts = shader.AddInput("experts", ShaderUsage::UseValueTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << "var<workgroup> partials: array<f16, "
                                      << columns_per_group_ * 32 << ">;\n";
    const auto tiles = (columns_ + columns_per_group_ - 1) / columns_per_group_;
    auto& code = shader.MainFunctionBody();
    code << "let lane = local_idx % 32u;\n"
         << "let row = workgroup_idx / " << tiles << "u;\n"
         << "let col = (workgroup_idx % " << tiles << "u) * " << columns_per_group_
         << "u + local_idx / 32u;\n"
         << "let expert = " << experts.GetByOffset("row") << ";\n"
         << "let weight_base = expert * " << columns_ * blocks_ << "u;\n"
         << "let a_base = " << (broadcast_ ? "0u" : "row * " + std::to_string(blocks_ * 8) + "u") << ";\n"
         << "var total = f16(0);\n"
         << "for (var tile = 0u; tile < " << (blocks_ + 31) / 32 << "u; tile++) {\n"
         << "  let block = tile * 32u + lane;\n"
         << "  if (col < " << columns_ << "u && block < " << blocks_ << "u) {\n"
         << "    let offset = weight_base + col * " << blocks_ << "u + block;\n"
         << "    let packed = " << b.GetByOffset("offset") << ";\n"
         << "    let scale = " << scales.GetByOffset("offset") << ";\n"
         << "    var sum = f16(0);\n"
         << "    for (var i = 0u; i < 4u; i++) {\n"
         << "      let low = vec4<f16>(unpack4xU8(packed[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "      let high = vec4<f16>(unpack4xU8((packed[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "      let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale;\n"
         << "      let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale;\n"
         << "      let x0 = " << a.GetByOffset("a_base + block * 8u + i * 2u") << ";\n"
         << "      let x1 = " << a.GetByOffset("a_base + block * 8u + i * 2u + 1u") << ";\n"
         << "      sum += dot(x0, w0) + dot(x1, w1);\n"
         << "    }\n"
         << "    total += sum;\n"
         << "  }\n"
         << "}\n"
         << "partials[local_idx] = total;\n"
         << "workgroupBarrier();\n"
         << "if (lane == 0u && col < " << columns_ << "u) {\n"
         << "  var value = f16(0);\n"
         << "  for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }\n";
    if (has_bias_) {
      code << "value += " << bias->GetByOffset("expert * " + std::to_string(columns_) + "u + col") << ";\n";
    }
    code << y.SetByOffset("row * " + std::to_string(columns_) + "u + col", "value") << "\n"
         << "}\n";
    return Status::OK();
  }

 private:
  uint32_t blocks_;
  uint32_t columns_;
  uint32_t columns_per_group_;
  bool broadcast_;
  bool has_bias_;
};
