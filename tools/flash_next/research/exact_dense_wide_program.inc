class ExactDenseWideProgram final : public Program<ExactDenseWideProgram> {
 public:
  ExactDenseWideProgram(uint32_t blocks, uint32_t columns)
      : Program{"ExactDenseWide"}, blocks_{blocks}, columns_{columns} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseValueTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << "var<workgroup> partials: array<f16, 64>;\n";
    auto& code = shader.MainFunctionBody();
    code << "let lane = local_idx % 32u;\n"
         << "let col = workgroup_idx * 2u + local_idx / 32u;\n"
         << "var total = f16(0);\n"
         << "for (var tile = 0u; tile < " << (blocks_ + 31) / 32 << "u; tile++) {\n"
         << "  let block = tile * 32u + lane;\n"
         << "  if (col < " << columns_ << "u && block < " << blocks_ << "u) {\n"
         << "    let packed = " << b.GetByOffset("col * " + std::to_string(blocks_) + "u + block") << ";\n"
         << "    let scale = " << scales.GetByOffset("col * " + std::to_string(blocks_) + "u + block") << ";\n"
         << "    var sum = f16(0);\n"
         << "    for (var i = 0u; i < 4u; i++) {\n"
         << "      let low = vec4<f16>(unpack4xU8(packed[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "      let high = vec4<f16>(unpack4xU8((packed[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "      let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale;\n"
         << "      let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale;\n"
         << "      let x0 = " << a.GetByOffset("block * 8u + i * 2u") << ";\n"
         << "      let x1 = " << a.GetByOffset("block * 8u + i * 2u + 1u") << ";\n"
         << "      sum += dot(x0, w0) + dot(x1, w1);\n"
         << "    }\n"
         << "    total += sum;\n"
         << "  }\n"
         << "}\n"
         << "partials[local_idx] = total;\n"
         << "workgroupBarrier();\n"
         << "if (lane == 0u && col < " << columns_ << "u) {\n"
         << "  var value = f16(0);\n"
         << "  for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }\n"
         << y.SetByOffset("col", "value") << "\n"
         << "}\n";
    return Status::OK();
  }

 private:
  uint32_t blocks_;
  uint32_t columns_;
};
