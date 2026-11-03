class MatMulNBitsDirectAProgram final : public Program<MatMulNBitsDirectAProgram> {
 public:
  MatMulNBitsDirectAProgram(bool bias, bool indirect, bool broadcast)
      : Program{"MatMulNBitsDirectA"}, bias_{bias}, indirect_{indirect}, broadcast_{broadcast} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    if (bias_) shader.AddInput("bias", ShaderUsage::UseElementTypeAlias);
    if (indirect_) shader.AddInput("experts", ShaderUsage::UseValueTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation()
        << "var<workgroup> partials: array<array<f16, 32>, 8>;\n";
    auto& code = shader.MainFunctionBody();
    code << "let row = workgroup_idx / uniforms.tiles;\n"
         << "let base_col = (workgroup_idx % uniforms.tiles) * 8u;\n"
         << "let lane = local_idx % 32u;\n"
         << "let group = local_idx / 32u;\n"
         << (broadcast_ ? "let input_row = 0u;\n" : "let input_row = row;\n")
         << (indirect_ ? "let expert = experts[row];\n" : "let expert = uniforms.weight_index;\n")
         << "let blocks = uniforms.K / 32u;\n"
         << "for (var tile = 0u; tile < blocks; tile += 32u) {\n"
         << "  let block = tile + lane;\n"
         << "  for (var offset = 0u; offset < 8u; offset += 4u) {\n"
         << "    let col = base_col + offset + group;\n"
         << "    if (col < uniforms.N && block < blocks) {\n"
         << "      let packed = " << b.GetByOffset("(expert * uniforms.N + col) * blocks + block") << ";\n"
         << "      let scale = " << scales.GetByOffset("(expert * uniforms.N + col) * blocks + block") << ";\n"
         << "      var sum = f16(0);\n"
         << "      for (var i = 0u; i < 4u; i++) {\n"
         << "        let low = vec4<f16>(unpack4xU8(packed[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "        let high = vec4<f16>(unpack4xU8((packed[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "        let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale;\n"
         << "        let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale;\n"
         << "        let x0 = " << a.GetByOffset("input_row * (uniforms.K / 4u) + block * 8u + i * 2u") << ";\n"
         << "        let x1 = " << a.GetByOffset("input_row * (uniforms.K / 4u) + block * 8u + i * 2u + 1u") << ";\n"
         << "        sum += dot(x0, w0) + dot(x1, w1);\n"
         << "      }\n"
         << "      partials[offset + group][lane] += sum;\n"
         << "    }\n"
         << "  }\n"
         << "}\n"
         << "workgroupBarrier();\n"
         << "if (local_idx < 8u) {\n"
         << "  let col = base_col + local_idx;\n"
         << "  if (col < uniforms.N) {\n"
         << "    var value = f16(0);\n"
         << "    for (var i = 0u; i < 32u; i++) { value += partials[local_idx][i]; }\n";
    if (bias_) code << "    value += bias[expert * uniforms.N + col];\n";
    code << y.SetByOffset("row * uniforms.N + col", "value") << "\n"
         << "  }\n}\n";
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"tiles", ProgramUniformVariableDataType::Uint32},
      {"weight_index", ProgramUniformVariableDataType::Uint32});

 private:
  bool bias_;
  bool indirect_;
  bool broadcast_;
};
