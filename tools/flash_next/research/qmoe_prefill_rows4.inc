class GroupedPrefillRows4Program final : public Program<GroupedPrefillRows4Program> {
 public:
  explicit GroupedPrefillRows4Program(bool bias)
      : Program{"MatMulNBitsGroupedRows4F16"}, bias_{bias} {}

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"num_N_tile", ProgramUniformVariableDataType::Uint32});

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseElementTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    if (bias_) shader.AddInput("bias", ShaderUsage::UseElementTypeAlias);
    const auto& experts = shader.AddInput("experts", ShaderUsage::UseValueTypeAlias);
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << R"(
var<workgroup> tile_a: array<vec4<f16>, 1024>;
var<workgroup> partials: array<array<array<f16, 32>, 8>, 4>;
var<workgroup> row_experts: array<u32, 4>;
)";
    auto& code = shader.MainFunctionBody();
    code << "let first_row = (workgroup_idx / uniforms.num_N_tile) * 4u;\n"
         << "let first_col = (workgroup_idx % uniforms.num_N_tile) * 8u;\n"
         << "let idx = local_idx % 32u;\n"
         << "let idy = local_idx / 32u;\n"
         << "if (local_idx < 4u) { row_experts[local_idx] = "
         << experts.GetByOffset("first_row + local_idx") << "; }\n"
         << "for (var kidx = 0u; kidx < uniforms.K; kidx += 1024u) {\n"
         << "  for (var i = local_idx; i < 1024u; i += 128u) {\n"
         << "    let row = i / 256u;\n"
         << "    let k = kidx / 4u + i % 256u;\n"
         << "    if (k < uniforms.K / 4u) { tile_a[i] = "
         << a.GetByOffset("(first_row + row) * (uniforms.K / 4u) + k") << "; }\n"
         << "    else { tile_a[i] = vec4<f16>(0); }\n"
         << "  }\n"
         << "  workgroupBarrier();\n"
         << "  for (var slot = 0u; slot < 8u; slot += 4u) {\n"
         << "    let col = first_col + slot + idy;\n"
         << "    let block = kidx / 32u + idx;\n"
         << "    if (block < uniforms.K / 32u) {\n"
         << "      var cached_expert = 0xffffffffu;\n"
         << "      var weights: array<vec4<f16>, 8>;\n"
         << "      for (var row = 0u; row < 4u; row++) {\n"
         << "        let expert = row_experts[row];\n"
         << "        if (expert != cached_expert) {\n"
         << "          let scale = "
         << scales.GetByOffset("(expert * uniforms.N + col) * (uniforms.K / 32u) + block")
         << ";\n"
         << "          let packed = "
         << b.GetByOffset("(expert * uniforms.N + col) * (uniforms.K / 32u) + block") << ";\n"
         << "          for (var word = 0u; word < 4u; word++) {\n"
         << "            let lo = vec4<f16>(unpack4xU8(packed[word] & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "            let hi = vec4<f16>(unpack4xU8((packed[word] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "            weights[word * 2u] = vec4<f16>(lo.x, hi.x, lo.y, hi.y) * scale;\n"
         << "            weights[word * 2u + 1u] = vec4<f16>(lo.z, hi.z, lo.w, hi.w) * scale;\n"
         << "          }\n"
         << "          cached_expert = expert;\n"
         << "        }\n"
         << "        var sum = f16(0);\n"
         << "        for (var word = 0u; word < 4u; word++) {\n"
         << "          let offset = row * 256u + idx * 8u + word * 2u;\n"
         << "          sum += dot(tile_a[offset], weights[word * 2u]) +\n"
         << "                 dot(tile_a[offset + 1u], weights[word * 2u + 1u]);\n"
         << "        }\n"
         << "        partials[row][slot + idy][idx] += sum;\n"
         << "      }\n"
         << "    }\n"
         << "  }\n"
         << "  workgroupBarrier();\n"
         << "}\n"
         << "if (local_idx < 8u) {\n"
         << "  let col = first_col + local_idx;\n"
         << "  for (var row = 0u; row < 4u; row++) {\n"
         << "    var value = f16(0);\n"
         << "    for (var lane = 0u; lane < 32u; lane++) {\n"
         << "      value += partials[row][local_idx][lane];\n"
         << "    }\n";
    if (bias_) code << "    value += bias[row_experts[row] * uniforms.N + col];\n";
    code << "    " << y.SetByOffset("(first_row + row) * uniforms.N + col", "value") << "\n"
         << "  }\n"
         << "}\n";
    return Status::OK();
  }

 private:
  bool bias_;
};
