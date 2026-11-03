class QmoeGroupedCountProgram final : public Program<QmoeGroupedCountProgram> {
 public:
  QmoeGroupedCountProgram() : Program{"QmoeGroupedExpertCount"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"rows", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& experts = shader.AddInput("experts", ShaderUsage::UseUniform);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
    shader.AdditionalImplementation() << "var<workgroup> counts: array<u32, 64>;\n";
    shader.MainFunctionBody()
        << "var count = 0u;\n"
        << "for (var row = local_idx; row < uniforms.rows; row += 64u) {\n"
        << "  if (" << experts.GetByOffset("row") << " == workgroup_idx) { count++; }\n"
        << "}\n"
        << "counts[local_idx] = count;\nworkgroupBarrier();\n"
        << "for (var width = 32u; width > 0u; width >>= 1u) {\n"
        << "  if (local_idx < width) { counts[local_idx] += counts[local_idx + width]; }\n"
        << "  workgroupBarrier();\n"
        << "}\n"
        << "if (local_idx == 0u) { " << output.SetByOffset("workgroup_idx", "counts[0]") << " }\n";
    return Status::OK();
  }
};

class QmoeGroupedOffsetProgram final : public Program<QmoeGroupedOffsetProgram> {
 public:
  QmoeGroupedOffsetProgram() : Program{"QmoeGroupedExpertOffsets"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"experts", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& counts = shader.AddInput("counts", ShaderUsage::UseUniform);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
    shader.MainFunctionBody()
        << "if (local_idx != 0u) { return; }\n"
        << "var offset = 0u;\n"
        << output.SetByOffset("0u", "0u") << "\n"
        << "for (var expert = 0u; expert < uniforms.experts; expert++) {\n"
        << "  offset += " << counts.GetByOffset("expert") << ";\n"
        << "  " << output.SetByOffset("expert + 1u", "offset") << "\n"
        << "}\n";
    return Status::OK();
  }
};

class QmoeGroupedPermutationProgram final : public Program<QmoeGroupedPermutationProgram> {
 public:
  QmoeGroupedPermutationProgram() : Program{"QmoeGroupedExpertPermutation"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"rows", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& experts = shader.AddInput("experts", ShaderUsage::UseUniform);
    const auto& offsets = shader.AddInput("offsets", ShaderUsage::UseUniform);
    const auto& rows = shader.AddOutput("rows", ShaderUsage::UseUniform);
    const auto& sorted = shader.AddOutput("sorted", ShaderUsage::UseUniform);
    shader.AdditionalImplementation() << "var<workgroup> cursor: atomic<u32>;\n";
    shader.MainFunctionBody()
        << "if (local_idx == 0u) { atomicStore(&cursor, 0u); }\n"
        << "workgroupBarrier();\n"
        << "for (var row = local_idx; row < uniforms.rows; row += 64u) {\n"
        << "  if (" << experts.GetByOffset("row") << " == workgroup_idx) {\n"
        << "    let position = " << offsets.GetByOffset("workgroup_idx") << " + atomicAdd(&cursor, 1u);\n"
        << "    " << rows.SetByOffset("position", "row") << "\n"
        << "    " << sorted.SetByOffset("position", "workgroup_idx") << "\n"
        << "  }\n"
        << "}\n";
    return Status::OK();
  }
};

class QmoeGroupedCopyProgram final : public Program<QmoeGroupedCopyProgram> {
 public:
  explicit QmoeGroupedCopyProgram(bool scatter)
      : Program{scatter ? "QmoeGroupedOutputScatter" : "QmoeGroupedHiddenGather"}, scatter_{scatter} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"hidden", ProgramUniformVariableDataType::Uint32},
      {"k", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& data = shader.AddInput("data", ShaderUsage::UseUniform);
    const auto& rows = shader.AddInput("rows", ShaderUsage::UseUniform);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
    shader.MainFunctionBody()
        << "if (global_idx >= uniforms.rows * uniforms.hidden) { return; }\n"
        << "let sorted_row = global_idx / uniforms.hidden;\n"
        << "let column = global_idx % uniforms.hidden;\n"
        << "let original_row = " << rows.GetByOffset("sorted_row") << ";\n"
        << output.SetByOffset(scatter_ ? "original_row * uniforms.hidden + column" : "global_idx",
                              data.GetByOffset(scatter_ ? "global_idx" :
                                               "(original_row / uniforms.k) * uniforms.hidden + column")) << "\n";
    return Status::OK();
  }
 private:
  bool scatter_;
};
