class ExactBranchNormProgram final : public Program<ExactBranchNormProgram> {
 public:
  ExactBranchNormProgram(bool has_scale, bool shared_scale)
      : Program{"ExactBranchNorm"}, has_scale_{has_scale}, shared_scale_{shared_scale} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& x = shader.AddInput("x", ShaderUsage::UseElementTypeAlias);
    const ShaderVariableHelper* scale = nullptr;
    if (has_scale_) scale = &shader.AddInput("scale");
    const auto& y = shader.AddOutput("y", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << "var<workgroup> partials: array<f32, 256>;\n";
    auto& code = shader.MainFunctionBody();
    code << "let base = workgroup_idx * uniforms.hidden;\n"
         << "var partial = 0.0;\n"
         << "for (var h = local_idx; h < uniforms.hidden; h += 256u) {\n"
         << "  let v = f32(" << x.GetByOffset("base + h") << ");\n"
         << "  partial += v * v;\n"
         << "}\n"
         << "partials[local_idx] = partial;\n"
         << "workgroupBarrier();\n"
         << "let lane = local_idx % 32u;\n"
         << "let even = (partials[lane] + partials[lane + 128u]) +\n"
         << "           (partials[lane + 64u] + partials[lane + 192u]);\n"
         << "let odd = (partials[lane + 32u] + partials[lane + 160u]) +\n"
         << "          (partials[lane + 96u] + partials[lane + 224u]);\n"
         << "var total = even + odd;\n"
         << "for (var offset = 16u; offset > 0u; offset >>= 1u) {\n"
         << "  let other = subgroupShuffleDown(total, offset);\n"
         << "  if (lane < offset) { total += other; }\n"
         << "}\n"
         << "let sum = subgroupBroadcastFirst(total);\n"
         << "let inv_rms = inverseSqrt(sum / f32(uniforms.hidden) + uniforms.epsilon);\n"
         << "for (var h = local_idx; h < uniforms.hidden; h += 256u) {\n"
         << "  var weight = 1.0;\n";
    if (has_scale_) {
      code << "  weight = f32(" << scale->GetByOffset(
          shared_scale_ ? "h" : "(workgroup_idx % uniforms.branches) * uniforms.hidden + h") << ");\n";
    }
    code << y.SetByOffset("base + h", "y_element_t(f32(" + x.GetByOffset("base + h") +
                                    ") * inv_rms * weight)") << "\n}\n";
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"hidden", ProgramUniformVariableDataType::Uint32},
      {"branches", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32});

 private:
  bool has_scale_;
  bool shared_scale_;
};
