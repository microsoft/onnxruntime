class QsaParallelMapProgram final : public Program<QsaParallelMapProgram> {
 public:
  QsaParallelMapProgram(bool has_mask, bool count_only)
      : Program{count_only ? "QsaParallelMapCount" : "QsaParallelMapScatter"},
        has_mask_{has_mask}, count_only_{count_only} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AdditionalImplementation() << "fn visible(token: u32) -> bool {\n";
    if (has_mask_) {
      const auto& mask = shader.AddInput("mask");
      shader.AdditionalImplementation()
          << "let value = " << mask.GetByOffset("token", true) << ";\n"
          << "return value.x != 0u || value.y != 0u;\n";
    } else {
      shader.AdditionalImplementation() << "return true;\n";
    }
    shader.AdditionalImplementation()
        << "}\nvar<workgroup> flags: array<u32, 64>;\n"
        << "var<workgroup> chunk_count_shared: u32;\n"
        << "var<workgroup> offset_shared: u32;\n";
    const ShaderVariableHelper* counts = nullptr;
    if (!count_only_) counts = &shader.AddInput("counts");
    const auto& result = shader.AddOutput("result");
    auto& code = shader.MainFunctionBody();
    code << "let begin = workgroup_idx * 512u;\n"
         << "let end = min(begin + 512u, uniforms.length);\n"
         << "var total = 0u;\n";
    if (count_only_) {
      code << "for (var token = begin + local_idx; token < end; token += 64u) {\n"
           << "  total += select(0u, 1u, visible(token));\n"
           << "}\n";
    } else {
      code << "for (var chunk = local_idx; chunk < workgroup_idx; chunk += 64u) {\n"
           << "  total += " << counts->GetByOffset("chunk") << ";\n"
           << "}\n"
           << "if (local_idx == 0u) { chunk_count_shared = "
           << counts->GetByOffset("workgroup_idx") << "; }\n";
    }
    code << "flags[local_idx] = total;\nworkgroupBarrier();\n"
         << "for (var stride = 32u; stride > 0u; stride /= 2u) {\n"
         << "  if (local_idx < stride) { flags[local_idx] += flags[local_idx + stride]; }\n"
         << "  workgroupBarrier();\n}\n";
    if (count_only_) {
      code << "if (local_idx == 0u) { "
           << result.SetByOffset("workgroup_idx", "flags[0]") << " }\n";
      return Status::OK();
    }
    code << "let prefix = workgroupUniformLoad(&flags[0]);\n"
         << "let count = workgroupUniformLoad(&chunk_count_shared);\n"
         << "if (local_idx == 0u && end == uniforms.length) { "
         << result.SetByOffset("uniforms.length", "prefix + count") << " }\n"
         << "if (count == 0u || count == end - begin) {\n"
         << "  if (count != 0u) {\n"
         << "    for (var token = begin + local_idx; token < end; token += 64u) {\n"
         << "      " << result.SetByOffset("prefix + token - begin", "token") << "\n"
         << "    }\n  }\n  return;\n}\n"
         << "if (local_idx == 0u) { offset_shared = prefix; }\n"
         << "workgroupBarrier();\n"
         << "for (var base = begin; base < end; base += 64u) {\n"
         << "  let token = base + local_idx;\n"
         << "  flags[local_idx] = select(0u, 1u, token < end && visible(token));\n"
         << "  workgroupBarrier();\n"
         << "  var rank = 0u;\n"
         << "  for (var lane = 0u; lane < local_idx; lane++) { rank += flags[lane]; }\n"
         << "  let offset = offset_shared;\n"
         << "  if (flags[local_idx] != 0u) { "
         << result.SetByOffset("offset + rank", "token") << " }\n"
         << "  workgroupBarrier();\n"
         << "  if (local_idx == 63u) { offset_shared = offset + rank + flags[local_idx]; }\n"
         << "  workgroupBarrier();\n}\n";
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"length", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_mask_;
  bool count_only_;
};
