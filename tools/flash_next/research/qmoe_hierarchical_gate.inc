class QmoeHierarchicalGateProgram final : public Program<QmoeHierarchicalGateProgram> {
 public:
  QmoeHierarchicalGateProgram() : Program{"QmoeHierarchicalGate1Token"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& logits = shader.AddInput("router_logits");
    const auto& values = shader.AddOutput("topk_values");
    const auto& experts = shader.AddOutput("indirect_experts");
    shader.AdditionalImplementation() <<
        "var<workgroup> candidates: array<u32, 160>;\n"
        "var<workgroup> selected_vals: array<f16, 10>;\n"
        "var<workgroup> selected_idxs: array<u32, 10>;\n"
        "fn ordered_key(value: f32) -> u32 {\n"
        "  let bits = bitcast<u32>(select(value, 0.0, value == 0.0));\n"
        "  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);\n"
        "}\n";
    auto& code = shader.MainFunctionBody();
    code << "let row = workgroup_idx;\n"
         << "if (row >= uniforms.rows) { return; }\n"
         << "let cols = uniforms.cols;\n"
         << "let output_base = row * cols;\n"
         << "var candidate_idx = local_idx;\n"
         << "let candidate_val = " << logits.GetByOffset("output_base + local_idx") << ";\n"
         << values.SetByOffset("output_base + local_idx", "f16(0)") << "\n"
         << "for (var rank = 0u; rank < 10u; rank++) {\n"
         << "  let key = ordered_key(f32(candidate_val));\n"
         << "  let best_key = subgroupMax(select(0u, key, candidate_idx < cols));\n"
         << "  let best_idx = subgroupMin(select(0xffffffffu, candidate_idx, candidate_idx < cols && key == best_key));\n"
         << "  if (sg_id == 0u) { candidates[subgroup_index * 10u + rank] = best_idx; }\n"
         << "  if (candidate_idx == best_idx) { candidate_idx = cols; }\n"
         << "}\n"
         << "workgroupBarrier();\n"
         << "storageBarrier();\n"
         << "if (subgroup_index == 0u) {\n"
         << "  let groups = (workgroup_size_x + sg_size - 1u) / sg_size;\n"
         << "  var cursor = 0u;\n"
         << "  var head_idx = cols;\n"
         << "  if (sg_id < groups) { head_idx = candidates[sg_id * 10u]; }\n"
         << "  for (var rank = 0u; rank < 10u; rank++) {\n"
         << "    var head_val = f16(-65504.0);\n"
         << "    if (head_idx < cols) { head_val = " << logits.GetByOffset("output_base + head_idx") << "; }\n"
         << "    let key = ordered_key(f32(head_val));\n"
         << "    let best_key = subgroupMax(select(0u, key, head_idx < cols));\n"
         << "    let best_idx = subgroupMin(select(0xffffffffu, head_idx, head_idx < cols && key == best_key));\n"
         << "    if (sg_id == 0u) {\n"
         << "      selected_idxs[rank] = best_idx;\n"
         << "      selected_vals[rank] = " << logits.GetByOffset("output_base + best_idx") << ";\n"
         << "    }\n"
         << "    if (head_idx == best_idx && sg_id < groups) {\n"
         << "      cursor++;\n"
         << "      head_idx = cols;\n"
         << "      if (cursor < 10u) { head_idx = candidates[sg_id * 10u + cursor]; }\n"
         << "    }\n"
         << "  }\n"
         << "}\n"
         << "workgroupBarrier();\n"
         << "if (local_idx == 0u) {\n"
         << "  let max_val = f32(selected_vals[0]);\n"
         << "  var sum = 0.0;\n"
         << "  for (var rank = 0u; rank < 10u; rank++) { sum += exp(f32(selected_vals[rank]) - max_val); }\n"
         << "  for (var rank = 0u; rank < 10u; rank++) {\n"
         << "    let expert = selected_idxs[rank];\n"
         << values.SetByOffset("output_base + expert", "f16(exp(f32(selected_vals[rank]) - max_val) / sum)") << "\n"
         << experts.SetByOffset("row * 10u + rank", "expert") << "\n"
         << "  }\n"
         << "}\n";
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32});
};
