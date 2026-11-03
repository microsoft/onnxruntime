class HCPostNormProjectionProgram final : public Program<HCPostNormProjectionProgram> {
 public:
  HCPostNormProjectionProgram(bool shared, bool shared_scale)
      : Program{"HCPostNormProjectionF16"}, shared_{shared}, shared_scale_{shared_scale} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& streams = shader.AddInput("streams");
    const auto& branch = shader.AddInput("branch");
    const auto& gate = shader.AddInput("gate");
    const auto& scale = shader.AddInput("norm_scale");
    const ShaderVariableHelper* shared = nullptr;
    const ShaderVariableHelper* shared_gate = nullptr;
    if (shared_) {
      shared = &shader.AddInput("shared_output");
      shared_gate = &shader.AddInput("shared_gate");
    }
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("projection_scales", ShaderUsage::UseValueTypeAlias);
    const auto& y = shader.AddOutput("y");
    const auto& normalized = shader.AddOutput("normalized");
    const auto& activated = shader.AddOutput("activated", ShaderUsage::UseElementTypeAlias);
    const auto& tail = shader.AddOutput("tail");
    shader.AdditionalImplementation() <<
        "var<workgroup> sums: array<f32, 1024>;\n"
        "var<workgroup> post_values: array<f16, 11520>;\n"
        "var<workgroup> partials: array<f16, 256>;\n"
        "fn stable_sigmoid(v: f32) -> f32 {\n"
        "  if (v >= 0.0) { return 1.0 / (1.0 + exp(-v)); }\n"
        "  let e = exp(v); return e / (1.0 + e);\n}\n";
    auto& code = shader.MainFunctionBody();
    if (shared_) {
      code << "let shared_sigmoid = f16(1.0 / (1.0 + exp(-"
           << shared_gate->GetByOffset("0u") << ")));\n";
    }
    for (int stream = 0; stream < 4; ++stream) {
      code << "let divided_gate" << stream << " = f16(" << gate.GetByOffset(std::to_string(stream) + "u")
           << " / f16(4.0));\n"
           << "let sigmoid_gate" << stream << " = f16(1.0 / (1.0 + exp(-divided_gate" << stream << ")));\n"
           << "let gate_value" << stream << " = f16(sigmoid_gate" << stream << " * f16(2.0));\n";
    }
    code << "for (var h = local_idx; h < uniforms.hidden; h += 256u) {\n";
    if (shared_) {
      code << "let branch_value = f16(" << branch.GetByOffset("h") << " + ("
           << shared->GetByOffset("h") << " * shared_sigmoid));\n";
    } else {
      code << "let branch_value = " << branch.GetByOffset("h") << ";\n";
    }
    for (int stream = 0; stream < 4; ++stream) {
      const auto suffix = std::to_string(stream);
      code << "let logical" << stream << " = " << stream << "u * uniforms.hidden + h;\n"
           << "let address" << stream << " = (logical" << stream << " / 32u) * 36u + logical" << stream << " % 32u;\n"
           << "var value" << stream << " = f32(" << streams.GetByOffset("logical" + suffix) << ");\n"
           << "value" << stream << " += f32(gate_value" << stream << ") * f32(branch_value);\n"
           << "let stored" << stream << " = f16(value" << stream << ");\n"
           << "post_values[address" << stream << "] = stored" << stream << ";\n"
           << "if (workgroup_idx == 0u) {\n"
           << y.SetByOffset("logical" + suffix, "stored" + suffix) << "\n}\n";
    }
    code << "}\nworkgroupBarrier();\n";
    for (int stream = 0; stream < 4; ++stream) code << "var sum" << stream << " = 0.0;\n";
    code << "for (var h = local_idx; h < uniforms.hidden; h += 256u) {\n";
    for (int stream = 0; stream < 4; ++stream) {
      code << "let logical" << stream << " = " << stream << "u * uniforms.hidden + h;\n"
           << "let address" << stream << " = (logical" << stream << " / 32u) * 36u + logical" << stream << " % 32u;\n"
           << "let v" << stream << " = f32(post_values[address" << stream << "]);\n"
           << "sum" << stream << " += v" << stream << " * v" << stream << ";\n";
    }
    code << "}\n";
    for (int stream = 0; stream < 4; ++stream) {
      code << "sums[" << stream * 256 << "u + local_idx] = sum" << stream << ";\n";
    }
    code << "workgroupBarrier();\n"
         << "for (var width = 128u; width > 0u; width >>= 1u) {\n"
         << "if (local_idx < width) {\n";
    for (int stream = 0; stream < 4; ++stream) {
      code << "sums[" << stream * 256 << "u + local_idx] += sums["
           << stream * 256 << "u + local_idx + width];\n";
    }
    code << "}\nworkgroupBarrier();\n}\n";
    for (int stream = 0; stream < 4; ++stream) {
      code << "let inv_rms" << stream << " = inverseSqrt(sums[" << stream * 256
           << "u] / f32(uniforms.hidden) + uniforms.epsilon);\n";
    }
    code << "for (var h = local_idx; h < uniforms.hidden; h += 256u) {\n";
    for (int stream = 0; stream < 4; ++stream) {
      const auto suffix = std::to_string(stream);
      code << "let logical" << stream << " = " << stream << "u * uniforms.hidden + h;\n"
           << "let address" << stream << " = (logical" << stream << " / 32u) * 36u + logical" << stream << " % 32u;\n"
           << "let weight" << stream << " = f32(" << scale.GetByOffset(shared_scale_ ? "h" : "logical" + suffix) << ");\n"
           << "let value" << stream << " = f16(f32(post_values[address" << stream << "]) * inv_rms"
           << stream << " * weight" << stream << ");\n"
           << "post_values[address" << stream << "] = value" << stream << ";\n"
           << "if (workgroup_idx == 0u) {\n"
           << normalized.SetByOffset("logical" + suffix, "value" + suffix) << "\n}\n";
    }
    code << "}\nworkgroupBarrier();\n"
         << "let lane = local_idx % 32u;\n"
         << "let col = workgroup_idx * 8u + local_idx / 32u;\n"
         << "var total = f16(0);\n"
         << "if (col < 324u) {\n"
         << "for (var tile = 0u; tile < 10u; tile++) {\n"
         << "  let block = tile * 32u + lane;\n"
         << "  let packed = " << b.GetByOffset("col * 320u + block") << ";\n"
         << "  let scale_value = " << scales.GetByOffset("col * 320u + block") << ";\n"
         << "  var sum = f16(0);\n"
         << "  for (var i = 0u; i < 4u; i++) {\n"
         << "    let low = vec4<f16>(unpack4xU8(packed[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "    let high = vec4<f16>(unpack4xU8((packed[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "    let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale_value;\n"
         << "    let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale_value;\n"
         << "    let base = block * 36u + i * 8u;\n"
         << "    let x0 = vec4<f16>(post_values[base], post_values[base + 1u], post_values[base + 2u], post_values[base + 3u]);\n"
         << "    let x1 = vec4<f16>(post_values[base + 4u], post_values[base + 5u], post_values[base + 6u], post_values[base + 7u]);\n"
         << "    sum += dot(x0, w0) + dot(x1, w1);\n"
         << "  }\n"
         << "  total += sum;\n"
         << "}\n}\n"
         << "partials[local_idx] = total;\n"
         << "workgroupBarrier();\n"
         << "if (lane == 0u && col < 324u) {\n"
         << "  var value = f16(0);\n"
         << "  for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }\n"
         << "  if (col < uniforms.prefix) {\n"
         << "    let z_t = activated_element_t(f32(value) * uniforms.alpha);\n"
         << "    let sigmoid_t = activated_element_t(stable_sigmoid(f32(z_t)));\n"
         << activated.SetByOffset("col", "activated_element_t(f32(z_t) * f32(sigmoid_t))") << "\n"
         << "  } else {\n"
         << tail.SetByOffset("col - uniforms.prefix", "value") << "\n"
         << "  }\n}\n";
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"hidden", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32},
      {"prefix", ProgramUniformVariableDataType::Uint32},
      {"alpha", ProgramUniformVariableDataType::Float32});

 private:
  bool shared_;
  bool shared_scale_;
};
