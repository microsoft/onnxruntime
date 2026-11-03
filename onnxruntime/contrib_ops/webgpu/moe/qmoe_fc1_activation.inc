class QmoeFC1SwiGLUProgram final : public Program<QmoeFC1SwiGLUProgram> {
 public:
  explicit QmoeFC1SwiGLUProgram(bool bias, bool bitcast) : Program{bitcast ? "QmoeFC1SwiGLUBitcastF16" : "QmoeFC1SwiGLUF16"}, bias_{bias}, bitcast_{bitcast} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    if (bias_) shader.AddInput("bias", ShaderUsage::UseElementTypeAlias);
    shader.AddInput("experts", ShaderUsage::UseValueTypeAlias);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation()
        << "var<workgroup> tile_a: array<vec4<f16>, 256>;\n"
        << "var<workgroup> partials: array<array<f16, 32>, 8>;\n";
    if (bitcast_) {
      shader.AdditionalImplementation() <<
          "fn decode_low_nibbles(word: u32) -> vec4<f16> {\n"
          "  let bottom = bitcast<vec2<f16>>((word & 0x000f000fu) | 0x64006400u) - vec2<f16>(1032.0);\n"
          "  let top = bitcast<vec2<f16>>(((word >> 8u) & 0x000f000fu) | 0x64006400u) - vec2<f16>(1032.0);\n"
          "  return vec4<f16>(bottom.x, top.x, bottom.y, top.y);\n"
          "}\n";
    }
    auto& code = shader.MainFunctionBody();
    code << "let row = workgroup_idx / uniforms.tiles;\n"
         << "let base_col = (workgroup_idx % uniforms.tiles) * 8u;\n"
         << "let lane = local_idx % 32u;\n"
         << "let group = local_idx / 32u;\n"
         << "let expert = experts[row];\n"
         << "let blocks = uniforms.K / 32u;\n"
         << "for (var kidx = 0u; kidx < uniforms.K; kidx += 1024u) {\n"
         << "  for (var id = local_idx; id < 256u; id += 128u) {\n"
         << "    let offset = kidx / 4u + id;\n"
         << "    if (offset < uniforms.K / 4u) {\n"
         << "      tile_a[id] = " << a.GetByOffset("offset") << ";\n"
         << "    } else { tile_a[id] = vec4<f16>(0); }\n"
         << "  }\n"
         << "  workgroupBarrier();\n"
         << "  for (var offset = 0u; offset < 8u; offset += 4u) {\n"
         << "    let col = base_col + offset + group;\n"
         << "    let block = kidx / 32u + lane;\n"
         << "    if (col < uniforms.cols * 2u && block < blocks) {\n"
         << "      let index = (expert * uniforms.cols * 2u + col) * blocks + block;\n"
         << "      let packed = " << b.GetByOffset("index") << ";\n"
         << "      let scale = " << scales.GetByOffset("index") << ";\n"
         << "      var sum = f16(0);\n"
         << "      for (var i = 0u; i < 4u; i++) {\n"
         << (bitcast_ ? "        let low = decode_low_nibbles(packed[i]);\n" : "        let low = vec4<f16>(unpack4xU8(packed[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\n")
         << (bitcast_ ? "        let high = decode_low_nibbles(packed[i] >> 4u);\n" : "        let high = vec4<f16>(unpack4xU8((packed[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n")
         << "        let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale;\n"
         << "        let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale;\n"
         << "        sum += dot(tile_a[lane * 8u + i * 2u], w0) + dot(tile_a[lane * 8u + i * 2u + 1u], w1);\n"
         << "      }\n"
         << "      partials[offset + group][lane] += sum;\n"
         << "    }\n"
         << "  }\n"
         << "  workgroupBarrier();\n"
         << "}\n"
         << "if (local_idx < 8u && base_col + local_idx < uniforms.cols * 2u) {\n"
         << "  var value = f16(0);\n"
         << "  for (var i = 0u; i < 32u; i++) { value += partials[local_idx][i]; }\n";
    if (bias_) code << "  value += bias[expert * uniforms.cols * 2u + base_col + local_idx];\n";
    code << "  partials[local_idx][0] = value;\n"
         << "}\n"
         << "workgroupBarrier();\n"
         << "if (local_idx < 4u) {\n"
         << "  let col = base_col / 2u + local_idx;\n"
         << "  if (col < uniforms.cols) {\n"
         << "    var gate_value = f32(partials[local_idx * 2u][0]);\n"
         << "    var linear_value = f32(partials[local_idx * 2u + 1u][0]);\n"
         << "    gate_value = min(gate_value, uniforms.swiglu_limit);\n"
         << "    linear_value = clamp(linear_value, -uniforms.swiglu_limit, uniforms.swiglu_limit);\n"
         << "    let glu = gate_value / (1.0 + exp(-uniforms.alpha * gate_value));\n"
         << output.SetByOffset("row * uniforms.cols + col", "f16(glu * (linear_value + uniforms.beta))") << "\n"
         << "  }\n}\n";
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"K", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32},
      {"tiles", ProgramUniformVariableDataType::Uint32},
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32},
      {"swiglu_limit", ProgramUniformVariableDataType::Float32});

 private:
  bool bias_;
  bool bitcast_;
};

bool QMoE::CanFuseFc1Activation(ComputeContext& context, bool enabled) const {
  const auto* input = context.Input<Tensor>(0);
  const auto* scales = context.Input<Tensor>(3);
  const auto* weights = context.Input<Tensor>(2);
  const auto* bias = context.Input<Tensor>(4);
  return enabled && activation_type_ == MoEActivationType::SwiGLU &&
         (swiglu_fusion_ == 0 || swiglu_fusion_ == 1) &&
         !is_block_fp8_ && !is_nvfp4_ && fc1_expert_weight_bits_ == 4 && block_size_ == 32 &&
         k_ == 10 && input != nullptr && input->IsDataType<MLFloat16>() &&
         input->Shape().NumDimensions() > 0 && input->Shape().GetDims().back() == 2560 &&
         input->Shape().Size() == 2560 && scales != nullptr && scales->IsDataType<MLFloat16>() &&
         weights != nullptr && weights->IsDataType<uint8_t>() &&
         weights->Shape().NumDimensions() == 3 && weights->Shape().GetDims()[2] == 1280 &&
         scales->Shape().NumDimensions() == 3 && scales->Shape().GetDims()[2] == 80 &&
         scales->Shape().GetDims()[0] == weights->Shape().GetDims()[0] &&
         scales->Shape().GetDims()[1] == weights->Shape().GetDims()[1] &&
         (bias == nullptr || bias->IsDataType<MLFloat16>()) &&
         context.Input<Tensor>(8) == nullptr && context.Input<Tensor>(11) == nullptr &&
         context.Input<Tensor>(14) == nullptr && !context.EnableMatmulFp32Accumulation() &&
         context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
         ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_INT4_COOPERATIVE", 0) == 0 &&
         ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_MATMUL_NBITS_SKINNY", 0) == 0 &&
         ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY", 0) == 0;
}
