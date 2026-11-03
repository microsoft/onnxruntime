#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "contrib_ops/webgpu/bert/scaled_silu.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/shader_helper.h"
#include "static_node_replay.h"

#include "contrib_ops/cpu/hyper_connection_helper.h"

namespace onnxruntime::contrib::webgpu {
using namespace onnxruntime::webgpu;

class HCPostNormProjectionBaselineNormProgram final : public Program<HCPostNormProjectionBaselineNormProgram> {
 public:
  HCPostNormProjectionBaselineNormProgram(bool shared, bool shared_scale, uint32_t hidden)
      : Program{"HCPostNormF16"}, shared_{shared}, shared_scale_{shared_scale}, hidden_{hidden} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& streams = shader.AddInput("streams");
    const auto& branch = shader.AddInput("branch");
    const auto& gate = shader.AddInput("gate");
    const auto& scale = shader.AddInput("scale");
    const ShaderVariableHelper* shared = nullptr;
    const ShaderVariableHelper* shared_gate = nullptr;
    if (shared_) {
      shared = &shader.AddInput("shared_output");
      shared_gate = &shader.AddInput("shared_gate");
    }
    const auto& y = shader.AddOutput("y");
    const auto& normalized = shader.AddOutput("normalized");
    shader.AdditionalImplementation() << "var<workgroup> sums: array<f32, 256>;\n";
    shader.AdditionalImplementation() << "var<workgroup> post_values: array<f16, " << hidden_ << ">;\n";
    auto& code = shader.MainFunctionBody();
    code << "let group = workgroup_idx;\n"
         << "let row = group / 4u;\n"
         << "let base = group * uniforms.hidden;\n"
         << "let divided_gate = f16(" << gate.GetByOffset("group") << " / f16(4.0));\n"
         << "let sigmoid_gate = f16(1.0 / (1.0 + exp(-divided_gate)));\n"
         << "let gate_value = f16(sigmoid_gate * f16(2.0));\n";
    if (shared_) {
      code << "let shared_sigmoid = f16(1.0 / (1.0 + exp(-"
           << shared_gate->GetByOffset("row") << ")));\n";
    }
    code << "for (var h = local_idx; h < uniforms.hidden; h += 256u) {\n"
         << "  let branch_offset = row * uniforms.hidden + h;\n";
    if (shared_) {
      code << "let branch_value = f16(" << branch.GetByOffset("branch_offset") << " + ("
           << shared->GetByOffset("branch_offset") << " * shared_sigmoid));\n";
    } else {
      code << "let branch_value = " << branch.GetByOffset("branch_offset") << ";\n";
    }
    code << "  var value = f32(" << streams.GetByOffset("base + h") << ");\n"
         << "  value += f32(gate_value) * f32(branch_value);\n"
         << "  let stored = f16(value);\n"
         << "  post_values[h] = stored;\n"
         << y.SetByOffset("base + h", "stored") << "\n"
         << "}\n"
         << "workgroupBarrier();\n"
         << "var sum = 0.0;\n"
         << "for (var h = local_idx; h < uniforms.hidden; h += 256u) {\n"
         << "  let v = f32(post_values[h]);\n"
         << "  sum += v * v;\n"
         << "}\n"
         << "sums[local_idx] = sum;\n"
         << "workgroupBarrier();\n"
         << "for (var width = 128u; width > 0u; width >>= 1u) {\n"
         << "  if (local_idx < width) { sums[local_idx] += sums[local_idx + width]; }\n"
         << "  workgroupBarrier();\n"
         << "}\n"
         << "let inv_rms = inverseSqrt(sums[0] / f32(uniforms.hidden) + uniforms.epsilon);\n"
         << "for (var h = local_idx; h < uniforms.hidden; h += 256u) {\n"
         << "  let weight = f32(" << scale.GetByOffset(
                shared_scale_ ? "h" : "(group % 4u) * uniforms.hidden + h") << ");\n"
         << normalized.SetByOffset("base + h", "f16(f32(post_values[h]) * inv_rms * weight)") << "\n"
         << "}\n";
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"hidden", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32});

 private:
  bool shared_;
  bool shared_scale_;
  uint32_t hidden_;
};

class HCPostNormProjectionBaselineMatMulProgram final : public Program<HCPostNormProjectionBaselineMatMulProgram> {
 public:
  HCPostNormProjectionBaselineMatMulProgram() : Program{"HCProjectionSiLUF16"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias);
    const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseValueTypeAlias);
    const auto& activated = shader.AddOutput("activated", ShaderUsage::UseElementTypeAlias);
    const auto& tail = shader.AddOutput("tail");
    shader.AdditionalImplementation() <<
        "fn stable_sigmoid(v: f32) -> f32 {\n"
        "  if (v >= 0.0) { return 1.0 / (1.0 + exp(-v)); }\n"
        "  let e = exp(v); return e / (1.0 + e);\n}\n";
    shader.AdditionalImplementation() << "var<workgroup> partials: array<f16, 64>;\n";
    auto& code = shader.MainFunctionBody();
    code << "let lane = local_idx % 32u;\n"
         << "let col = workgroup_idx * 2u + local_idx / 32u;\n"
         << "var total = f16(0);\n"
         << "for (var tile = 0u; tile < 10u; tile++) {\n"
         << "  let block = tile * 32u + lane;\n"
         << "  let packed = " << b.GetByOffset("col * 320u + block") << ";\n"
         << "  let scale = " << scales.GetByOffset("col * 320u + block") << ";\n"
         << "  var sum = f16(0);\n"
         << "  for (var i = 0u; i < 4u; i++) {\n"
         << "    let low = vec4<f16>(unpack4xU8(packed[i] & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "    let high = vec4<f16>(unpack4xU8((packed[i] >> 4u) & 0x0f0f0f0fu)) - vec4<f16>(8);\n"
         << "    let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale;\n"
         << "    let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale;\n"
         << "    let x0 = " << a.GetByOffset("block * 8u + i * 2u") << ";\n"
         << "    let x1 = " << a.GetByOffset("block * 8u + i * 2u + 1u") << ";\n"
         << "    sum += dot(x0, w0) + dot(x1, w1);\n"
         << "  }\n"
         << "  total += sum;\n"
         << "}\n"
         << "partials[local_idx] = total;\n"
         << "workgroupBarrier();\n"
         << "if (lane == 0u) {\n"
         << "  var value = f16(0);\n"
         << "  for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }\n"
         << "  if (col < uniforms.prefix) {\n"
         << "    let z_t = activated_element_t(f32(value) * uniforms.alpha);\n"
         << "    let sigmoid_t = activated_element_t(stable_sigmoid(f32(z_t)));\n"
         << activated.SetByOffset("col", "activated_element_t(f32(z_t) * f32(sigmoid_t))") << "\n"
         << "  } else {\n"
         << tail.SetByOffset("col - uniforms.prefix", "value") << "\n"
         << "  }\n"
         << "}\n";
    return Status::OK();
  }
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"prefix", ProgramUniformVariableDataType::Uint32},
      {"alpha", ProgramUniformVariableDataType::Float32});
};

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

class HCPostNormProjectionF16 : public WebGpuKernel {
 public:
  explicit HCPostNormProjectionF16(const OpKernelInfo& info) : HCPostNormProjectionF16(info, false) {}

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {
    const int enabled = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_POST_NORM_PROJECTION", 0);
    const int projection_mode = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_PROJECTION_SILU", 0);
    ORT_RETURN_IF_NOT(enabled == 0 || enabled == 1, "ORT_WEBGPU_HC_POST_NORM_PROJECTION must be 0 or 1");
    ORT_RETURN_IF_NOT(projection_mode == 0 || projection_mode == 1, "ORT_WEBGPU_HC_PROJECTION_SILU must be 0 or 1");
    const int norm_inputs = shared_ ? 6 : 4;
    for (int index = 0; index < norm_inputs; ++index) {
      ORT_RETURN_IF_NOT(context.Input(index)->IsDataType<MLFloat16>(),
                        "HCPostNormProjectionF16 requires FP16 normalization inputs");
    }
    const auto* streams = context.Input(0);
    const auto* scale = context.Input(3);
    const auto* sizes = context.Input(norm_inputs);
    const auto* weights = context.Input(norm_inputs + 1);
    const auto* scales = context.Input(norm_inputs + 2);
    ORT_RETURN_IF_NOT(sizes->IsDataType<int64_t>() && weights->IsDataType<uint8_t>() &&
                          scales->IsDataType<MLFloat16>() && sizes->Shape() == TensorShape({2}) &&
                          weights->Shape() == TensorShape({324, 320, 16}) &&
                          scales->Shape().Size() == 324 * 320,
                      "HCPostNormProjectionF16 requires canonical INT4 weights/scales and INT64 split sizes");
    hyper_connection::StreamShape params;
    ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(streams->Shape(), 4, params));
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateReduced(context.Input(1)->Shape(), params));
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateScale(scale->Shape(), params));
    hyper_connection::GateLayout layout;
    ORT_RETURN_IF_ERROR(hyper_connection::ResolveGateShape(
        context.Input(2)->Shape(), streams->Shape(), params, false, layout));
    ORT_RETURN_IF_NOT(layout == hyper_connection::GateLayout::Branch &&
                          streams->Shape().NumDimensions() > 0 && streams->Shape().GetDims().back() == 10240,
                      "HCPostNormProjectionF16 requires four flattened branches and one gate per branch");
    if (shared_) {
      auto expected_gate = params.reduced_shape;
      expected_gate.back() = 1;
      ORT_RETURN_IF_NOT(context.Input(4)->Shape() == context.Input(1)->Shape() &&
                            context.Input(5)->Shape() == TensorShape(expected_gate),
                        "HCPostNormProjectionF16 received incompatible shared branch/gate shapes");
    }
    const auto* split = sizes->Data<int64_t>();
    ORT_RETURN_IF_NOT(split[0] > 0 && split[0] < 324 && split[1] == 324 - split[0],
                      "HCPostNormProjectionF16 requires positive split sizes summing to 324");
    MatMulComputeHelper helper;
    ORT_RETURN_IF_ERROR(helper.Compute(streams->Shape(), TensorShape({324, 10240}), false, true));
    TensorShapeVector shape(helper.OutputShape().GetDims().begin(), helper.OutputShape().GetDims().end());
    auto prefix_shape = shape;
    auto suffix_shape = shape;
    prefix_shape.back() = split[0];
    suffix_shape.back() = split[1];
    auto* y = context.Output(0, streams->Shape());
    auto* normalized = context.Output(1, streams->Shape());
    auto* activated = context.Output(2, TensorShape(prefix_shape));
    auto* tail = context.Output(3, TensorShape(suffix_shape));
    if (activated->Shape().Size() == 0) return Status::OK();
    const auto hidden = onnxruntime::narrow<uint32_t>(params.hidden);
    const auto groups = onnxruntime::narrow<uint32_t>(streams->Shape().Size() / params.hidden);
    const auto prefix = onnxruntime::narrow<uint32_t>(split[0]);
    const auto& limits = context.DeviceLimits();
    ORT_RETURN_IF_NOT(limits.maxComputeWorkgroupStorageSize >= 1024 &&
                          hidden <= (limits.maxComputeWorkgroupStorageSize - 1024) / sizeof(MLFloat16),
                      "HCPostNormProjectionF16 normalization exceeds workgroup storage limit");
    const bool shared_scale = scale->Shape().Size() == params.hidden;
    const bool projection_fused = projection_mode == 1 &&
        context.AdapterInfo().vendor == std::string_view{"nvidia"} && streams->Shape().Size() == 10240 &&
        !context.EnableMatmulFp32Accumulation() &&
        ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_HC_EXACT_DENSE", 0) != 0;
    const bool fused = enabled == 1 && projection_fused && hidden == 2560 &&
        limits.maxComputeWorkgroupSizeX >= 256 && limits.maxComputeInvocationsPerWorkgroup >= 256 &&
        limits.maxComputeWorkgroupStorageSize >= 27648 &&
        limits.maxStorageBuffersPerShaderStage >= (shared_ ? 12u : 10u);
    if (fused) {
      return replay_.Run(context, {y, normalized, activated, tail}, [&]() -> Status {
        HCPostNormProjectionProgram program{shared_, shared_scale};
        program.CacheHint(shared_, shared_scale)
            .AddInput({streams, ProgramTensorMetadataDependency::Type})
            .AddInput({context.Input(1), ProgramTensorMetadataDependency::Type})
            .AddInput({context.Input(2), ProgramTensorMetadataDependency::Type})
            .AddInput({scale, ProgramTensorMetadataDependency::Type});
        if (shared_) {
          program.AddInput({context.Input(4), ProgramTensorMetadataDependency::Type})
              .AddInput({context.Input(5), ProgramTensorMetadataDependency::Type});
        }
        program.AddInput({weights, ProgramTensorMetadataDependency::TypeAndRank, 16})
            .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
            .AddOutput({y, ProgramTensorMetadataDependency::Type})
            .AddOutput({normalized, ProgramTensorMetadataDependency::Type})
            .AddOutput({activated, ProgramTensorMetadataDependency::Type})
            .AddOutput({tail, ProgramTensorMetadataDependency::Type})
            .AddUniformVariables({{hidden}, {epsilon_}, {prefix}, {alpha_}})
            .SetWorkgroupSize(256).SetDispatchGroupSize(41);
        return context.RunProgram(program);
      });
    }
    HCPostNormProjectionBaselineNormProgram norm{shared_, shared_scale, hidden};
    norm.CacheHint(shared_, shared_scale, hidden)
        .AddInput({streams, ProgramTensorMetadataDependency::Type})
        .AddInput({context.Input(1), ProgramTensorMetadataDependency::Type})
        .AddInput({context.Input(2), ProgramTensorMetadataDependency::Type})
        .AddInput({scale, ProgramTensorMetadataDependency::Type});
    if (shared_) {
      norm.AddInput({context.Input(4), ProgramTensorMetadataDependency::Type})
          .AddInput({context.Input(5), ProgramTensorMetadataDependency::Type});
    }
    norm.AddOutput({y, ProgramTensorMetadataDependency::Type})
        .AddOutput({normalized, ProgramTensorMetadataDependency::Type})
        .AddUniformVariables({{hidden}, {epsilon_}})
        .SetWorkgroupSize(256).SetDispatchGroupSize(groups);
    ORT_RETURN_IF_ERROR(context.RunProgram(norm));
    if (projection_fused) {
      HCPostNormProjectionBaselineMatMulProgram program;
      program.SetWorkgroupSize(64).SetDispatchGroupSize(162);
      program.AddInput({normalized, ProgramTensorMetadataDependency::TypeAndRank, 4})
          .AddInput({weights, ProgramTensorMetadataDependency::TypeAndRank, 16})
          .AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank})
          .AddOutput({activated, ProgramTensorMetadataDependency::TypeAndRank})
          .AddOutput({tail, ProgramTensorMetadataDependency::TypeAndRank})
          .AddUniformVariables({{prefix}, {alpha_}});
      return context.RunProgram(program);
    }
    auto intermediate = context.CreateGPUTensor(streams->DataType(), TensorShape(shape));
    ORT_RETURN_IF_ERROR(ApplyMatMulNBits(normalized, weights, scales, nullptr, nullptr, 10240, 324, 32,
                                        accuracy_, 4, context, &intermediate));
    SplitScaledSiLUF16Program program;
    const auto count = onnxruntime::narrow<uint32_t>(intermediate.Shape().Size());
    program.AddInput({&intermediate, ProgramTensorMetadataDependency::Type})
        .AddOutput({activated, ProgramTensorMetadataDependency::None})
        .AddOutput({tail, ProgramTensorMetadataDependency::None})
        .AddUniformVariables({{count}, {324u}, {prefix},
                              {onnxruntime::narrow<uint32_t>(split[1])}, {alpha_}})
        .SetDispatchGroupSize((count + 255) / 256).SetWorkgroupSize(256);
    return context.RunProgram(program);
  }

 protected:
  HCPostNormProjectionF16(const OpKernelInfo& info, bool shared)
      : WebGpuKernel(info), shared_{shared}, epsilon_{info.GetAttrOrDefault<float>("epsilon", 1e-6f)},
        alpha_{info.GetAttrOrDefault<float>("alpha", 1.0f)},
        accuracy_{info.GetAttrOrDefault<int64_t>("accuracy_level", 4)},
        replay_{shared ? "HCSharedPostNormProjectionF16" : "HCPostNormProjectionF16",
                shared ? std::initializer_list<int>{0, 1, 2, 3, 4, 5, 7, 8}
                       : std::initializer_list<int>{0, 1, 2, 3, 5, 6}} {
    ORT_ENFORCE(info.GetAttr<int64_t>("K") == 10240 && info.GetAttr<int64_t>("N") == 324 &&
                    info.GetAttr<int64_t>("block_size") == 32 && info.GetAttr<int64_t>("bits") == 4,
                "HCPostNormProjectionF16 requires K10240/N324 symmetric block32 INT4");
  }

 private:
  bool shared_;
  float epsilon_;
  float alpha_;
  int64_t accuracy_;
  mutable StaticNodeReplay replay_;
};

class HCSharedPostNormProjectionF16 final : public HCPostNormProjectionF16 {
 public:
  explicit HCSharedPostNormProjectionF16(const OpKernelInfo& info) : HCPostNormProjectionF16(info, true) {}
};

ONNX_OPERATOR_KERNEL_EX(
    HCPostNormProjectionF16, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).InputMemoryType(OrtMemTypeCPUInput, 4), HCPostNormProjectionF16);
ONNX_OPERATOR_KERNEL_EX(
    HCSharedPostNormProjectionF16, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).InputMemoryType(OrtMemTypeCPUInput, 6), HCSharedPostNormProjectionF16);

}
