// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/scaled_silu.h"

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime::contrib::webgpu {

namespace {
constexpr const char* kStableSigmoid =
    "fn stable_sigmoid(v: f32) -> f32 {\n"
    "  if (v >= 0.0) { return 1.0 / (1.0 + exp(-v)); }\n"
    "  let e = exp(v); return e / (1.0 + e);\n"
    "}\n";
}

ONNX_OPERATOR_KERNEL_EX(
    ScaledSiLU, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("M", WebGpuSupportedFloatTypes()),
    ScaledSiLU);

ONNX_OPERATOR_KERNEL_EX(
    SplitScaledSiLUF16, kMSDomain, 1, kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Tin0", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("tensor(int64)in1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("Tout0", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("Tout1", DataTypeImpl::GetTensorType<MLFloat16>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    SplitScaledSiLUF16);

Status ScaledSiLUProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* scale = nullptr;
  if (has_scale_) scale = &shader.AddInput("scale", ShaderUsage::UseUniform);
  const auto& y = shader.AddOutput("y", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  shader.AdditionalImplementation() << kStableSigmoid;
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.count")
      << "  var scale_value = uniforms.alpha;\n";
  if (has_scale_) {
    shader.MainFunctionBody() << "  scale_value = f32(" << scale->GetByOffset("0u") << ");\n";
  }
  shader.MainFunctionBody()
      << "  let z_t = y_element_t(f32(" << x.GetByOffset("global_idx")
      << ") * scale_value);\n"
         "  let sigmoid_t = y_element_t(stable_sigmoid(f32(z_t)));\n"
      << "  " << y.SetByOffset("global_idx", "y_element_t(f32(z_t) * f32(sigmoid_t))") << "\n";
  return Status::OK();
}

ScaledSiLU::ScaledSiLU(const OpKernelInfo& info)
    : WebGpuKernel(info), alpha_(info.GetAttrOrDefault<float>("alpha", 1.0f)) {}

Status ScaledSiLU::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* scale = context.Input(1);
  ORT_RETURN_IF_NOT(scale == nullptr || scale->Shape().NumDimensions() == 0, "scale must be a scalar");
  auto* y = context.Output(0, x->Shape());
  const uint32_t count = onnxruntime::narrow<uint32_t>(x->Shape().Size());
  if (count == 0) return Status::OK();
  ScaledSiLUProgram program{scale != nullptr};
  program.CacheHint(scale != nullptr).AddInput({x, ProgramTensorMetadataDependency::Type});
  if (scale != nullptr) program.AddInput({scale, ProgramTensorMetadataDependency::Type});
  program.AddOutput({y, ProgramTensorMetadataDependency::None})
      .AddUniformVariables({{count}, {alpha_}})
      .SetDispatchGroupSize((count + 255) / 256)
      .SetWorkgroupSize(256);
  return context.RunProgram(program);
}

Status SplitScaledSiLUF16Program::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform);
  const auto& activated = shader.AddOutput("activated", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& tail = shader.AddOutput("tail", ShaderUsage::UseUniform);
  shader.AdditionalImplementation() << kStableSigmoid;
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.count")
      << "  let row = global_idx / uniforms.width;\n"
         "  let col = global_idx % uniforms.width;\n"
         "  if (col < uniforms.prefix) {\n"
      << "    let z_t = activated_element_t(f32(" << x.GetByOffset("global_idx")
      << ") * uniforms.alpha);\n"
         "    let sigmoid_t = activated_element_t(stable_sigmoid(f32(z_t)));\n"
      << "    " << activated.SetByOffset("row * uniforms.prefix + col",
                                        "activated_element_t(f32(z_t) * f32(sigmoid_t))")
      << "\n  } else {\n"
      << "    " << tail.SetByOffset("row * uniforms.suffix + col - uniforms.prefix",
                                   x.GetByOffset("global_idx"))
      << "\n  }\n";
  return Status::OK();
}

Status SplitScaledSiLUF16::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto* sizes = context.Input(1);
  ORT_RETURN_IF_NOT(x->Shape().NumDimensions() > 0 && sizes->Shape().NumDimensions() == 1 &&
                        sizes->Shape().Size() == 2,
                    "SplitScaledSiLUF16 requires a ranked tensor and exactly two split sizes");
  const auto* split = sizes->Data<int64_t>();
  const int64_t width = x->Shape().GetDims().back();
  ORT_RETURN_IF_NOT(split[0] > 0 && split[0] < width && split[1] == width - split[0],
                    "SplitScaledSiLUF16 requires two positive sizes summing to the last dimension");
  TensorShapeVector prefix_shape(x->Shape().GetDims().begin(), x->Shape().GetDims().end());
  TensorShapeVector suffix_shape = prefix_shape;
  prefix_shape.back() = split[0];
  suffix_shape.back() = split[1];
  auto* activated = context.Output(0, TensorShape(prefix_shape));
  auto* tail = context.Output(1, TensorShape(suffix_shape));
  const uint32_t count = onnxruntime::narrow<uint32_t>(x->Shape().Size());
  if (count == 0) return Status::OK();
  return static_replay_.Run(context, {activated, tail}, [&]() -> Status {
  SplitScaledSiLUF16Program program;
  program.AddInput({x, ProgramTensorMetadataDependency::Type})
      .AddOutput({activated, ProgramTensorMetadataDependency::None})
      .AddOutput({tail, ProgramTensorMetadataDependency::None})
      .AddUniformVariables({{count}, {onnxruntime::narrow<uint32_t>(width)},
                            {onnxruntime::narrow<uint32_t>(split[0])},
                            {onnxruntime::narrow<uint32_t>(split[1])}, {alpha_}})
      .SetDispatchGroupSize((count + 255) / 256)
      .SetWorkgroupSize(256);
  return context.RunProgram(program);
  });
}

}  // namespace onnxruntime::contrib::webgpu
