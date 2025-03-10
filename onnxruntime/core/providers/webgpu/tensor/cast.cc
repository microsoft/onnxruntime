// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/providers/webgpu/tensor/cast.h"

#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

namespace {
const std::vector<MLDataType>& CastOpTypeConstraints() {
  // currently support boolean, integer and float types that explicitly allowed in WGSL:
  // https://gpuweb.github.io/gpuweb/wgsl/#plain-types-section
  //
  static std::vector<MLDataType> types{
      DataTypeImpl::GetTensorType<MLFloat16>(),
      DataTypeImpl::GetTensorType<float>(),
      DataTypeImpl::GetTensorType<int32_t>(),
      DataTypeImpl::GetTensorType<uint32_t>(),
      DataTypeImpl::GetTensorType<bool>()};
  return types;
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Cast,
    kOnnxDomain,
    6, 8,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", CastOpTypeConstraints())
        .TypeConstraint("T2", CastOpTypeConstraints()),
    Cast);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Cast,
    kOnnxDomain,
    9, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", CastOpTypeConstraints())
        .TypeConstraint("T2", CastOpTypeConstraints()),
    Cast);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Cast,
    kOnnxDomain,
    13, 18,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", CastOpTypeConstraints())
        .TypeConstraint("T2", CastOpTypeConstraints()),
    Cast);
ONNX_OPERATOR_KERNEL_EX(
    Cast,
    kOnnxDomain,
    19,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", CastOpTypeConstraints())
        .TypeConstraint("T2", CastOpTypeConstraints()),
    Cast);

Status Cast::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  auto* output_tensor = context.Output(0, input_tensor->Shape());
  int64_t size = input_tensor->Shape().Size();
  if (size == 0) {
    return Status::OK();
  }
  uint32_t vec_size = gsl::narrow<uint32_t>((size + 3) / 4);

  CastProgram program{to_};
  program
      .AddInput({input_tensor, ProgramTensorMetadataDependency::Type, {vec_size}, 4})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::None, {vec_size}, 4})
      .SetDispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({
          {static_cast<uint32_t>(vec_size)},
      })
      .CacheHint(std::to_string(to_));
  return context.RunProgram(program);
}

Status CastProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& input = sh.AddInput("x", ShaderUsage::UseUniform);
  const auto& output = sh.AddOutput("y", ShaderUsage::UseUniform);
  std::string expression;
  switch (to_) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      expression = "vec4<f16>(a)";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      expression = "vec4<f32>(a)";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      expression = "vec4<i32>(a)";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      expression = "vec4<u32>(a)";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      expression = "vec4<bool>(a)";
      break;
    default:
      ORT_NOT_IMPLEMENTED("Cast to type ", to_, " is not supported.");
  }
  sh.MainFunctionBody() << sh.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size")
                        << "  let a = " << input.GetByOffset("global_idx") << ";\n  "
                        << output.SetByOffset("global_idx", expression);

  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
