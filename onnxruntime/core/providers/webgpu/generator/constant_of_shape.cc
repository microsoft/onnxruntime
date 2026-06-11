// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/generator/constant_of_shape.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

Status ConstantOfShapeProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& output = sh.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  sh.MainFunctionBody() << sh.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                        << "  " << output.SetByOffset("global_idx", "bitcast<output_value_t>(uniforms.value)");

  return Status::OK();
}

Status ConstantOfShape::ComputeInternal(ComputeContext& context) const {
  Tensor* output_tensor = nullptr;
  ORT_RETURN_IF_ERROR(PrepareCompute(&context, &output_tensor));

  const auto output_size = output_tensor->Shape().Size();
  if (output_size == 0) {
    return Status::OK();
  }

  uint32_t value_u32 = 0;
  const void* value_ptr = GetValuePtr();
  if (value_ptr != nullptr) {
    std::memcpy(&value_u32, value_ptr, std::min(sizeof(uint32_t), static_cast<size_t>(output_tensor->DataType()->Size())));
  }

  ConstantOfShapeProgram program;
  program.AddOutput({output_tensor, ProgramTensorMetadataDependency::Type})
      .SetDispatchGroupSize((static_cast<uint32_t>(output_size) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({
          static_cast<uint32_t>(output_size),
          value_u32,
      });

  return context.RunProgram(program);
}

namespace {

std::vector<MLDataType> GetConstantOfShapeTypeConstraints(bool enable_int64) {
  auto types = std::vector<MLDataType>{
      DataTypeImpl::GetTensorType<float>(),
      DataTypeImpl::GetTensorType<MLFloat16>(),
      DataTypeImpl::GetTensorType<int32_t>(),
      DataTypeImpl::GetTensorType<uint8_t>(),
      DataTypeImpl::GetTensorType<int8_t>(),
      DataTypeImpl::GetTensorType<bool>(),
  };
  if (enable_int64) {
    types.push_back(DataTypeImpl::GetTensorType<int64_t>());
  }
  return types;
}

KernelCreatePtrFn GetConstantOfShapeKernelCreateFn() {
  return [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<ConstantOfShape>(info);
    return Status::OK();
  };
}

template <int StartVersion, int EndVersion>
KernelCreateInfo CreateConstantOfShapeVersionedKernelInfo(bool enable_int64) {
  return {
      KernelDefBuilder()
          .SetName("ConstantOfShape")
          .SetDomain(kOnnxDomain)
          .SinceVersion(StartVersion, EndVersion)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
          .TypeConstraint("T2", GetConstantOfShapeTypeConstraints(enable_int64))
          .InputMemoryType(OrtMemTypeCPU, 0)
          .Build(),
      GetConstantOfShapeKernelCreateFn()};
}

template <int SinceVersion>
KernelCreateInfo CreateConstantOfShapeKernelInfo(bool enable_int64) {
  return {
      KernelDefBuilder()
          .SetName("ConstantOfShape")
          .SetDomain(kOnnxDomain)
          .SinceVersion(SinceVersion)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
          .TypeConstraint("T2", GetConstantOfShapeTypeConstraints(enable_int64))
          .InputMemoryType(OrtMemTypeCPU, 0)
          .Build(),
      GetConstantOfShapeKernelCreateFn()};
}

}  // namespace

void RegisterConstantOfShapeKernels(KernelRegistry& kernel_registry, bool enable_int64) {
  ORT_THROW_IF_ERROR(kernel_registry.Register(CreateConstantOfShapeVersionedKernelInfo<9, 19>(enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry.Register(CreateConstantOfShapeVersionedKernelInfo<20, 20>(enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry.Register(CreateConstantOfShapeVersionedKernelInfo<21, 22>(enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry.Register(CreateConstantOfShapeVersionedKernelInfo<23, 23>(enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry.Register(CreateConstantOfShapeKernelInfo<24>(enable_int64)));
}

}  // namespace webgpu
}  // namespace onnxruntime
