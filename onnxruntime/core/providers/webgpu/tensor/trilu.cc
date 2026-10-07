// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "core/providers/common.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/tensor/trilu.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    Trilu,
    kOnnxDomain,
    14,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes())
        .InputMemoryType(OrtMemTypeCPU, 1),
    Trilu);

Status TriluProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "let row = i32((global_idx / uniforms.matrix_w) % uniforms.matrix_h);\n"
                            << "let col = i32(global_idx % uniforms.matrix_w);\n"
                            << "let input_value = " << input.GetByOffset("global_idx") << ";\n";
  if (upper_) {
    shader.MainFunctionBody() << "let value = select(input_value_t(0), input_value, (row + uniforms.k) <= col);\n";
  } else {
    shader.MainFunctionBody() << "let value = select(input_value_t(0), input_value, (row + uniforms.k) >= col);\n";
  }
  shader.MainFunctionBody() << output.SetByOffset("global_idx", "value");
  return Status::OK();
}

static Status GetTriluK(const Tensor* k_tensor, int32_t& k) {
  if (k_tensor == nullptr) {
    k = 0;
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(k_tensor), "k should be a 1-D or 0-D tensor.");

  int64_t k_value = 0;
  if (k_tensor->IsDataType<int64_t>()) {
    k_value = *(k_tensor->Data<int64_t>());
  } else if (k_tensor->IsDataType<int32_t>()) {
    k_value = *(k_tensor->Data<int32_t>());
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "k should be of type int32 or int64.");
  }

  ORT_RETURN_IF_NOT(k_value >= std::numeric_limits<int32_t>::min() &&
                        k_value <= std::numeric_limits<int32_t>::max(),
                    "k is out of range for WebGPU Trilu.");
  k = static_cast<int32_t>(k_value);
  return Status::OK();
}

Status Trilu::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const auto* k_tensor = context.InputCount() > 1 ? context.Input(1) : nullptr;

  const TensorShape& input_shape = input_tensor->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() >= 2, "Input tensor should have a rank of at least 2.");

  auto* output_tensor = context.Output(0, input_shape);
  const int64_t output_size = output_tensor->Shape().Size();
  if (output_size == 0) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(output_size <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                    "Trilu output size exceeds WebGPU supported maximum.");

  const int64_t matrix_h = input_shape[input_shape.NumDimensions() - 2];
  const int64_t matrix_w = input_shape[input_shape.NumDimensions() - 1];
  ORT_RETURN_IF_NOT(matrix_h > 0 && matrix_h <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                    "Trilu matrix height is out of supported range.");
  ORT_RETURN_IF_NOT(matrix_w > 0 && matrix_w <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                    "Trilu matrix width is out of supported range.");

  int32_t k = 0;
  ORT_RETURN_IF_ERROR(GetTriluK(k_tensor, k));

  TriluProgram program{upper_};
  program
      .CacheHint(upper_)
      .AddInput({input_tensor, ProgramTensorMetadataDependency::Type})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::Type})
      .SetDispatchGroupSize((static_cast<uint32_t>(output_size) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{static_cast<uint32_t>(output_size)},
                            {static_cast<uint32_t>(matrix_h)},
                            {static_cast<uint32_t>(matrix_w)},
                            {k}});

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
