// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/generator/range.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

template <typename T>
Status Range<T>::ComputeInternal(ComputeContext& context) const {
  T start = context.Input<Tensor>(0)->Data<T>()[0];
  T limit = context.Input<Tensor>(1)->Data<T>()[0];
  T delta = context.Input<Tensor>(2)->Data<T>()[0];

  GSL_SUPPRESS(io.2)  // Ignore warning about potential overflow in (limit - start)
  int64_t n = static_cast<int64_t>(ceil((1.0 * (limit - start)) / delta));
  if (n <= 0) {
    n = 0;
  }
  auto* output_tensor = context.Output(0, TensorShape{n});
  if (n == 0) {
    return Status::OK();
  }

  uint32_t output_size = onnxruntime::narrow<uint32_t>(n);
  RangeProgram program{output_tensor->GetElementType()};
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

  program.AddOutput({output_tensor, ProgramTensorMetadataDependency::Type})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({
          output_size,
          *reinterpret_cast<uint32_t*>(&start),
          *reinterpret_cast<uint32_t*>(&delta),
      });

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

  return context.RunProgram(program);
}

Status RangeProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& output = sh.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  sh.MainFunctionBody() << sh.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size");

  // For int64, we need to cast to i32 first, then assign to output (which handles vec2<u32> conversion)
  // For int32 and float, we can use output_value_t directly
  if (data_type_ == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    // int64 case: bitcast to i32, compute with i32, then assign (automatic conversion to vec2<u32>)
    sh.MainFunctionBody() << "  let value = bitcast<i32>(uniforms.start) + i32(global_idx) * bitcast<i32>(uniforms.delta);\n"
                          << output.SetByOffset("global_idx", "value");
  } else {
    // float or int32 case: use output_value_t
    sh.MainFunctionBody() << "  let value = bitcast<output_value_t>(uniforms.start) + output_value_t(global_idx) * bitcast<output_value_t>(uniforms.delta);\n"
                          << output.SetByOffset("global_idx", "value");
  }

  return Status();
}

#define WEBGPU_RANGE_KERNEL(TYPE)                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                    \
      Range,                                                        \
      kOnnxDomain,                                                  \
      11,                                                           \
      TYPE,                                                         \
      kWebGpuExecutionProvider,                                     \
      KernelDefBuilder()                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()) \
          .InputMemoryType(OrtMemTypeCPU, 0)                        \
          .InputMemoryType(OrtMemTypeCPU, 1)                        \
          .InputMemoryType(OrtMemTypeCPU, 2),                       \
      Range<TYPE>);

WEBGPU_RANGE_KERNEL(float)
WEBGPU_RANGE_KERNEL(int32_t)
WEBGPU_RANGE_KERNEL(int64_t)

}  // namespace webgpu
}  // namespace onnxruntime
