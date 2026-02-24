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

  // For int64, we need to ensure values fit in int32 range since we use 4 bytes in uniforms
  uint32_t start_u32, delta_u32;
  if constexpr (std::is_same_v<T, int64_t>) {
    // Check if values fit in int32 range
    ORT_ENFORCE(start >= std::numeric_limits<int32_t>::min() && start <= std::numeric_limits<int32_t>::max(),
                "Range start value ", start, " is out of int32 range");
    ORT_ENFORCE(delta >= std::numeric_limits<int32_t>::min() && delta <= std::numeric_limits<int32_t>::max(),
                "Range delta value ", delta, " is out of int32 range");
    int32_t start_i32 = static_cast<int32_t>(start);
    int32_t delta_i32 = static_cast<int32_t>(delta);
    start_u32 = std::bit_cast<uint32_t>(start_i32);
    delta_u32 = std::bit_cast<uint32_t>(delta_i32);
  } else {
    start_u32 = std::bit_cast<uint32_t>(start);
    delta_u32 = std::bit_cast<uint32_t>(delta);
  }

  program.AddOutput({output_tensor, ProgramTensorMetadataDependency::Type})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({
          output_size,
          start_u32,
          delta_u32,
      });

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

// Explicit template instantiations (needed for linking)
template class Range<float>;
template class Range<int32_t>;
template class Range<int64_t>;

void RegisterRangeKernels(KernelRegistry& kernel_registry, bool enable_int64) {
  // Helper lambda to create kernel
  auto create_range_kernel_info = [](auto type_tag) {
    using T = decltype(type_tag);
    KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
      out = std::make_unique<Range<T>>(info);
      return Status::OK();
    };

    return KernelCreateInfo(
        KernelDefBuilder()
            .SetName("Range")
            .SetDomain(kOnnxDomain)
            .SinceVersion(11)
            .Provider(kWebGpuExecutionProvider)
            .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())
            .InputMemoryType(OrtMemTypeCPU, 0)
            .InputMemoryType(OrtMemTypeCPU, 1)
            .InputMemoryType(OrtMemTypeCPU, 2)
            .Build(),
        kernel_create_fn);
  };

  // Always register float and int32_t
  ORT_THROW_IF_ERROR(kernel_registry.Register(create_range_kernel_info(float{})));
  ORT_THROW_IF_ERROR(kernel_registry.Register(create_range_kernel_info(int32_t{})));

  // Register int64_t only if int64 support is enabled
  if (enable_int64) {
    ORT_THROW_IF_ERROR(kernel_registry.Register(create_range_kernel_info(int64_t{})));
  }
}

}  // namespace webgpu
}  // namespace onnxruntime
