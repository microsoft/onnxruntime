// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/generator/range.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

Status Range::ComputeInternal(ComputeContext& context) const {
  const auto* start_tensor = context.Input<Tensor>(0);
  const auto* limit_tensor = context.Input<Tensor>(1);
  const auto* delta_tensor = context.Input<Tensor>(2);

  auto data_type = start_tensor->GetElementType();

  int64_t n = 0;
  uint32_t start_bits = 0;
  uint32_t delta_bits = 0;

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    float start = start_tensor->Data<float>()[0];
    float limit = limit_tensor->Data<float>()[0];
    float delta = delta_tensor->Data<float>()[0];
    GSL_SUPPRESS(io.2)
    n = static_cast<int64_t>(ceil((1.0 * (limit - start)) / delta));
    start_bits = *reinterpret_cast<const uint32_t*>(&start);
    delta_bits = *reinterpret_cast<const uint32_t*>(&delta);
  } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    int32_t start = start_tensor->Data<int32_t>()[0];
    int32_t limit = limit_tensor->Data<int32_t>()[0];
    int32_t delta = delta_tensor->Data<int32_t>()[0];
    GSL_SUPPRESS(io.2)
    n = static_cast<int64_t>(ceil((1.0 * (limit - start)) / delta));
    start_bits = *reinterpret_cast<const uint32_t*>(&start);
    delta_bits = *reinterpret_cast<const uint32_t*>(&delta);
  } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    int64_t start = start_tensor->Data<int64_t>()[0];
    int64_t limit = limit_tensor->Data<int64_t>()[0];
    int64_t delta = delta_tensor->Data<int64_t>()[0];
    GSL_SUPPRESS(io.2)
    n = static_cast<int64_t>(ceil((1.0 * (limit - start)) / delta));
    int32_t start_i32 = static_cast<int32_t>(start);
    int32_t delta_i32 = static_cast<int32_t>(delta);
    start_bits = *reinterpret_cast<const uint32_t*>(&start_i32);
    delta_bits = *reinterpret_cast<const uint32_t*>(&delta_i32);
  }

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

  if (n <= 0) {
    n = 0;
  }
  auto* output_tensor = context.Output(0, TensorShape{n});
  if (n == 0) {
    return Status::OK();
  }

  uint32_t output_size = onnxruntime::narrow<uint32_t>(n);
  RangeProgram program{data_type};

  program.AddOutput({output_tensor, ProgramTensorMetadataDependency::Type})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({
          output_size,
          start_bits,
          delta_bits,
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

namespace {
const std::vector<MLDataType>& RangeOpTypeConstraints(bool enable_graph_capture) {
  // Base types that are always supported
  static std::vector<MLDataType> base_types{
      DataTypeImpl::GetTensorType<float>(),
      DataTypeImpl::GetTensorType<int32_t>()};

  if (enable_graph_capture) {
    printf("Range: Returning types_with_int64\n");
    static std::vector<MLDataType> types_with_int64 = []() {
      auto types = base_types;
      types.push_back(DataTypeImpl::GetTensorType<int64_t>());
      return types;
    }();
    return types_with_int64;
  } else {
    return base_types;
  }
}
}  // namespace

KernelCreateInfo CreateRangeKernelInfo(bool enable_graph_capture) {
  const auto& type_constraints = RangeOpTypeConstraints(enable_graph_capture);

  KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Range>(info);
    return Status::OK();
  };

  return {
      KernelDefBuilder()
          .SetName("Range")
          .SetDomain(kOnnxDomain)
          .SinceVersion(11)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T", type_constraints)
          .InputMemoryType(OrtMemTypeCPU, 0)
          .InputMemoryType(OrtMemTypeCPU, 1)
          .InputMemoryType(OrtMemTypeCPU, 2)
          .Build(),
      kernel_create_fn};
}

}  // namespace webgpu
}  // namespace onnxruntime
