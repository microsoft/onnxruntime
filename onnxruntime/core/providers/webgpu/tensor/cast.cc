// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/providers/webgpu/tensor/cast.h"

#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

namespace {
const std::vector<MLDataType>& CastOpTypeConstraints(bool enable_graph_capture) {
  // Base types that are always supported - boolean, integer and float types that explicitly allowed in WGSL:
  // https://gpuweb.github.io/gpuweb/wgsl/#plain-types-section
  static std::vector<MLDataType> base_types{
      DataTypeImpl::GetTensorType<MLFloat16>(),
      DataTypeImpl::GetTensorType<float>(),
      DataTypeImpl::GetTensorType<int32_t>(),
      DataTypeImpl::GetTensorType<uint32_t>(),
      DataTypeImpl::GetTensorType<bool>()};

  if (enable_graph_capture) {
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

Status Cast::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  auto* output_tensor = context.Output(0, input_tensor->Shape());
  int64_t size = input_tensor->Shape().Size();
  if (size == 0) {
    return Status::OK();
  }
  bool is_from_int64 = input_tensor->DataType() == DataTypeImpl::GetType<int64_t>();
  const int in_components = is_from_int64 ? 1 : 4;
  const int out_components = to_ == ONNX_NAMESPACE::TensorProto_DataType_INT64 ? 1 : 4;
  uint32_t vec_size = onnxruntime::narrow<uint32_t>((size + 3) / 4);
  uint32_t in_vec_size = onnxruntime::narrow<uint32_t>(in_components == 1 ? size : vec_size);
  uint32_t out_vec_size = onnxruntime::narrow<uint32_t>(out_components == 1 ? size : vec_size);

  CastProgram program{to_, is_from_int64};
  program
      .AddInput({input_tensor, ProgramTensorMetadataDependency::Type, {in_vec_size}, in_components})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::None, {out_vec_size}, out_components})
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
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      expression = "int32(a)";
      break;
    default:
      ORT_NOT_IMPLEMENTED("Cast to type ", to_, " is not supported.");
  }

  sh.MainFunctionBody() << sh.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size");
  if (is_from_int64_) {
    sh.MainFunctionBody() << "  let a0 = " << input.GetByOffset("global_idx * 4") << ";\n"
                          << "  let a1 = " << input.GetByOffset("global_idx * 4 + 1") << ";\n"
                          << "  let a2 = " << input.GetByOffset("global_idx * 4 + 2") << ";\n"
                          << "  let a3 = " << input.GetByOffset("global_idx * 4 + 3") << ";\n"
                          << "  let a = vec4<i32>(a0, a1, a2, a3);\n";
  } else {
    sh.MainFunctionBody() << "  let a = " << input.GetByOffset("global_idx") << ";\n";
  }
  if (to_ == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    sh.MainFunctionBody() << output.SetByOffset("global_idx * 4", "a.x") << "\n"
                          << output.SetByOffset("global_idx * 4 + 1", "a.y") << "\n"
                          << output.SetByOffset("global_idx * 4 + 2", "a.z") << "\n"
                          << output.SetByOffset("global_idx * 4 + 3", "a.w") << "\n";
  } else {
    sh.MainFunctionBody() << output.SetByOffset("global_idx", expression);
  }

  return Status::OK();
}

template <int StartVersion, int EndVersion>
KernelCreateInfo CreateCastKernelInfo(bool enable_int64) {
  const auto& type_constraints = CastOpTypeConstraints(enable_int64);

  KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Cast>(info);
    return Status::OK();
  };

  if constexpr (StartVersion == EndVersion) {
    // Non-versioned kernel
    return {
        KernelDefBuilder()
            .SetName("Cast")
            .SetDomain(kOnnxDomain)
            .SinceVersion(StartVersion)
            .Provider(kWebGpuExecutionProvider)
            .TypeConstraint("T1", type_constraints)
            .TypeConstraint("T2", type_constraints)
            .Build(),
        kernel_create_fn};
  } else {
    // Versioned kernel
    return {
        KernelDefBuilder()
            .SetName("Cast")
            .SetDomain(kOnnxDomain)
            .SinceVersion(StartVersion, EndVersion)
            .Provider(kWebGpuExecutionProvider)
            .TypeConstraint("T1", type_constraints)
            .TypeConstraint("T2", type_constraints)
            .Build(),
        kernel_create_fn};
  }
}

// Explicit template instantiations
template KernelCreateInfo CreateCastKernelInfo<6, 8>(bool);
template KernelCreateInfo CreateCastKernelInfo<9, 12>(bool);
template KernelCreateInfo CreateCastKernelInfo<13, 18>(bool);
template KernelCreateInfo CreateCastKernelInfo<19, 20>(bool);
template KernelCreateInfo CreateCastKernelInfo<21, 22>(bool);
template KernelCreateInfo CreateCastKernelInfo<23>(bool);

}  // namespace webgpu
}  // namespace onnxruntime
