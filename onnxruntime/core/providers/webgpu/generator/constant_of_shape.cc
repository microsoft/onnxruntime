// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/generator/constant_of_shape.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

Status ConstantOfShapeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  let value_u32 = bitcast<u32>(uniforms.value);\n";

  auto var_type = Outputs()[0].var_type;
  if (var_type == ProgramVariableDataType::Float16) {
    // f16: value stored as f32 in the uniform; convert to f16 in shader
    shader.MainFunctionBody() << "  " << output.SetByOffset("global_idx", "output_value_t(uniforms.value)");
  } else if (var_type == ProgramVariableDataType::Int32x2) {
    // int64 is stored as vec2<u32> in WebGPU; put value in low word, sign-extend to high word
    shader.MainFunctionBody() << "  let sign_ext = select(0u, 0xFFFFFFFFu, (value_u32 & 0x80000000u) != 0u);\n"
                              << "  " << output.SetByOffset("global_idx", "output_value_t(value_u32, sign_ext)");
  } else {
    // f32, i32, u32: direct bitcast
    shader.MainFunctionBody() << "  " << output.SetByOffset("global_idx", "bitcast<output_value_t>(value_u32)");
  }
  return Status::OK();
}

void ConstantOfShape::ExtractValue(const ONNX_NAMESPACE::TensorProto& t_proto,
                                   const void* raw_data, size_t raw_data_len) {
  // Extract the single-element value and store as float (bit-cast for integers).
  // The shader will bitcast back to the appropriate type.
  switch (tensor_type_) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
      float val = 0.0f;
      ORT_THROW_IF_ERROR(utils::UnpackTensor(t_proto, raw_data, raw_data_len, &val, 1));
      value_as_float_ = val;
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
      MLFloat16 val;
      ORT_THROW_IF_ERROR(utils::UnpackTensor(t_proto, raw_data, raw_data_len, &val, 1));
      // Convert f16 to f32 for the uniform; shader will convert back to f16
      value_as_float_ = val.ToFloat();
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      int32_t val = 0;
      ORT_THROW_IF_ERROR(utils::UnpackTensor(t_proto, raw_data, raw_data_len, &val, 1));
      uint32_t uval = static_cast<uint32_t>(val);
      value_as_float_ = *reinterpret_cast<float*>(&uval);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
      int64_t val = 0;
      ORT_THROW_IF_ERROR(utils::UnpackTensor(t_proto, raw_data, raw_data_len, &val, 1));
      // For int64, store the lower 32 bits (most ConstantOfShape values are small)
      uint32_t uval = static_cast<uint32_t>(val);
      value_as_float_ = *reinterpret_cast<float*>(&uval);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL: {
      bool val = false;
      ORT_THROW_IF_ERROR(utils::UnpackTensor(t_proto, raw_data, raw_data_len, &val, 1));
      uint32_t uval = val ? 1u : 0u;
      value_as_float_ = *reinterpret_cast<float*>(&uval);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
      uint8_t val = 0;
      ORT_THROW_IF_ERROR(utils::UnpackTensor(t_proto, raw_data, raw_data_len, &val, 1));
      uint32_t uval = static_cast<uint32_t>(val);
      value_as_float_ = *reinterpret_cast<float*>(&uval);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32: {
      uint32_t val = 0;
      ORT_THROW_IF_ERROR(utils::UnpackTensor(t_proto, raw_data, raw_data_len, &val, 1));
      value_as_float_ = *reinterpret_cast<float*>(&val);
      break;
    }
    default:
      // For unsupported types, default to 0
      value_as_float_ = 0.0f;
      break;
  }
}

Status ConstantOfShape::ComputeInternal(ComputeContext& context) const {
  const auto* shape_tensor = context.Input(0);
  const auto shape_span = shape_tensor->DataAsSpan<int64_t>();
  TensorShape output_shape(shape_span);

  auto* output_tensor = context.Output(0, output_shape);
  int64_t output_size = output_tensor->Shape().Size();

  if (output_size == 0) {
    return Status::OK();
  }

  uint32_t data_size = onnxruntime::narrow<uint32_t>(output_size);

  ConstantOfShapeProgram program;
  program
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::Type, {data_size}, 1})
      .SetDispatchGroupSize((data_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{data_size}, {value_as_float_}});
  return context.RunProgram(program);
}

template <int StartVersion, int EndVersion>
KernelCreateInfo CreateConstantOfShapeVersionedKernelInfo(bool enable_int64) {
  const auto& type_constraints = GetOpTypeConstraints(enable_int64, false);

  KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<ConstantOfShape>(info);
    return Status::OK();
  };

  return {
      KernelDefBuilder()
          .SetName("ConstantOfShape")
          .SetDomain(kOnnxDomain)
          .SinceVersion(StartVersion, EndVersion)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T2", type_constraints)
          .InputMemoryType(OrtMemTypeCPU, 0)
          .Build(),
      kernel_create_fn};
}

template <int SinceVersion>
KernelCreateInfo CreateConstantOfShapeKernelInfo(bool enable_int64) {
  const auto& type_constraints = GetOpTypeConstraints(enable_int64, false);

  KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<ConstantOfShape>(info);
    return Status::OK();
  };

  return {
      KernelDefBuilder()
          .SetName("ConstantOfShape")
          .SetDomain(kOnnxDomain)
          .SinceVersion(SinceVersion)
          .Provider(kWebGpuExecutionProvider)
          .TypeConstraint("T2", type_constraints)
          .InputMemoryType(OrtMemTypeCPU, 0)
          .Build(),
      kernel_create_fn};
}

// Explicit template instantiations
template KernelCreateInfo CreateConstantOfShapeVersionedKernelInfo<9, 19>(bool);
template KernelCreateInfo CreateConstantOfShapeKernelInfo<20>(bool);

}  // namespace webgpu
}  // namespace onnxruntime
