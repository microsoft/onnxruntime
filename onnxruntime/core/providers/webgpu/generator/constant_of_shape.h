// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace webgpu {

class ConstantOfShapeProgram final : public Program<ConstantOfShapeProgram> {
 public:
  ConstantOfShapeProgram() : Program{"ConstantOfShape"} {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"value", ProgramUniformVariableDataType::Float32});
};

class ConstantOfShape final : public WebGpuKernel {
 public:
  ConstantOfShape(const OpKernelInfo& info) : WebGpuKernel(info) {
    ONNX_NAMESPACE::TensorProto t_proto;
    if (info.GetAttr<ONNX_NAMESPACE::TensorProto>("value", &t_proto).IsOK()) {
      has_value_ = true;
      tensor_type_ = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(t_proto.data_type());
      const void* raw_data = utils::HasRawData(t_proto) ? t_proto.raw_data().data() : nullptr;
      const size_t raw_data_len = utils::HasRawData(t_proto) ? t_proto.raw_data().size() : 0;
      ExtractValue(t_proto, raw_data, raw_data_len);
    }
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  void ExtractValue(const ONNX_NAMESPACE::TensorProto& t_proto,
                    const void* raw_data, size_t raw_data_len);

  bool has_value_ = false;
  ONNX_NAMESPACE::TensorProto_DataType tensor_type_ = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
  // Store the constant value as a float for the shader uniform.
  // Integer values are bit-cast to float for transfer via uniform buffer.
  float value_as_float_ = 0.0f;
};

// Factory functions for kernel creation with conditional int64 support
template <int StartVersion, int EndVersion>
KernelCreateInfo CreateConstantOfShapeVersionedKernelInfo(bool enable_int64);
template <int SinceVersion>
KernelCreateInfo CreateConstantOfShapeKernelInfo(bool enable_int64);

}  // namespace webgpu
}  // namespace onnxruntime
