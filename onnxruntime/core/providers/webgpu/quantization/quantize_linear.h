// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class DequantizeLinearProgram final : public Program<DequantizeLinearProgram> {
 public:
  DequantizeLinearProgram(const int64_t axis, const int64_t block_size,
                          const bool packed, const bool issigned, const bool per_layer_quantization,
                          const bool per_axis_quantization, const int components,
                          const int input_component, bool has_zeropoint) : Program<DequantizeLinearProgram>{"DequantizeLinear"},
                                                       axis_{axis},
                                                       block_size_{block_size},
                                                       packed_{packed},
                                                       signed_{issigned},
                                                       per_layer_quantization_{per_layer_quantization},
                                                       per_axis_quantization_{per_axis_quantization},
                                                       components_{components},
                                                       input_component_{input_component},
                                                       has_zeropoint_{has_zeropoint} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"axis", ProgramUniformVariableDataType::Int32},
                                          {"block_size", ProgramUniformVariableDataType::Int32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  int64_t axis_;
  int64_t block_size_;
  bool packed_;
  bool signed_;
  bool per_layer_quantization_;
  bool per_axis_quantization_;
  int components_;
  int input_component_;
  bool has_zeropoint_;
};

class DequantizeLinear final : public WebGpuKernel {
 public:
  DequantizeLinear(const OpKernelInfo& info) : WebGpuKernel(info) {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", 1);
    block_size_ = info.GetAttrOrDefault<int64_t>("block_size", 0);
    output_dtype_ = info.GetAttrOrDefault<int64_t>("output_dtype", 0);
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t axis_;
  int64_t block_size_;
  int64_t output_dtype_;
};

}  // namespace webgpu
}  // namespace onnxruntime
