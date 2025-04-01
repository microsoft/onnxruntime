// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class DequantizeLinearProgram final : public Program<DequantizeLinearProgram> {
 public:
  DequantizeLinearProgram(const int64_t axis, const int64_t block_size,
                          const bool packed, const bool issigned, const bool per_layer,
                          const bool per_axis, const int components,
                          const int input_component, bool has_zeropoint) : Program<DequantizeLinearProgram>{"DequantizeLinear"},
                                                       axis_{axis},
                                                       block_size_{block_size},
                                                       packed_{packed},
                                                       signed_{issigned},
                                                       per_layer_{per_layer},
                                                       per_axis_{per_axis},
                                                       components_{components},
                                                       input_component_{input_component},
                                                       has_zeropoint_{has_zeropoint} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"axis", ProgramUniformVariableDataType::Int32},
                                          {"block_size", ProgramUniformVariableDataType::Uint32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  int64_t axis_;
  int64_t block_size_;
  bool packed_;
  bool signed_;
  bool per_layer_;
  bool per_axis_;
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
