// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

class GatherBlockQuantizedProgram final : public Program<GatherBlockQuantizedProgram> {
 public:
  GatherBlockQuantizedProgram(const bool issigned, int indices_rank, int gather_axis, bool has_zeropoint,
                              TensorShape x_shape, TensorShape output_shape) : Program<GatherBlockQuantizedProgram>{"GatherBlockQuantized"},
                                                                               signed_{issigned},
                                                                               indices_rank_{indices_rank},
                                                                               gather_axis_{gather_axis},
                                                                               has_zeropoint_{has_zeropoint},
                                                                               x_shape_{x_shape},
                                                                               output_shape_{output_shape} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"quantize_axis", ProgramUniformVariableDataType::Uint32},
                                          {"gather_axis", ProgramUniformVariableDataType::Uint32},
                                          {"block_size", ProgramUniformVariableDataType::Uint32});

 private:
  bool signed_;
  int indices_rank_;
  int gather_axis_;
  bool has_zeropoint_;
  TensorShape x_shape_;
  TensorShape output_shape_;
};

class GatherBlockQuantized final : public WebGpuKernel {
 public:
  GatherBlockQuantized(const OpKernelInfo& info) : WebGpuKernel(info) {
    gather_axis_ = info.GetAttrOrDefault<int64_t>("gather_axis", 0);
    block_size_ = info.GetAttrOrDefault<int64_t>("block_size", 128);
    quantize_axis_ = info.GetAttrOrDefault<int64_t>("quantize_axis", 1);
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t gather_axis_;
  int64_t quantize_axis_;
  int64_t block_size_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
