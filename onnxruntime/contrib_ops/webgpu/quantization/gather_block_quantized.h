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
  GatherBlockQuantizedProgram(const bool is_signed, const bool is_uint8, size_t indices_rank, int gather_axis, bool has_zeropoint,
                              TensorShape x_shape, TensorShape output_shape) : Program<GatherBlockQuantizedProgram>{"GatherBlockQuantized"},
                                                                               is_signed_{is_signed},
                                                                               is_uint8_{is_uint8},
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
  bool is_signed_;
  bool is_uint8_;
  size_t indices_rank_;
  int gather_axis_;
  bool has_zeropoint_;
  TensorShape x_shape_;
  TensorShape output_shape_;
};

class GatherBlockQuantized final : public WebGpuKernel {
 public:
  GatherBlockQuantized(const OpKernelInfo& info) : WebGpuKernel(info) {
    gather_axis_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("gather_axis", 0));
    block_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("block_size", 128));
    quantize_axis_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("quantize_axis", 1));
    bits_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("bits", 4));

    ORT_ENFORCE(bits_ == 4 || bits_ == 8, "'bits' must be 4 or 8.");
    ORT_ENFORCE(block_size_ >= 16 && ((block_size_ - 1) & block_size_) == 0,
                "'block_size' must be 2's power and not less than 16.");
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int gather_axis_;
  int quantize_axis_;
  int block_size_;
  int bits_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
