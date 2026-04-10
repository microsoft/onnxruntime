// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/quantization/quantization_utils.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class DequantizeLinearProgram final : public Program<DequantizeLinearProgram> {
 public:
  DequantizeLinearProgram(util::U32PackingMode packing_mode, bool is_packed_signed,
                          util::QuantizationType quantization_type, bool has_zeropoint, int rank = 0)
      : Program<DequantizeLinearProgram>{"DequantizeLinear"},
        packing_mode_{packing_mode},
        packed_signed_{is_packed_signed},
        quantization_type_{quantization_type},
        has_zeropoint_{has_zeropoint},
        rank_{rank} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"axis", ProgramUniformVariableDataType::Uint32},
                                          {"block_size", ProgramUniformVariableDataType::Uint32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  util::U32PackingMode packing_mode_;
  bool packed_signed_;
  util::QuantizationType quantization_type_;
  bool has_zeropoint_;
  int rank_;
};

class DequantizeLinear final : public WebGpuKernel {
 public:
  DequantizeLinear(const OpKernelInfo& info) : WebGpuKernel(info) {
    axis_ = info.GetAttrOrDefault<int64_t>("axis", 1);
    block_size_ = info.GetAttrOrDefault<int64_t>("block_size", 0);
    output_dtype_ = info.GetAttrOrDefault<int64_t>("output_dtype", 0);
    ORT_ENFORCE(block_size_ >= 0, "'block_size' must be non-negative.");
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t axis_;
  int64_t block_size_;
  int64_t output_dtype_;
};

}  // namespace webgpu
}  // namespace onnxruntime
