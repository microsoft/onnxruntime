// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

// How the quantized input is packed into u32 words.
enum class PackingMode {
  None,     // no packing (e.g. int32)
  Packed8,  // 8-bit: 4 elements per u32, uses unpack4x[I/U]8
  Packed4,  // 4-bit: 8 elements per u32, manual bit extraction
};

class DequantizeLinearProgram final : public Program<DequantizeLinearProgram> {
 public:
  DequantizeLinearProgram(PackingMode packing, bool is_packed_signed, bool per_layer,
                          bool per_axis, bool has_zeropoint, int rank = 0)
      : Program<DequantizeLinearProgram>{"DequantizeLinear"},
        packing_{packing},
        packed_signed_{is_packed_signed},
        per_layer_{per_layer},
        per_axis_{per_axis},
        has_zeropoint_{has_zeropoint},
        rank_{rank} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"axis", ProgramUniformVariableDataType::Uint32},
                                          {"block_size", ProgramUniformVariableDataType::Uint32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  PackingMode packing_;
  bool packed_signed_;
  bool per_layer_;
  bool per_axis_;
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
