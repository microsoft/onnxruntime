// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include <string>

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class SubgroupMatrixMatMulNBitsProgram final : public Program<SubgroupMatrixMatMulNBitsProgram> {
 public:
  SubgroupMatrixMatMulNBitsProgram(uint32_t nbits, int32_t config_index, const wgpu::StringView& vendor) : Program{"SubgroupMatrixMatMulNBits"}, nbits_(nbits), config_index_(config_index), vendor_(vendor) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t nbits_;
  int32_t config_index_;
  std::string vendor_;
};

Status ApplySubgroupMatrixMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales,
                                      uint32_t M,
                                      uint32_t N,
                                      uint32_t K,
                                      uint32_t nbits,
                                      int32_t config_index,
                                      onnxruntime::webgpu::ComputeContext& context,
                                      Tensor* y);

bool CanApplySubgroupMatrixMatMulNBits(onnxruntime::webgpu::ComputeContext& context,
                                       uint64_t accuracy_level,
                                       uint32_t block_size,
                                       uint32_t batch_count,
                                       uint32_t N,
                                       uint32_t K,
                                       bool has_zero_points,
                                       int32_t& config_index);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime

#endif
