// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

// Transpose OIHW Weight to OHWI
class OIHW2OHWIProgram final : public Program<OIHW2OHWIProgram> {
 public:
  OIHW2OHWIProgram() : Program("OIHW2OHWI") {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"O", ProgramUniformVariableDataType::Uint32},
      {"I", ProgramUniformVariableDataType::Uint32},
      {"H", ProgramUniformVariableDataType::Uint32},
      {"W", ProgramUniformVariableDataType::Uint32},
      {"Ci_tiles", ProgramUniformVariableDataType::Uint32},
      {"H_W_tiles", ProgramUniformVariableDataType::Uint32});
};

class Transpose final : public WebGpuKernel, public TransposeBase {
 public:
  Transpose(const OpKernelInfo& info) : WebGpuKernel{info}, TransposeBase{info} {
  }
  Status ComputeInternal(ComputeContext& context) const override;
  static Status DoTranspose(onnxruntime::webgpu::ComputeContextBase& context, gsl::span<const size_t> permutations, const Tensor& input, Tensor& output);

  // Tile edge, and how many of its rows one workgroup row covers. A 32-wide tile makes each
  // coalesced global access 64 bytes instead of 32, but a full 32x32 workgroup (1024 threads)
  // measured slower than 16x16; 32x8 with four rows per thread keeps the wider access and the
  // smaller workgroup.
  constexpr static uint32_t TILE_SIZE = 32;
  constexpr static uint32_t TILE_ROWS = 8;
};

class TransposeProgram final : public Program<TransposeProgram> {
 public:
  TransposeProgram(const gsl::span<const size_t>& permutations, bool use_shared)
      : Program{"Transpose"}, perm_(permutations.begin(), permutations.end()), use_shared_(use_shared) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32});
  WEBGPU_PROGRAM_DEFINE_CONSTANTS({"tile_size", Transpose::TILE_SIZE},
                                  {"tile_rows", Transpose::TILE_ROWS});

 private:
  InlinedVector<int64_t> perm_;
  const bool use_shared_;
};

}  // namespace webgpu
}  // namespace onnxruntime
