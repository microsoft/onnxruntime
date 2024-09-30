// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class Transpose final : public WebGpuKernel, public TransposeBase {
 public:
  Transpose(const OpKernelInfo& info) : WebGpuKernel{info}, TransposeBase{info} {
  }
  Status ComputeInternal(ComputeContext& context) const override;
  constexpr static uint32_t TILE_SIZE = 16;
};

class TransposeProgram final : public Program<TransposeProgram> {
 public:
  TransposeProgram(const gsl::span<const size_t>& permutations, bool use_shared)
      : Program{"Transpose"}, perm_(permutations.begin(), permutations.end()), use_shared_(use_shared) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32});
  WEBGPU_PROGRAM_DEFINE_CONSTANTS({"tile_size", Transpose::TILE_SIZE});

 private:
  InlinedVector<int64_t> perm_;
  const bool use_shared_;
};

}  // namespace webgpu
}  // namespace onnxruntime
