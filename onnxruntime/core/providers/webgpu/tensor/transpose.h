// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class TransposeProgram final : public Program<TransposeProgram> {
 public:
  TransposeProgram(const gsl::span<const size_t>& permutations, bool use_shared, const int tile_size)
      : Program{"Transpose"}, perm_(permutations.begin(), permutations.end()), use_shared_(use_shared), tile_size_(tile_size) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  InlinedVector<int64_t> perm_;
  const bool use_shared_;
  const uint32_t tile_size_;
};

class Transpose final : public WebGpuKernel, public TransposeBase {
 public:
  Transpose(const OpKernelInfo& info) : WebGpuKernel{info}, TransposeBase{info} {
  }

  Status ComputeInternal(ComputeContext& context) const override;
};

}  // namespace webgpu
}  // namespace onnxruntime
