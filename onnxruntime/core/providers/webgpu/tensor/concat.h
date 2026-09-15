// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/cpu/tensor/concatbase.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class ConcatProgram final : public Program<ConcatProgram> {
 public:
  ConcatProgram(size_t axis, bool is_int64) : Program{"Concat"}, axis_{axis}, is_int64_{is_int64} {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"offsets", ProgramUniformVariableDataType::Uint32},
                                          {"sizes_in_concat_axis", ProgramUniformVariableDataType::Uint32},
                                          {"output_size", ProgramUniformVariableDataType::Uint32});

 private:
  size_t axis_;
  bool is_int64_;
};

class Concat final : public WebGpuKernel, public ConcatBase {
 public:
  Concat(const OpKernelInfo& info) : WebGpuKernel(info), ConcatBase(info) {
  }

  Status ComputeInternal(ComputeContext& context) const override;
};

// Create Concat kernel info with appropriate type constraints based on int64 support
KernelCreateInfo CreateConcatVersionedKernelInfo(int start_version, int end_version, bool enable_int64);
KernelCreateInfo CreateConcatKernelInfo(int since_version, bool enable_int64);

}  // namespace webgpu
}  // namespace onnxruntime
