// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class HadamardTransformProgram final : public Program<HadamardTransformProgram> {
 public:
  HadamardTransformProgram(int head_size_log2, int components)
      : Program{"HadamardTransform"},
        head_size_log2_(head_size_log2),
        components_(components) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"seq_length", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32});

 private:
  int head_size_log2_;
  int components_;
};

// Apply the normalized Walsh-Hadamard transform in-place along the head dimension.
// The transform is orthogonal (H * H = I with normalization), so applying it
// twice recovers the original data. This means the same function serves as
// both the forward and inverse transform.
//
// Data layout: contiguous slices of [head_size] elements, with
// batch_size * seq_length * num_heads total slices.
// head_size must be a power of 2.
Status ApplyHadamardTransform(onnxruntime::webgpu::ComputeContext& context,
                              const Tensor* input,
                              Tensor* output,
                              int batch_size,
                              int seq_length,
                              int num_heads,
                              int head_size);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
