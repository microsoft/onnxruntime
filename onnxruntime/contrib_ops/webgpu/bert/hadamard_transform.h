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
  HadamardTransformProgram(int slice_size_log2, int components)
      : Program{"HadamardTransform"},
        slice_size_log2_(slice_size_log2),
        components_(components) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"num_slices", ProgramUniformVariableDataType::Uint32});

 private:
  int slice_size_log2_;
  int components_;
};

// Apply the normalized Walsh-Hadamard transform.
// The normalized Hadamard matrix is symmetric (H == H^T) and orthogonal
// (H @ H^T = I), so applying it twice recovers the original data.
// This means the same function serves as both the forward and inverse transform.
//
// If explicit_slice_size > 0, it is used as the transform dimension size.
// Otherwise, the last dimension of the tensor shape is used.
// The transform size must be a power of 2 (>= 4).
// All elements are divided into slices of that size, each transformed independently.
Status ApplyHadamardTransform(onnxruntime::webgpu::ComputeContext& context,
                              const Tensor* input,
                              Tensor* output,
                              int explicit_slice_size = 0);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
