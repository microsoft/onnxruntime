// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

// TurboQuant Hadamard rotation program.
// Applies a normalized Hadamard transform to the last dimension of a BNSH tensor in-place.
// Uses the Fast Walsh-Hadamard Transform (FWHT) in O(n log n) via shared memory.
// The Hadamard matrix is symmetric and orthogonal (H = H^T, H @ H^T = I),
// so applying the same transform twice is the identity (self-inverse).
//
// Dispatched as: one workgroup per (batch * num_heads * sequence_length).
// Workgroup size = head_size (must be power of 2, typically 128).
class TurboQuantRotateProgram final : public Program<TurboQuantRotateProgram> {
 public:
  TurboQuantRotateProgram(uint32_t head_size, uint32_t num_heads)
      : Program{"TurboQuantRotate"},
        head_size_(head_size),
        num_heads_(num_heads) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"start_token", ProgramUniformVariableDataType::Uint32},
      {"n_reps", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t head_size_;
  uint32_t num_heads_;
};

// Apply Hadamard rotation to a BNSH tensor (present_key or present_value) in-place.
// Only rotates tokens in [start_token, start_token + num_tokens) range.
Status ApplyTurboQuantRotation(onnxruntime::webgpu::ComputeContext& context,
                               Tensor* tensor,
                               uint32_t head_size,
                               uint32_t num_heads,
                               uint32_t batch_size,
                               uint32_t num_tokens,
                               uint32_t present_sequence_length,
                               uint32_t start_token,
                               uint32_t n_reps);

// Apply inverse Hadamard rotation to attention output (BSNH format) in-place.
// Since Hadamard is self-inverse, this uses the same transform.
Status ApplyTurboQuantInverseRotation(onnxruntime::webgpu::ComputeContext& context,
                                      Tensor* output,
                                      uint32_t head_size,
                                      uint32_t num_heads,
                                      uint32_t batch_size,
                                      uint32_t sequence_length,
                                      uint32_t n_reps);

// TurboQuant fused rotate+quantize program.
// Applies FWHT rotation, computes L2 norm, normalizes to unit sphere,
// then maps each element to its nearest TurboQuant centroid index (0–15).
// Stores centroid indices in the first head_size-2 elements (as the native element type),
// and the f32 L2 norm bitcast into the last 2 element slots.
//
// Dispatched as: one workgroup per (batch * kv_num_heads * num_tokens).
// Workgroup size = head_size (must be power of 2, typically 128).
class TurboQuantRotateQuantizeProgram final : public Program<TurboQuantRotateQuantizeProgram> {
 public:
  TurboQuantRotateQuantizeProgram(uint32_t head_size, uint32_t num_heads, bool is_fp16, bool source_BSNH)
      : Program{"TurboQuantRotateQuantize"},
        head_size_(head_size),
        num_heads_(num_heads),
        is_fp16_(is_fp16),
        source_BSNH_(source_BSNH) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"kv_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"start_token", ProgramUniformVariableDataType::Uint32},
      {"n_reps", ProgramUniformVariableDataType::Uint32},
      {"compressed_dim", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t head_size_;
  uint32_t num_heads_;
  bool is_fp16_;
  bool source_BSNH_;
};

// Apply fused Hadamard rotation + quantization + int4 packing.
// Reads from source (fp16, BSNH or BNSH) and writes packed int4 to dest (BNSH compressed fp16).
// source: the original K or V input tensor (uncompressed).
// dest: the present_key or present_value output tensor (compressed layout).
Status ApplyTurboQuantRotateAndQuantize(onnxruntime::webgpu::ComputeContext& context,
                                        const Tensor* source,
                                        Tensor* dest,
                                        uint32_t head_size,
                                        uint32_t num_heads,
                                        uint32_t batch_size,
                                        uint32_t kv_sequence_length,
                                        uint32_t present_sequence_length,
                                        uint32_t start_token,
                                        uint32_t n_reps,
                                        uint32_t compressed_dim,
                                        bool is_fp16,
                                        bool source_BSNH);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
