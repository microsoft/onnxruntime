// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "core/framework/tensor.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/nn/conv.h"
#include "core/providers/webgpu/nn/fuse_utils.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

// Computes a channels-last (NHWC) 2D convolution as an implicit-GEMM using
// subgroupMatrixMultiplyAccumulate. The activation operand is the im2col of the
// input, gathered on the fly into workgroup memory (never materialized in global
// memory); the weight is pre-transposed to OHWI, i.e. B stored column-major.
// config_index selects the device subgroup-matrix config (into
// supported_subgroup_matrix_configs); sg_mat_count_m/n select how many subgroup
// matrices the output tile spans along M/N; split_k is the number of subgroups that
// cooperatively reduce K.
//
// The bias add and the fused activation live in a generated `write_output` helper
// rather than in the WGSL template: the activation body comes from
// GetActivationSnippet / GetActivationDeclaration, so every ActivationKind those
// support works here without the template knowing the kinds.
class SubgroupMatrixConvProgram final : public Program<SubgroupMatrixConvProgram> {
 public:
  SubgroupMatrixConvProgram(bool has_bias, int32_t config_index,
                            uint32_t sg_mat_count_m, uint32_t sg_mat_count_n, uint32_t split_k,
                            uint32_t vec_size, const Activation& activation)
      : Program{"SubgroupMatrixConv"},
        has_bias_(has_bias),
        config_index_(config_index),
        sg_mat_count_m_(sg_mat_count_m),
        sg_mat_count_n_(sg_mat_count_n),
        split_k_(split_k),
        vec_size_(vec_size),
        activation_(activation) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"num_n_tile", ProgramUniformVariableDataType::Uint32},
                                          // Number of batch slices, i.e. the logical z extent of the tile
                                          // grid. The kernel needs it to drop the extra workgroups a
                                          // normalized dispatch can add (see the guard in the template).
                                          {"batch", ProgramUniformVariableDataType::Uint32},
                                          {"src_h", ProgramUniformVariableDataType::Uint32},
                                          {"src_w", ProgramUniformVariableDataType::Uint32},
                                          {"channel_i", ProgramUniformVariableDataType::Uint32},
                                          {"kernel_h", ProgramUniformVariableDataType::Uint32},
                                          {"kernel_w", ProgramUniformVariableDataType::Uint32},
                                          {"output_w", ProgramUniformVariableDataType::Uint32},
                                          {"dilations", ProgramUniformVariableDataType::Uint32},
                                          {"pads", ProgramUniformVariableDataType::Uint32},
                                          {"strides", ProgramUniformVariableDataType::Uint32},
                                          // Activation uniforms must remain last because definitions and values
                                          // are matched by index (see AppendActivationUniformsData).
                                          WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES);

 private:
  const bool has_bias_;
  const int32_t config_index_;
  const uint32_t sg_mat_count_m_;
  const uint32_t sg_mat_count_n_;
  const uint32_t split_k_;
  const uint32_t vec_size_;
  // Held by value, like the other subgroup-matrix/MatMul programs (and unlike the
  // neighbouring Conv programs, which hold a reference into the kernel). The program
  // outlives the call that builds it -- CacheKey() and the uniform values are read
  // during Run -- and Activation is two floats plus an enum, so a copy is cheaper
  // than having to reason about the referent's lifetime.
  const Activation activation_;
};

// Returns true when the subgroup-matrix Conv fast path can run for this problem:
// fp16, channels-last, group == 1, a 4D non-1x1 kernel, and a device that reports
// the 8x16x16 F16 subgroup-matrix config, supports subgroup-size control (the kernel
// pins its 32 lanes per subgroup) and has a vendor tiling policy. The GEMM
// contraction dim K = kernel_h * kernel_w * Cin must be a multiple of the
// subgroup-matrix K (16), and N = Cout must be at least one subgroup-matrix N block
// (16) wide so that a trailing partial N tile can be shifted back far enough for its
// unchecked weight loads to stay in bounds. Above that, N needs no parity or
// block-multiple (the OHWI weight is column-major with stride K, so load alignment
// comes from K); M = out_h * out_w is unconstrained. Callers fall back to the default
// conv paths when this returns false.
//
// The fused activation is deliberately not part of the test: the write-out epilogue
// is generated from GetActivationSnippet, so every ActivationKind is handled and
// there is nothing to reject. Unlike CanApplyIm2ColMatMulProgram, this does not need
// an IsActivationSupported() companion.
//
// This is the shape/device-level eligibility test, decided from the weight shape
// alone: SubgroupMatrixConvImpl calls it from Compute, and Conv::PrePackInternal
// calls it to skip prepacking for a weight this path will consume in its original
// OIHW layout.
bool CanApplySubgroupMatrixConv(ComputeContextBase& context,
                                bool is_channels_last,
                                const TensorShape& kernel_shape,
                                uint32_t group,
                                MLDataType data_type);

// Creates a subgroup-matrix Conv implementation on devices whose vendor policy
// supports it; returns nullptr otherwise, so the caller falls back to the normal Conv
// paths. The impl reads the Conv attributes from `parent`, and the per-problem output
// tiling comes from a vendor-specific selector chosen internally from the device
// context.
template <bool is_channels_last, bool is_fused>
std::unique_ptr<typename Conv<is_channels_last, is_fused>::ConvOptImpl> CreateSubgroupMatrixConvImpl(
    const Conv<is_channels_last, is_fused>& parent,
    const ComputeContextBase& context);

}  // namespace webgpu
}  // namespace onnxruntime
