// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// MXFP4 (e2m1) batched GEMV fast path for symmetric weight-only MoE at small expanded
// row counts (e.g. batch-1 decode with top_k experts). Companion to the INT path declared
// in moe_gemv.h; both share the device-side machinery in moe_gemv_device.cuh.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "contrib_ops/cuda/llm/moe_gemm/common.h"

namespace onnxruntime::llm {
namespace kernels {
namespace moe_gemv {

// Tiling/parallelization knob selected by the FP4 GEMV autotuner. CtaN/Threads are pure
// tiling knobs (numerically bit-exact), so the sweep only picks the fastest. kSplitK2 is an
// INT-only split-K config that the FP4 launchers do not use; it is kept on the shared enum
// surface so the config-name helper stays exhaustive.
enum class MoeGemvConfig {
  kDefault,
  kCtaN16,
  kThreads64,
  kSplitK2,
};

// MXFP4 GEMV shape support for the non-interleaved ColumnMajor layout (kInterleave = 1).
// Requires sm >= 80, group_size == 32, n divisible by the kernel tile width (kCtaN) selected
// by `config`, and the profiled small-decode row/dim bounds. See launch_moe_gemv_fp4_symmetric.
bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size);
bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                               MoeGemvConfig config);

// Launches the MXFP4 (e2m1) MoE GEMV in the non-interleaved ColumnMajor layout.
//   act:      [expanded_num_rows, k]  permuted activations (row-major), T = half/bf16
//   weight:   [num_experts, n, k/2]  e2m1 codes packed two per byte (even-K low nibble)
//             == LaunchQMoERepackFP4ColToRow output
//   scales:   [num_experts, k/32, n]  TypeA block scales already folded with the per-expert
//             global scale == LaunchQMoECombineFp4ScalesForGemv output
//   bias:     [num_experts, n] (T) or nullptr
//   out:      [expanded_num_rows, n] (row-major)
// group_size is the MXFP4 block size (32).
template <typename T>
void launch_moe_gemv_fp4_symmetric(
    T const* act, uint8_t const* weight, T const* scales, T const* bias, T* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts, int64_t expanded_num_rows,
    int64_t n, int64_t k, int group_size, int sm, MoeGemvConfig config, cudaStream_t stream);

// Launches the MXFP4 MoE GEMV and fuses interleaved SwiGLU activation.
//   weight/scales/bias use raw FC1 output width n = 2 * inter_size
//   out is post-activation [expanded_num_rows, inter_size]
template <typename T>
void launch_moe_gemv_fp4_symmetric_interleaved_swiglu(
    T const* act, uint8_t const* weight, T const* scales, T const* bias, T* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts, int64_t expanded_num_rows,
    int64_t inter_size, int64_t k, int group_size, int sm, cutlass_kernels::ActivationParams activation_params,
    MoeGemvConfig config, cudaStream_t stream);

}  // namespace moe_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
