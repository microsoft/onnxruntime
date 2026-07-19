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
// tiling knobs (numerically bit-exact), so the sweep only picks the fastest.
enum class MoeGemvConfig {
  kDefault,
  kCtaN16,
  kThreads64,
};

// True when the opt-in interleaved MXFP4 GEMV path is enabled (env ORT_FP4_GEMV_INTERLEAVED=1).
// It combines three changes over the default path: (a) the INT4-style ColumnMajorInterleaved FP4
// weight layout (kInterleave=4, kStepK=32) for 4x fewer K-trips, (b) dtype-conditional accumulation
// (fp32 for bf16) to keep bf16 accuracy across the longer K-chains, and (c) a smaller CtaN to
// recover the occupancy the interleave + fp32-accum cost. Default OFF; when off the shipping
// single-pass ColumnMajor path is byte-for-byte unchanged. Both PrePack (weight layout) and the
// compute dispatch query this so the prepacked weights and the kernel always agree.
bool Fp4MoeGemvUseInterleaved();

// FP4 GEMV shape support for the non-interleaved ColumnMajor layout (kInterleave = 1). Shared by
// both MXFP4 (group_size == 32) and NVFP4 (group_size == 16). Requires sm >= 80, n divisible by
// the kernel tile width (kCtaN) selected by `config`, and the profiled small-decode row/dim
// bounds. (The opt-in interleaved layout is MXFP4-only; see is_moe_gemv_fp4_supported in the .cu.)
// See launch_moe_gemv_fp4_symmetric.
bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size);
bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                               MoeGemvConfig config);

// Launches the MXFP4 (e2m1) MoE GEMV in the non-interleaved ColumnMajor layout.
//   act:      [expanded_num_rows, k]  permuted activations (row-major), T = half/bf16
//   weight:   [num_experts, n, k/2]  e2m1 codes packed two per byte (even-K low nibble)
//             == LaunchQMoERepackFP4ColToRow output
//   scales:   [num_experts, k/group_size, n]  TypeA block scales already folded with the per-expert
//             global scale == LaunchQMoECombineFp4ScalesForGemv (MXFP4, group_size 32) or
//             LaunchQMoECombineNvfp4ScalesForGemv (NVFP4, group_size 16) output
//   bias:     [num_experts, n] (T) or nullptr
//   out:      [expanded_num_rows, n] (row-major)
// group_size is the FP4 block size (32 for MXFP4, 16 for NVFP4).
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
