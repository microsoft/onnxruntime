// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Batched GEMV fast path for symmetric int weight-only MoE at small expanded row
// counts (e.g. batch-1 decode with top_k experts). Each expanded row is a single
// token-expert pair; one thread-block handles one row and offsets the
// weight/scale/bias pointers by that row's expert. Reuses the device-side math
// (layout, dequantize, mma, epilogue) from the dense fpA_intB_gemv kernel.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "contrib_ops/cuda/llm/moe_gemm/common.h"

namespace onnxruntime::llm {
namespace kernels {
namespace moe_gemv {

inline constexpr int64_t kMaxProfiledExpandedRows = 8;
inline constexpr int64_t kMaxProfiledExpandedRowsForSmallProblemDim = 4;
inline constexpr int64_t kMinProfiledProblemDim = 512;
// Lowered from 704 to 512 so block-wise decode shapes (e.g. Qwen top_k=8,
// inter_size=512) take the GEMV path. This also covers per-column INT4 shapes
// with inter_size in [512, 704); both bands are gated by ORT_DISABLE_MOE_GEMV.
inline constexpr int64_t kMinProfiledProblemDimForExpandedRowsAbove4 = 512;

// Returns true if the batched MoE GEMV fast path supports this problem shape.
// Requirements: FP16/BF16 activations, sm >= 80, small expanded_num_rows, supported
// INT weight type, supported group size, and n divisible by the kernel tile width.
bool is_moe_gemv_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k,
                           int weight_bits, int group_size);

// Backward-compatible per-channel INT4 shape check.
bool is_moe_gemv_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k);

// Launches symmetric INT MoE GEMV. group_size <= 0 means per-channel scales;
// group_size 32/64/128 means block-wise scales laid out as [num_experts, k_blocks, n].
// T is half or __nv_bfloat16. WeightType is cutlass::uint4b_t or uint8_t.
template <typename T, typename WeightType>
void launch_moe_gemv_int_symmetric(
    T const* act, WeightType const* weight, T const* scales, T const* bias, T* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts, int64_t expanded_num_rows,
    int64_t n, int64_t k, int group_size, int sm, cudaStream_t stream);

// Launches symmetric INT MoE GEMV and fuses interleaved SwiGLU activation.
// weight/bias use raw FC1 output width n = 2 * inter_size. Scales are
// [num_experts, n] for group_size <= 0 and [num_experts, k_blocks, n] for
// block-wise group_size 32/64/128.
template <typename T, typename WeightType>
void launch_moe_gemv_int_symmetric_interleaved_swiglu(
    T const* act, WeightType const* weight, T const* scales, T const* bias, T* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts, int64_t expanded_num_rows,
    int64_t inter_size, int64_t k, int group_size, int sm, cutlass_kernels::ActivationParams activation_params,
    float* splitk_partials,
    cudaStream_t stream);

// Launches the int4 per-channel MoE GEMV.
//   act:      [expanded_num_rows, k]  permuted activations (row-major)
//   weight:   [num_experts, k, n] packed int4 in Sm80 ColumnMajorInterleave layout (uint8)
//   scales:   [num_experts, n] per-channel scales (T)
//   bias:     [num_experts, n] per-expert bias (T) or nullptr
//   out:      [expanded_num_rows, n] (row-major)
//   expert_first_token_offset: [num_experts + 1] prefix offsets of permuted rows
//   permuted_row_to_expert: [expanded_num_rows] local expert id for each permuted row, or nullptr to scan offsets
// T is half or __nv_bfloat16.
template <typename T>
void launch_moe_gemv_int4_per_channel(
    T const* act, uint8_t const* weight, T const* scales, T const* bias, T* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts, int64_t expanded_num_rows,
    int64_t n, int64_t k, int sm, cudaStream_t stream);

// Launches the int4 per-channel MoE GEMV and fuses interleaved SwiGLU activation.
//   weight/scales/bias use raw FC1 output width [num_experts, k, 2 * inter_size]
//   out is post-activation [expanded_num_rows, inter_size]
// Only interleaved SwiGLU layout (`swiglu_fusion == 1`) is supported.
template <typename T>
void launch_moe_gemv_int4_per_channel_interleaved_swiglu(
    T const* act, uint8_t const* weight, T const* scales, T const* bias, T* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts, int64_t expanded_num_rows,
    int64_t inter_size, int64_t k, int sm, cutlass_kernels::ActivationParams activation_params,
    cudaStream_t stream);

}  // namespace moe_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
