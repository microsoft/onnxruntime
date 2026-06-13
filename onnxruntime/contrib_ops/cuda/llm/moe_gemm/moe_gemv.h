// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Batched GEMV fast path for int4 weight-only per-channel MoE at small expanded
// row counts (e.g. batch-1 decode with top_k experts). Each expanded row is a
// single token-expert pair; one thread-block handles one row and offsets the
// weight/scale/bias pointers by that row's expert. Reuses the device-side math
// (layout, dequantize, mma, epilogue) from the dense fpA_intB_gemv kernel.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace onnxruntime::llm {
namespace kernels {
namespace moe_gemv {

// Returns true if the batched MoE GEMV fast path supports this problem shape.
// Requirements: int4 per-channel weights, half/bf16 activations, sm >= 80,
// small expanded_num_rows, and n divisible by the kernel tile width.
bool is_moe_gemv_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k);

// Launches the int4 per-channel MoE GEMV.
//   act:      [expanded_num_rows, k]  permuted activations (row-major)
//   weight:   [num_experts, k, n] packed int4 in Sm80 ColumnMajorInterleave layout (uint8)
//   scales:   [num_experts, n] per-channel scales (T)
//   bias:     [num_experts, n] per-expert bias (T) or nullptr
//   out:      [expanded_num_rows, n] (row-major)
//   expert_first_token_offset: [num_experts + 1] prefix offsets of permuted rows
// T is half or __nv_bfloat16.
template <typename T>
void launch_moe_gemv_int4_per_channel(
    T const* act, uint8_t const* weight, T const* scales, T const* bias, T* out,
    int64_t const* expert_first_token_offset, int num_experts, int64_t expanded_num_rows,
    int64_t n, int64_t k, int sm, cudaStream_t stream);

}  // namespace moe_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
