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

// Tiling/parallelization knob selected by the FP4 GEMV autotuner. Every config computes the same
// dot products with the same accumulation dtype, so the sweep only picks the fastest. CtaN is
// bit-exact (it only changes how many output columns a block owns). Threads is *not*: a block
// walks K in strides of StepK * Threads and the epilogue reduces across Threads/32 warps, so
// changing it changes the summation order and the low bits of the result can move.
enum class MoeGemvConfig {
  kDefault,
  kCtaN16,
  kThreads64,
};

// Cover Qwen-style top_k=8 MTP decode. An (N+1)-token verification for
// num_speculative_tokens=N expands to (N+1)*8 rows, up to 64 for N=7.
inline constexpr int64_t kMaxProfiledExpandedRowsFp4 = 64;

// True when the opt-in interleaved MXFP4 GEMV path is enabled (env ORT_FP4_GEMV_INTERLEAVED=1).
// It combines three changes over the default path: (a) the INT4-style ColumnMajorInterleaved FP4
// weight layout (kInterleave=4, kStepK=32) for 4x fewer K-trips, (b) dtype-conditional accumulation
// (fp32 for bf16) to keep bf16 accuracy across the longer K-chains, and (c) a smaller CtaN to
// recover the occupancy the interleave + fp32-accum cost. Default OFF; when off the shipping
// single-pass ColumnMajor path is byte-for-byte unchanged. Both PrePack (weight layout) and the
// compute dispatch query this so the prepacked weights and the kernel always agree.
bool Fp4MoeGemvUseInterleaved();

// Shape-derived default tiling for the non-interleaved ColumnMajor FP4 GEMV. Used whenever the
// runtime does not have a profiled result for the shape, which is the shipping default because
// ORT_FP4_GEMV_AUTOTUNE is off (it synchronizes the inference stream) and is skipped entirely
// during CUDA-graph capture. Only Threads is derived; CtaN stays at the default, so the result
// never changes which shapes is_moe_gemv_fp4_supported accepts. Note that Threads sets the K
// partition, so the choice moves the last bits of the output (see MoeGemvConfig above). Set
// ORT_FP4_GEMV_DEFAULT_TILING=0 to fall back to the fixed default tiling.
MoeGemvConfig Fp4MoeGemvDefaultConfig(int64_t expanded_num_rows, int64_t n, int64_t k,
                                      int multi_processor_count);

// FP4 GEMV shape support for the non-interleaved ColumnMajor layout (kInterleave = 1). Shared by
// both MXFP4 (group_size == 32) and NVFP4 (group_size == 16). Requires sm >= 80, n divisible by
// the kernel tile width (kCtaN) selected by `config`, and the profiled small-decode row/dim
// bounds. (The opt-in interleaved layout is MXFP4-only; see is_moe_gemv_fp4_supported in the .cu.)
// See launch_moe_gemv_fp4_symmetric.
bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size);
bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                               MoeGemvConfig config);
// `sm80_pair_interleaved` = the weights were pre-packed in the SM80 grouped-GEMM layout (the
// interleaved layout plus the [e0,e2,e4,e6,e1,e3,e5,e7] nibble pair-interleave) and the GEMV will
// un-permute it in-register. It implies the interleaved shape rules (MXFP4 group_size 32,
// n % 16 == 0, k % 64 == 0) regardless of ORT_FP4_GEMV_INTERLEAVED.
bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                               MoeGemvConfig config, bool sm80_pair_interleaved);

// Layout-only (row-count independent) form of the rules above, for PrePack: true when a GEMV can
// decode an [n, k] MXFP4 problem straight out of the SM80 grouped-GEMM pair-interleaved buffer, so
// no separate ColToRow decode copy of the expert weights has to be allocated.
bool is_moe_gemv_fp4_sm80_layout_supported(int64_t n, int64_t k, int group_size);

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
// When `sm80_pair_interleaved` is true, `weight` is instead the SM80 grouped-GEMM buffer produced
// by QMoE::PrePackRepackFP4Weights(gemv_interleaved=true, sm80_pair_interleave=true) and the
// kernel inverts the nibble pair-interleave while decoding, so prefill and decode share one copy.
template <typename T>
void launch_moe_gemv_fp4_symmetric(
    const T* act, const uint8_t* weight, const T* scales, const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts, int64_t expanded_num_rows,
    int64_t n, int64_t k, int group_size, int sm, MoeGemvConfig config, bool sm80_pair_interleaved,
    cudaStream_t stream);

// Launches the MXFP4 MoE GEMV and fuses interleaved SwiGLU activation.
//   weight/scales/bias use raw FC1 output width n = 2 * inter_size
//   out is post-activation [expanded_num_rows, inter_size]
template <typename T>
void launch_moe_gemv_fp4_symmetric_interleaved_swiglu(
    const T* act, const uint8_t* weight, const T* scales, const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts, int64_t expanded_num_rows,
    int64_t inter_size, int64_t k, int group_size, int sm, cutlass_kernels::ActivationParams activation_params,
    MoeGemvConfig config, bool sm80_pair_interleaved,
    const int* permuted_row_to_source_row, int64_t num_rows, cudaStream_t stream);

}  // namespace moe_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
