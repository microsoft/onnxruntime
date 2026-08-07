#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

namespace onnxruntime::llm::kernels::deep_gemm_sm90 {

constexpr int kNumExperts = 32;
constexpr int kPaddedTokensPerExpert = 64;
constexpr int kMaxTokensPerExpert = 8;
constexpr int kHiddenSize = 4096;
constexpr int kInterSize = 2048;
constexpr int kFc1OutputSize = 4096;

// The FP8 1D2D kernel scales A per (row, 128 contiguous K) and B per [128 N, 128 K] block.
constexpr int kQuantBlockSize = 128;

// fp32 scale-factor count for a [num_experts, n, k] e4m3 weight tensor, in the K-major
// [num_experts, n / 128, k / 128] SFB layout the kernel indexes.
constexpr size_t WeightScaleCount(int num_experts, int n, int k) {
  return static_cast<size_t>(num_experts) * (n / kQuantBlockSize) * (k / kQuantBlockSize);
}

size_t GetWorkspaceSize();

// Quantizes the activations to e4m3 on the fly, runs both grouped GEMMs against the
// prepacked e4m3 expert weights, and writes bf16 rows back into the compact expert-major
// layout ORT's finalize routing kernel consumes. fc1_weight_scales / fc2_weight_scales are
// the fp32 per-[128 N, 128 K] block scales sized by WeightScaleCount().
void Run(const __nv_bfloat16* compact_input, const int64_t* expert_first_token_offset,
         const __nv_fp8_e4m3* fc1_weights, const float* fc1_weight_scales,
         const __nv_fp8_e4m3* fc2_weights, const float* fc2_weight_scales,
         __nv_bfloat16* compact_output, float alpha, float beta, float limit,
         void* workspace, cudaStream_t stream);

}  // namespace onnxruntime::llm::kernels::deep_gemm_sm90