#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace onnxruntime::llm::kernels::deep_gemm_sm90 {

constexpr int kNumExperts = 32;
constexpr int kPaddedTokensPerExpert = 64;
constexpr int kMaxTokensPerExpert = 8;
constexpr int kHiddenSize = 4096;
constexpr int kInterSize = 2048;
constexpr int kFc1OutputSize = 4096;

// The packed FC1 output and FC2 output do not overlap. FC1 input aliases FC2
// output because FC1 has completed before FC2 starts.
size_t GetWorkspaceSize();

// Pack compact expert-major rows into DeepGEMM's [G, 64, K] layout and produce
// int32 row counts from the runner's int64 prefix offsets.
void PackInput(const __nv_bfloat16* compact_input, const int64_t* expert_first_token_offset,
               __nv_bfloat16* packed_input, int* masked_m, cudaStream_t stream);

// Apply DSV4's interleaved SwiGLU ordering to [G, 64, 4096] FC1 output and
// produce [G, 64, 2048] FC2 input.
void ApplyInterleavedSwiGLU(const __nv_bfloat16* fc1_output, __nv_bfloat16* fc2_input,
                           float alpha, float beta, float limit, cudaStream_t stream);

// Unpack [G, 64, 4096] output to the compact expert-major row layout consumed
// by ORT's existing finalize routing kernel.
void UnpackOutput(const __nv_bfloat16* packed_output, const int64_t* expert_first_token_offset,
                  __nv_bfloat16* compact_output, cudaStream_t stream);

void LaunchFc1(const __nv_bfloat16* packed_input, const __nv_bfloat16* weights,
               __nv_bfloat16* packed_output, int* masked_m, cudaStream_t stream);
void LaunchFc2(const __nv_bfloat16* packed_input, const __nv_bfloat16* weights,
               __nv_bfloat16* packed_output, int* masked_m, cudaStream_t stream);

void Run(const __nv_bfloat16* compact_input, const int64_t* expert_first_token_offset,
         const __nv_bfloat16* fc1_weights, const __nv_bfloat16* fc2_weights,
         __nv_bfloat16* compact_output, float alpha, float beta, float limit,
         void* workspace, cudaStream_t stream);

}  // namespace onnxruntime::llm::kernels::deep_gemm_sm90