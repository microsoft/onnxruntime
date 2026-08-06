#include "contrib_ops/cuda/llm/moe_gemm/deep_gemm_sm90.h"

#include <cmath>

#include <cuda.h>
#include <deep_gemm/impls/sm90_bf16_gemm.cuh>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::llm::kernels::deep_gemm_sm90 {
namespace {

constexpr int kThreads = 256;
constexpr int kNumSms = 132;
constexpr int kDynamicSmemBytes = 196864;
constexpr int kClusterSize = 2;
constexpr int kBlockM = 64;
// TMA flattens [expert, row], so each expert stride must end on a store-tile boundary.
static_assert(kPaddedTokensPerExpert % kBlockM == 0);

using Fc1Kernel = decltype(&deep_gemm::sm90_bf16_gemm_impl<
                           cute::UMMA::Major::K, cute::UMMA::Major::K,
                           0, kFc1OutputSize, kHiddenSize, kNumExperts,
                           kBlockM, 256, 64, 128, 128, 128, 4, 128, 128, 2, false,
                           kNumSms, deep_gemm::GemmType::MGroupedMasked, false, cutlass::bfloat16_t>);
using Fc2Kernel = decltype(&deep_gemm::sm90_bf16_gemm_impl<
                           cute::UMMA::Major::K, cute::UMMA::Major::K,
                           0, kHiddenSize, kInterSize, kNumExperts,
                           kBlockM, 256, 64, 128, 128, 128, 4, 128, 128, 2, false,
                           kNumSms, deep_gemm::GemmType::MGroupedMasked, false, cutlass::bfloat16_t>);

template <int K>
__global__ void PackInputKernel(const __nv_bfloat16* input, const int64_t* offsets,
                                __nv_bfloat16* output, int* masked_m) {
  const int expert = blockIdx.y;
  const int count = static_cast<int>(offsets[expert + 1] - offsets[expert]);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    masked_m[expert] = count;
  }
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kPaddedTokensPerExpert * K; index += blockDim.x * gridDim.x) {
    const int row = index / K;
    const int col = index % K;
    output[(expert * kPaddedTokensPerExpert + row) * K + col] =
        row < count ? input[(offsets[expert] + row) * K + col] : __float2bfloat16(0.0f);
  }
}

__global__ void InterleavedSwiGLUKernel(const __nv_bfloat16* input, __nv_bfloat16* output,
                                        float alpha, float beta, float limit) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kNumExperts * kPaddedTokensPerExpert * kInterSize;
  if (index >= total) return;
  const int row = index / kInterSize;
  const int col = index % kInterSize;
  float gate = __bfloat162float(input[row * kFc1OutputSize + 2 * col]);
  float linear = __bfloat162float(input[row * kFc1OutputSize + 2 * col + 1]);
  if (isfinite(limit)) {
    gate = fminf(gate, limit);
    linear = fminf(fmaxf(linear, -limit), limit);
  }
  linear += beta;
  output[index] = __float2bfloat16(gate * (1.0f / (1.0f + expf(-alpha * gate))) * linear);
}

template <int N>
__global__ void UnpackOutputKernel(const __nv_bfloat16* input, const int64_t* offsets,
                                   __nv_bfloat16* output) {
  const int expert = blockIdx.y;
  const int count = static_cast<int>(offsets[expert + 1] - offsets[expert]);
  const int total = count * N;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total; index += blockDim.x * gridDim.x) {
    const int row = index / N;
    const int col = index % N;
    output[(offsets[expert] + row) * N + col] =
        input[(expert * kPaddedTokensPerExpert + row) * N + col];
  }
}

CUtensorMap MakeTensorMap(void* pointer, int inner, int outer, int box_inner, int box_outer) {
  CUtensorMap map{};
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(inner), static_cast<cuuint64_t>(outer)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(inner * sizeof(__nv_bfloat16))};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(box_inner), static_cast<cuuint32_t>(box_outer)};
  const cuuint32_t elem_strides[2] = {1, 1};
  CUresult result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, pointer, dims, strides, box, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  ORT_ENFORCE(result == CUDA_SUCCESS, "DeepGEMM cuTensorMapEncodeTiled failed: ", static_cast<int>(result));
  return map;
}

template <int N, int K, typename Kernel>
void Launch(Kernel kernel, const __nv_bfloat16* input, const __nv_bfloat16* weights,
            __nv_bfloat16* output, int* masked_m, cudaStream_t stream) {
  auto map_a = MakeTensorMap(const_cast<__nv_bfloat16*>(input), K,
                             kNumExperts * kPaddedTokensPerExpert, 64, kBlockM);
  auto map_b = MakeTensorMap(const_cast<__nv_bfloat16*>(weights), K, kNumExperts * N, 64, 256);
  auto map_d = MakeTensorMap(output, N, kNumExperts * kPaddedTokensPerExpert, 64, kBlockM);

  CUDA_CALL_THROW(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmemBytes));
  cudaLaunchAttribute attributes[2]{};
  attributes[0].id = cudaLaunchAttributeClusterDimension;
  attributes[0].val.clusterDim = {kClusterSize, 1, 1};
  attributes[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[1].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kNumSms, 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = kDynamicSmemBytes;
  config.stream = stream;
  config.attrs = attributes;
  config.numAttrs = 2;
  CUDA_CALL_THROW(cudaLaunchKernelEx(&config, kernel, masked_m, kPaddedTokensPerExpert, N, K,
                                     map_a, map_b, map_d));
}

}  // namespace

size_t GetWorkspaceSize() {
  constexpr size_t fc1_input_or_fc2_output =
      static_cast<size_t>(kNumExperts) * kPaddedTokensPerExpert * kHiddenSize * sizeof(__nv_bfloat16);
  constexpr size_t fc1_output =
      static_cast<size_t>(kNumExperts) * kPaddedTokensPerExpert * kFc1OutputSize * sizeof(__nv_bfloat16);
  constexpr size_t fc2_input =
      static_cast<size_t>(kNumExperts) * kPaddedTokensPerExpert * kInterSize * sizeof(__nv_bfloat16);
  constexpr size_t counts = kNumExperts * sizeof(int);
  return fc1_input_or_fc2_output + fc1_output + fc2_input + counts;
}

void PackInput(const __nv_bfloat16* compact_input, const int64_t* offsets,
               __nv_bfloat16* packed_input, int* masked_m, cudaStream_t stream) {
  PackInputKernel<kHiddenSize><<<dim3(32, kNumExperts), kThreads, 0, stream>>>(
      compact_input, offsets, packed_input, masked_m);
}

void ApplyInterleavedSwiGLU(const __nv_bfloat16* input, __nv_bfloat16* output,
                            float alpha, float beta, float limit, cudaStream_t stream) {
  const int total = kNumExperts * kPaddedTokensPerExpert * kInterSize;
  InterleavedSwiGLUKernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      input, output, alpha, beta, limit);
}

void UnpackOutput(const __nv_bfloat16* input, const int64_t* offsets,
                  __nv_bfloat16* output, cudaStream_t stream) {
  UnpackOutputKernel<kHiddenSize><<<dim3(32, kNumExperts), kThreads, 0, stream>>>(input, offsets, output);
}

void LaunchFc1(const __nv_bfloat16* input, const __nv_bfloat16* weights,
               __nv_bfloat16* output, int* masked_m, cudaStream_t stream) {
  auto kernel = &deep_gemm::sm90_bf16_gemm_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K, 0, kFc1OutputSize, kHiddenSize, kNumExperts,
      kBlockM, 256, 64, 128, 128, 128, 4, 128, 128, 2, false, kNumSms,
      deep_gemm::GemmType::MGroupedMasked, false, cutlass::bfloat16_t>;
  Launch<kFc1OutputSize, kHiddenSize>(kernel, input, weights, output, masked_m, stream);
}

void LaunchFc2(const __nv_bfloat16* input, const __nv_bfloat16* weights,
               __nv_bfloat16* output, int* masked_m, cudaStream_t stream) {
  auto kernel = &deep_gemm::sm90_bf16_gemm_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K, 0, kHiddenSize, kInterSize, kNumExperts,
      kBlockM, 256, 64, 128, 128, 128, 4, 128, 128, 2, false, kNumSms,
      deep_gemm::GemmType::MGroupedMasked, false, cutlass::bfloat16_t>;
  Launch<kHiddenSize, kInterSize>(kernel, input, weights, output, masked_m, stream);
}

void Run(const __nv_bfloat16* compact_input, const int64_t* offsets,
         const __nv_bfloat16* fc1_weights, const __nv_bfloat16* fc2_weights,
         __nv_bfloat16* compact_output, float alpha, float beta, float limit,
         void* workspace, cudaStream_t stream) {
  constexpr size_t packed_hidden_elements =
      static_cast<size_t>(kNumExperts) * kPaddedTokensPerExpert * kHiddenSize;
  constexpr size_t packed_inter_elements =
      static_cast<size_t>(kNumExperts) * kPaddedTokensPerExpert * kInterSize;

  auto* packed_io = static_cast<__nv_bfloat16*>(workspace);
  auto* fc1_output = packed_io + packed_hidden_elements;
  auto* fc2_input = fc1_output + packed_hidden_elements;
  auto* masked_m = reinterpret_cast<int*>(fc2_input + packed_inter_elements);

  PackInput(compact_input, offsets, packed_io, masked_m, stream);
  LaunchFc1(packed_io, fc1_weights, fc1_output, masked_m, stream);
  ApplyInterleavedSwiGLU(fc1_output, fc2_input, alpha, beta, limit, stream);
  LaunchFc2(fc2_input, fc2_weights, packed_io, masked_m, stream);
  UnpackOutput(packed_io, offsets, compact_output, stream);
}

}  // namespace onnxruntime::llm::kernels::deep_gemm_sm90