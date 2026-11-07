// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if USE_MEMORY_EFFICIENT_ATTENTION

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "contrib_ops/cuda/bert/causal_right_attention_impl.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/kernel_forward.h"
#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {
namespace {

constexpr int kHeadSize = 256;

struct WorkspaceLayout {
  size_t bias_bytes;
  size_t output_bytes;
  size_t total_bytes;
};

size_t Align(size_t bytes) {
  return (SafeInt<size_t>(bytes) + 255) / 256 * 256;
}

WorkspaceLayout GetWorkspaceLayout(int batch, int heads, int queries, int keys) {
  const int pitch = (keys + 31) / 32 * 32;
  const size_t bias_bytes = Align(SafeInt<size_t>(queries) * pitch * sizeof(uint16_t));
  const size_t elements = SafeInt<size_t>(batch) * heads * queries * kHeadSize;
  const size_t output_bytes = Align(SafeInt<size_t>(elements) * sizeof(uint16_t));
  return {bias_bytes, output_bytes,
          SafeInt<size_t>(bias_bytes) + output_bytes + SafeInt<size_t>(elements) * sizeof(float)};
}

template <typename T>
using CutlassType = std::conditional_t<std::is_same_v<T, half>, cutlass::half_t, cutlass::bfloat16_t>;

template <typename T>
using Kernel = AttentionKernel<CutlassType<T>, cutlass::arch::Sm80, true, 32, 128, 256, false, true>;

template <typename T>
__global__ void CausalBias(T* bias, int queries, int keys, int pitch) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < queries * pitch) {
    const int row = i / pitch;
    const int col = i % pitch;
    bias[i] = T(col < keys && col <= keys - queries + row ? 0.0f : -INFINITY);
  }
}

template <typename T>
__global__ void ToBHSD(const T* input, T* output, int batch, int heads, int queries) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch * heads * queries * kHeadSize) {
    const int x = i / kHeadSize;
    const int col = i % kHeadSize;
    const int row = x % queries;
    const int head = (x / queries) % heads;
    const int batch_id = x / (queries * heads);
    output[i] = input[((batch_id * queries + row) * heads + head) * kHeadSize + col];
  }
}

}  // namespace

size_t CausalRightAttentionWorkspaceSize(int batch, int heads, int queries, int keys) {
  return GetWorkspaceLayout(batch, heads, queries, keys).total_bytes;
}

template <typename T>
void ConfigureCausalRightAttention() {
  CUDA_CALL_THROW(cudaFuncSetAttribute(attention_kernel_batched_impl<Kernel<T>>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       sizeof(typename Kernel<T>::SharedStorage)));
}

template <typename T>
Status LaunchCausalRightAttention(cudaStream_t stream, const T* query, const T* key, const T* value,
                                  T* output, void* workspace, int batch, int heads, int queries, int keys,
                                  float scale) {
  using Scalar = CutlassType<T>;
  using AK = Kernel<T>;
  static_assert(AK::kNeedsOutputAccumulatorBuffer);
  const auto layout = GetWorkspaceLayout(batch, heads, queries, keys);
  const int pitch = (keys + 31) / 32 * 32;
  auto* bias = static_cast<Scalar*>(workspace);
  auto* intermediate = reinterpret_cast<Scalar*>(static_cast<char*>(workspace) + layout.bias_bytes);
  auto* accum = reinterpret_cast<float*>(static_cast<char*>(workspace) + layout.bias_bytes + layout.output_bytes);

  CausalBias<<<(queries * pitch + 255) / 256, 256, 0, stream>>>(bias, queries, keys, pitch);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  typename AK::Params p;
  p.query_ptr = const_cast<Scalar*>(reinterpret_cast<const Scalar*>(query));
  p.key_ptr = const_cast<Scalar*>(reinterpret_cast<const Scalar*>(key));
  p.value_ptr = const_cast<Scalar*>(reinterpret_cast<const Scalar*>(value));
  p.attn_bias_ptr = bias;
  p.output_ptr = intermediate;
  p.output_accum_ptr = accum;
  p.head_dim = p.head_dim_value = kHeadSize;
  p.num_batches = batch;
  p.num_heads = heads;
  p.num_queries = queries;
  p.num_keys = keys;
  p.scale = scale;
  p.q_strideM = p.k_strideM = p.v_strideM = kHeadSize;
  p.q_strideH = queries * kHeadSize;
  p.k_strideH = p.v_strideH = keys * kHeadSize;
  p.q_strideB = int64_t{heads} * queries * kHeadSize;
  p.k_strideB = p.v_strideB = int64_t{heads} * keys * kHeadSize;
  p.o_strideM = heads * kHeadSize;
  // Align physical bias rows without padding the logical key sequence.
  p.bias_strideM = pitch;
  p.bias_strideB = p.bias_strideH = 0;
  ORT_RETURN_IF_NOT(AK::check_supported(p), "Unsupported CausalRightAttention CUTLASS parameters");
  // Retain the bias path: a causal specialization can change the reduction/rounding schedule.
  attention_kernel_batched_impl<AK><<<p.getBlocksGrid(), p.getThreadsGrid(),
                                      sizeof(typename AK::SharedStorage), stream>>>(p);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  const int count = batch * heads * queries * kHeadSize;
  ToBHSD<<<(count + 255) / 256, 256, 0, stream>>>(
      intermediate, reinterpret_cast<Scalar*>(output), batch, heads, queries);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_CAUSAL_RIGHT_ATTENTION(T)                                                          \
  template void ConfigureCausalRightAttention<T>();                                                    \
  template Status LaunchCausalRightAttention<T>(cudaStream_t, const T*, const T*, const T*, T*, void*, \
                                                int, int, int, int, float);

INSTANTIATE_CAUSAL_RIGHT_ATTENTION(half)
INSTANTIATE_CAUSAL_RIGHT_ATTENTION(__nv_bfloat16)

#undef INSTANTIATE_CAUSAL_RIGHT_ATTENTION

}  // namespace onnxruntime::contrib::cuda

#endif  // USE_MEMORY_EFFICIENT_ATTENTION
