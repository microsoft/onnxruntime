// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "xqa_loader.h"
#include "utils.h"
#include <cassert>
#include <cuda_bf16.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Forward declarations of instantiated kernels from H128 and H64 namespaces
namespace H128 {
template <typename T>
Status LaunchXQAKernelImpl(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* past_seq_lens,
    const float* kv_cache_scale,
    const XqaQuantType kv_quant_type,
    void* workspace,
    size_t workspace_size);

}  // namespace H128

namespace H64 {
template <typename T>
Status LaunchXQAKernelImpl(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* past_seq_lens,
    const float* kv_cache_scale,
    const XqaQuantType kv_quant_type,
    void* workspace,
    size_t workspace_size);

}  // namespace H64

namespace H256 {
template <typename T>
Status LaunchXQAKernelImpl(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* past_seq_lens,
    const float* kv_cache_scale,
    const XqaQuantType kv_quant_type,
    void* workspace,
    size_t workspace_size);

}  // namespace H256

// Dispatcher Implementation

template <typename T>
Status LaunchXQAKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* past_seq_lens,
    const float* kv_cache_scale,
    const XqaQuantType kv_quant_type,
    void* workspace,
    size_t workspace_size) {
  if (device_prop.major < 8) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA is only supported on Ampere (SM80) or newer GPUs.");
  }

  if (head_size == 256) {
    return H256::LaunchXQAKernelImpl<T>(
        device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size,
        max_seq_len, scale, is_bsnh, past_seq_lens, kv_cache_scale, kv_quant_type, workspace, workspace_size);
  } else if (head_size == 128) {
    return H128::LaunchXQAKernelImpl<T>(
        device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size,
        max_seq_len, scale, is_bsnh, past_seq_lens, kv_cache_scale, kv_quant_type, workspace, workspace_size);
  } else if (head_size == 64) {
    return H64::LaunchXQAKernelImpl<T>(
        device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size,
        max_seq_len, scale, is_bsnh, past_seq_lens, kv_cache_scale, kv_quant_type, workspace, workspace_size);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA only supports head_size=64, 128, or 256. Input has ", head_size);
  }
}

size_t GetXQAScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int head_size,
    int max_seq_len,
    [[maybe_unused]] XqaQuantType kv_quant_type,
    [[maybe_unused]] bool is_bf16) {
  if (device_prop.major < 8) {
    return 0;
  }

  uint32_t nbSeq = static_cast<uint32_t>(batch_size * kv_num_heads);
  // nbSubSeqPerSeq calculation matches computeNbSubSeqPerSeqMHA in mha_impl.cuh
  // ctaTile.x is 256 for all current configurations
  uint32_t nbSubSeqPerSeq = std::min<uint32_t>(
      std::max<uint32_t>(1U, static_cast<uint32_t>(device_prop.multiProcessorCount) / nbSeq),
      (static_cast<uint32_t>(max_seq_len) + 255) / 256);
  uint32_t nbSubSeq = nbSeq * nbSubSeqPerSeq;

  int group_size = num_heads / kv_num_heads;
  // M_TILESIZE: 8 for group_size <= 8, 16 for group_size <= 16, 32 for group_size <= 32
  int m_tilesize = (group_size <= 8) ? 8 : (group_size <= 16 ? 16 : 32);

  // sizeof(SMemWarpRowMax) is 128 (4 * 8 * 4) for all group sizes <= 32
  // sizeof(VecT) is head_size * m_tilesize * 2 (2 bytes per element for fp16/bf16 intermediate results)
  size_t vec_size = static_cast<size_t>(head_size) * m_tilesize * 2;

  size_t scratch_size = 0;
  // 1. rowMax
  scratch_size = roundUp<size_t>(scratch_size, 128);
  scratch_size += 128 * nbSubSeq;
  // 2. rowSum
  scratch_size = roundUp<size_t>(scratch_size, 128);
  scratch_size += 128 * nbSubSeq;
  // 3. scratchBuffers
  scratch_size = roundUp<size_t>(scratch_size, vec_size);
  scratch_size += vec_size * nbSubSeq;

  size_t semaphore_size = nbSeq * sizeof(uint32_t);
  return roundUp<size_t>(semaphore_size, 128) + scratch_size;
}

// Instantiate template for half
template Status LaunchXQAKernel<half>(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* past_seq_lens,
    const float* kv_cache_scale,
    const XqaQuantType kv_quant_type,
    void* workspace,
    size_t workspace_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
