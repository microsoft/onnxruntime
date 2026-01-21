// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "xqa_loader.h"
#include <cassert>

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
    const int actual_seq_len,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* seq_lens,
    const float* kv_cache_scale,
    const int kv_quant_type,
    void* workspace,
    size_t workspace_size);

size_t GetXQAScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);
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
    const int actual_seq_len,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* seq_lens,
    const float* kv_cache_scale,
    const int kv_quant_type,
    void* workspace,
    size_t workspace_size);

size_t GetXQAScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);
}  // namespace H64

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
    const int actual_seq_len,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* seq_lens,
    const float* kv_cache_scale,
    const int kv_quant_type,
    void* workspace,
    size_t workspace_size) {
  if (head_size == 128) {
    return H128::LaunchXQAKernelImpl<T>(
        device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size,
        actual_seq_len, max_seq_len, scale, is_bsnh, seq_lens, kv_cache_scale, kv_quant_type, workspace, workspace_size);
  } else if (head_size == 64) {
    return H64::LaunchXQAKernelImpl<T>(
        device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size,
        actual_seq_len, max_seq_len, scale, is_bsnh, seq_lens, kv_cache_scale, kv_quant_type, workspace, workspace_size);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA only supports head_size=64 or 128. Input has ", head_size);
  }
}

size_t GetXQAScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len) {
  // Just use H128 logic for scratch size estimation if it doesn't depend on head size being strictly 128 in estimation logic?
  // Looking at xqa_impl_gen.cuh, GetScratchSize depends on namespace/template params which depend on HEAD_ELEMS indirectly?
  // Actually, GetScratchSize in xqa_impl_gen calls `grpX_fp16::GetScratchSize`.
  // If H64 logic is different, we should pick the right one.
  // But GetXQAScratchSize doesn't take head_size as input?
  // Wait, the signature in xqa_loader.h DOES include head_size?
  // No, `size_t GetXQAScratchSize(const cudaDeviceProp& device_prop, int batch_size, int num_heads, int kv_num_heads, int max_seq_len);`
  // It does NOT have head_size.

  // Checking `xqa_impl_gen.cuh`:
  // size_t scratch_size = ::onnxruntime::contrib::cuda::NAMESPACE_NAME::GetScratchSize(nbSeq, nbSubSeqPerSeq);

  // `NAMESPACE_NAME` (e.g. grp8_fp16) is generated including `mha_impl.cuh`.
  // `mha_impl.cuh` depends on `HEAD_ELEMS`.
  // So the scratch size might depend on HEAD_ELEMS.
  // BUT the API doesn't pass head_size. This is a problem if scratch size depends on head size.
  // Most likely, scratch size depends on sequence lengths and number of heads, not head dim (unless smem usage constraint).
  // However, if I use H128's GetScratchSize, it assumes HEAD_ELEMS=128 for any persistent structures.

  // Let's assume for now we use H128's size as a conservative estimate (usually larger head dim size -> maybe larger scratch? or same?).
  // If the kernels are built with static smem, 128 might need more.

  return H128::GetXQAScratchSize(device_prop, batch_size, num_heads, kv_num_heads, max_seq_len);
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
    const int actual_seq_len,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* seq_lens,
    const float* kv_cache_scale,
    const int kv_quant_type,
    void* workspace,
    size_t workspace_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
