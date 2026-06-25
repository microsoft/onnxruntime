// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "xqa_loader.h"
#include <cassert>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Forward declarations of LaunchXQAIn8KernelBF16 from H64, H128, H256 namespaces
namespace H64 {
Status LaunchXQAInt8KernelBF16(
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
    void* workspace,
    size_t workspace_size);
}  // namespace H64

namespace H128 {
Status LaunchXQAInt8KernelBF16(
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
    void* workspace,
    size_t workspace_size);
}  // namespace H128

namespace H256 {
Status LaunchXQAInt8KernelBF16(
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
    void* workspace,
    size_t workspace_size);
}  // namespace H256

// Dispatcher for INT8 BF16 query kernel based on head_size
Status LaunchXQAInt8KernelBF16(
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
    void* workspace,
    size_t workspace_size) {
  if (head_size == 256) {
    return H256::LaunchXQAInt8KernelBF16(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, max_seq_len, scale, is_bsnh, past_seq_lens, kv_cache_scale, workspace, workspace_size);
  } else if (head_size == 128) {
    return H128::LaunchXQAInt8KernelBF16(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, max_seq_len, scale, is_bsnh, past_seq_lens, kv_cache_scale, workspace, workspace_size);
  } else if (head_size == 64) {
    return H64::LaunchXQAInt8KernelBF16(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, max_seq_len, scale, is_bsnh, past_seq_lens, kv_cache_scale, workspace, workspace_size);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA INT8 BF16 only supports head_size=64, 128, or 256. Input has ", head_size);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
