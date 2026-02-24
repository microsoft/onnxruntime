// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define HEAD_ELEMS 256
#define HEAD_DIM_NAMESPACE H256

#include "xqa_loader_fp16_impl.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Explicit instantiation
template Status HEAD_DIM_NAMESPACE::LaunchXQAKernelImpl<half>(
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
