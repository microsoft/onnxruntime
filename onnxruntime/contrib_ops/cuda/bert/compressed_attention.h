// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status LaunchCompressedAttentionKernel(
    cudaStream_t stream, T* output, T* present_local_kv, const T* query, const T* local_kv,
    const T* past_local_kv, const int64_t* position_ids,
    const T* compressed_kv, const T* attention_bias, const int64_t* selected_indices,
    const T* head_sink, int batch_size, int num_heads, int sequence_length,
    int head_size, int local_count, int compressed_count, int selected_count,
    int sink_count, int64_t bias_b, int64_t bias_n, int64_t bias_s, int64_t bias_k,
    float scale, bool fixed_cache, int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
